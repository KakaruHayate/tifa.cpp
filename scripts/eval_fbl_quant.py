"""Score a converted FoxBreatheLabeler GGUF against a TextGrid ground truth.

Takes a directory of wav/TextGrid pairs, reads the GGUF (dequantising whatever
dtype it holds), loads the weights into the reference torch model, and runs the
same AP decoding the CLI uses -- so the number this prints is the AP quality the
shipped weights would produce, without needing the C++ integration first.

The BatchNorm the converter folds to scale/shift is reconstructed as a
BatchNorm1d with mean 0 and var 1-eps, which makes its eval-time transform
exactly `x * scale + shift`.
"""
import argparse
import importlib.util
import pathlib
import sys
import time

import numpy as np
import soundfile as sf
import scipy.signal
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import eval_common as S  # noqa: E402




import gguf  # noqa: E402

SR = 44100
HOP = 882
PAD = 71
FRAME = HOP / SR


def torch_key(name: str) -> str:
    k = name
    if k.startswith("fbl."):
        k = "model." + k[len("fbl."):]
    k = k.replace(".att.q.", ".att.to_q.")
    k = k.replace(".att.kv.", ".att.to_kv.")
    k = k.replace(".att.out.", ".att.to_out.0.")
    k = k.replace(".in.", ".inlinear.")
    k = k.replace(".out.", ".outlinear.")
    k = k.replace(".conv.pw1.", ".conv.pointwise_conv1.")
    k = k.replace(".conv.pw2.", ".conv.pointwise_conv2.")
    k = k.replace(".conv.dw.", ".conv.depthwise_conv.")
    k = k.replace(".conv.bn.scale", ".conv.norm.weight")
    k = k.replace(".conv.bn.shift", ".conv.norm.bias")
    k = k.replace(".ffn1.1.", ".ffn1.ln1.")
    k = k.replace(".ffn1.2.", ".ffn1.ln2.")
    k = k.replace(".ffn2.1.", ".ffn2.ln1.")
    k = k.replace(".ffn2.2.", ".ffn2.ln2.")
    return k


def load_gguf(path):
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        data = t.data
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            data = gguf.dequantize(data, t.tensor_type)
        out[torch_key(t.name)] = torch.from_numpy(np.asarray(data, dtype=np.float32))
    return out


def load_model(gguf_path, cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text(encoding="utf-8"))
    model = Fbl4(cfg, output_size=1)
    sd = load_gguf(gguf_path)

    # reconstruct the folded BatchNorm: mean 0, var 1-eps -> x * scale + shift
    eps = 1e-5
    for i in range(int(cfg["model_arg"]["num_layers"])):
        base = f"model.enc.{i}.conv.norm"
        if base + ".weight" in sd:
            sd[base + ".running_mean"] = torch.zeros_like(sd[base + ".weight"])
            sd[base + ".running_var"] = torch.full_like(sd[base + ".weight"], 1.0 - eps)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing={list(missing)[:4]} "
                         f"unexpected={list(unexpected)[:4]}")
    model.eval()
    return model, cfg


@torch.no_grad()
def ap_prob(model, y):
    # Fbl4 frames the waveform itself (pad + unfold), exactly like the exported
    # graph -- feeding it frames would frame them twice.
    out = model(torch.from_numpy(y[None, :].astype(np.float32)))
    return torch.sigmoid(out).squeeze().numpy()       # [T]


def decode(prob, threshold=0.4, max_gap=5, min_frames=4):
    events, start, gap = [], None, 0
    n = len(prob)
    for i in range(n):
        if prob[i] >= threshold:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            if gap < max_gap:
                gap += 1
            else:
                end = i - gap - 1
                if end > start and (end - start) >= min_frames:
                    events.append((start * FRAME, end * FRAME))
                start, gap = None, 0
    if start is not None and (n - start) >= min_frames:
        events.append((start * FRAME, (n - 1) * FRAME))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", required=True,
                    help="the FBL config.yaml the checkpoint was trained with")
    ap.add_argument("--wavs", required=True,
                    help="directory of *.wav with a sibling *.TextGrid ground truth")
    ap.add_argument("--out", default=None,
                    help="where to write the score JSON (default: out_<tag>.json)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()

    size = pathlib.Path(args.gguf).stat().st_size / 1e6
    model, _ = load_model(args.gguf, args.config)
    print(f"[{args.tag}] {size:.1f} MB, threshold={args.threshold}")

    wav_dir = pathlib.Path(args.wavs)
    wavs = sorted(wav_dir.glob("*.wav"))[: args.limit]
    acc = {}
    t0 = time.time()
    n_ok = 0
    for wav in wavs:
        tg = wav.with_suffix(".TextGrid")
        if not tg.exists():
            continue
        y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if sr != SR:
            g = np.gcd(int(sr), SR)
            y = scipy.signal.resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
        ref_i = S.truth_intervals(tg)
        n_frames = int(round(len(y) / SR * S.FPS))
        ref = S.to_frames(ref_i, n_frames)
        events = decode(ap_prob(model, y), args.threshold)
        hyp = ["V"] * n_frames
        for a, b in events:
            for f in range(int(round(a * S.FPS)), min(n_frames, int(round(b * S.FPS)))):
                hyp[f] = "AP"
        S.merge_scores(acc.setdefault("frame", {}), S.frame_scores(ref, hyp))
        S.merge_scores(acc.setdefault("event", {}),
                       S.event_scores(S.events_of(ref), S.events_of(hyp)))
        n_ok += 1
    print(f"[{args.tag}] files={n_ok} wall={time.time() - t0:.0f}s")
    print(f"     frame: {S.fmt(acc['frame'])}")
    print(f"     event: {S.fmt(acc['event'])}")
    out = args.out or f"out_quant_{args.tag}.json"
    S.dump(out, {"tag": args.tag, "mb": size, "files": n_ok, **acc})


if __name__ == "__main__":
    main()
