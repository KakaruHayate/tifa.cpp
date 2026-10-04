#!/usr/bin/env python3
"""Convert a FoxBreatheLabeler checkpoint into a GGUF for tifa.cpp's `breathe`.

FoxBreatheLabeler (github.com/autumn-DL/FoxBreatheLabeler) publishes PyTorch
checkpoints, not ONNX: the release archive holds `config.yaml` + `model.ckpt`.
The model is a CVNT conformer over *raw waveform frames* -- unlike BreathLab it
has no mel/F0 front end, so the framing happens inside the graph:

    pad(x, (win-hop)//2, (win-hop+1)//2)
    frame -> [T, spec_win]  -> inlinear(spec_win, dim)
    N x conform_blocke_full(dim, kernel, heads x dim_head)
    final_norm -> outlinear(dim, 1) -> sigmoid

Usage:
    python convert_fbl_to_gguf.py --model-dir model_folder -o breath-fbl.gguf
    python convert_fbl_to_gguf.py --ckpt model.ckpt --config config.yaml -o out.gguf
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np
import yaml

try:
    import gguf  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install gguf") from exc

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install torch (CPU build is enough)") from exc

BREATH_ARCH = "breath-ap"
ARCH_VERSION = 1

log = logging.getLogger("convert_fbl")


def f32_only(name: str) -> bool:
    """Tensors ggml needs in F32: biases, norms and the depthwise kernel."""
    return (name.endswith(".bias")
            or ".norm" in name
            or ".bn." in name
            or ".dw." in name
            or name.endswith(".weight") and ("norm" in name))


def rename(key: str) -> str:
    """`model.enc.0.att.to_out.0.weight` -> `fbl.enc.0.att.out.weight`."""
    k = key
    if k.startswith("model."):
        k = "fbl." + k[len("model."):]
    # the container names (att./conv.) are already in the path, so only the
    # leaf is rewritten
    k = k.replace(".to_out.0.", ".out.")
    k = k.replace(".to_q.", ".q.")
    k = k.replace(".to_kv.", ".kv.")
    k = k.replace(".inlinear.", ".in.")
    k = k.replace(".outlinear.", ".out.")
    k = k.replace(".pointwise_conv1.", ".pw1.")
    k = k.replace(".pointwise_conv2.", ".pw2.")
    k = k.replace(".depthwise_conv.", ".dw.")
    k = k.replace(".ffn1.ln1.", ".ffn1.1.")
    k = k.replace(".ffn1.ln2.", ".ffn1.2.")
    k = k.replace(".ffn2.ln1.", ".ffn2.1.")
    k = k.replace(".ffn2.ln2.", ".ffn2.2.")
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help="folder holding config.yaml + *.ckpt")
    ap.add_argument("--ckpt")
    ap.add_argument("--config")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--name", default="breath-fbl")
    ap.add_argument("--dtype", choices=["f16", "f32", "q8_0", "q4_0"], default="f16")
    ap.add_argument("--ap-threshold", type=float, default=0.4,
                    help="AP probability threshold (textgrid_add_ap.py default)")
    ap.add_argument("--ap-dur", type=float, default=0.08,
                    help="shortest AP kept, seconds")
    ap.add_argument("--max-gap", type=int, default=5,
                    help="frames of sub-threshold dip tolerated inside an AP run")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else None
    cfg_path = pathlib.Path(args.config) if args.config else None
    if args.model_dir:
        d = pathlib.Path(args.model_dir)
        ckpt = ckpt or next(iter(sorted(d.glob("*.ckpt"))), None)
        cfg_path = cfg_path or (d / "config.yaml")
    if ckpt is None or not ckpt.exists():
        raise SystemExit("error: no checkpoint found (--ckpt or --model-dir)")
    if cfg_path is None or not cfg_path.exists():
        raise SystemExit("error: no config.yaml found")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    arg = cfg["model_arg"]
    dim = int(arg["encoder_conform_dim"])
    layers = int(arg["num_layers"])
    heads = int(arg["encoder_conform_attention_heads"])
    head_dim = int(arg["encoder_conform_attention_heads_dim"])
    kernel = int(arg["encoder_conform_kernel_size"])
    spec_win = int(cfg["spec_win"])
    hop = int(cfg["hop_size"])
    sr = int(cfg["audio_sample_rate"])

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = state.get("model", state)
    weights = {}
    for key, tensor in state.items():
        if key.endswith("num_batches_tracked"):
            continue
        weights[rename(key)] = tensor.detach().cpu().numpy().astype(np.float32)

    # Fold each conform_conv's BatchNorm1d into a per-channel scale/shift.  At
    # inference the running statistics are constant, so (x - mean)/sqrt(var+eps)
    # * gamma + beta collapses to one multiply-add and the graph needs no
    # variance op (ggml has none).
    folded = {}
    for key in list(weights):
        if not key.endswith(".conv.norm.weight"):
            continue
        base = key[: -len(".weight")]                  # ...conv.norm
        out_base = base[: -len("norm")] + "bn"         # ...conv.bn
        g = weights.pop(base + ".weight")
        b = weights.pop(base + ".bias")
        mean = weights.pop(base + ".running_mean")
        var = weights.pop(base + ".running_var")
        scale = g / np.sqrt(var + 1e-5)
        folded[out_base + ".scale"] = scale.astype(np.float32)
        folded[out_base + ".shift"] = (b - mean * scale).astype(np.float32)
    weights.update(folded)

    # sanity: the head must be a single AP logit
    out_w = weights["fbl.out.weight"]
    if out_w.shape[0] != 1:
        raise SystemExit(
            f"error: this checkpoint has {out_w.shape[0]} output classes; the "
            "AP-only release (0.2/20ms) is the supported one")

    fps = sr / hop
    min_frames = max(1, int(round(args.ap_dur * fps)))

    writer = gguf.GGUFWriter(args.output, BREATH_ARCH)
    writer.add_name(args.name)
    writer.add_string("general.version", str(ARCH_VERSION))
    writer.add_string("breath.source", "FoxBreatheLabeler")
    # Discriminator for the loader: BreathLab carries mel+F0 features, FBL
    # frames the raw waveform itself.
    writer.add_string("breath.model_kind", "fbl")
    writer.add_int32("fbl.spec_win", spec_win)
    writer.add_int32("fbl.hop", hop)
    writer.add_int32("fbl.sample_rate", sr)
    writer.add_float32("fbl.fps", float(fps))
    writer.add_int32("fbl.dim", dim)
    writer.add_int32("fbl.layers", layers)
    writer.add_int32("fbl.heads", heads)
    writer.add_int32("fbl.head_dim", head_dim)
    writer.add_int32("fbl.kernel", kernel)
    writer.add_float32("fbl.threshold", float(args.ap_threshold))
    writer.add_int32("fbl.min_dur_frames", min_frames)
    writer.add_int32("fbl.max_gap", int(args.max_gap))

    dtype_map = {
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
    }
    block = {gguf.GGMLQuantizationType.Q8_0: 32, gguf.GGMLQuantizationType.Q4_0: 32}
    qtype = dtype_map[args.dtype]

    total = quantized = 0
    for key in sorted(weights):
        arr = np.ascontiguousarray(weights[key], dtype=np.float32)
        eff = gguf.GGMLQuantizationType.F32
        if qtype != gguf.GGMLQuantizationType.F32 and not f32_only(key):
            if qtype == gguf.GGMLQuantizationType.F16:
                arr, eff = arr.astype(np.float16), qtype
                quantized += 1
            elif arr.ndim == 2 and arr.shape[-1] % block[qtype] == 0:
                # only the 2-D matmul weights are worth quantizing
                arr, eff = gguf.quants.quantize(arr, qtype), qtype
                quantized += 1
            else:
                arr, eff = arr.astype(np.float16), gguf.GGMLQuantizationType.F16
        writer.add_tensor(key, arr, raw_dtype=eff)
        total += arr.nbytes
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    log.info("wrote %s: %d tensors (%d quantized), %.1f MB",
             args.output, len(weights), quantized, total / 1e6)
    log.info("dim=%d layers=%d heads=%dx%d kernel=%d spec_win=%d hop=%d sr=%d fps=%.0f",
             dim, layers, heads, head_dim, kernel, spec_win, hop, sr, fps)
    log.info("threshold=%.2f min_dur=%d frames max_gap=%d frames",
             args.ap_threshold, min_frames, args.max_gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
