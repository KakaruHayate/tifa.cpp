#!/usr/bin/env python3
"""Dump TIFA reference activations for the ggml port (golden data).

Runs the *official* PyTorch implementation (openvpi/TIFA) on one audio file
with a known phone sequence and writes every intermediate stage to disk so the
C++ port can be compared stage by stage:

    mel.npy               [T, 80]      log-mel spectrogram (model rate)
    tokens.npy            [N]          int64 token ids
    frame_features.npy    [T, 256]
    token_features.npy    [N, 256]
    frame_logits.npy      [T, 256]
    token_logits.npy      [N, 256]
    similarity.npy        [T, N]       cross cosine similarity
    spans.npy             [N, 2]       Viterbi spans in *frames*
    agreement.npy         []           mean token probability
    meta.json                          shapes, config, phone labels

Usage:
    python dump_tifa_reference.py --tifa-root ../refs/TIFA \
        --model-dir ../models/TIFA-1.0-ST --audio sample.wav \
        --phones "AP zh e n" -l zh --out-dir golden/ --audio-out sample_48k.wav
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import soundfile as sf
import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tifa-root", type=pathlib.Path, required=True)
    ap.add_argument("--model-dir", type=pathlib.Path, required=True)
    ap.add_argument("--audio", type=pathlib.Path, required=True)
    ap.add_argument("--phones", type=str, required=True,
                    help="space separated phone labels (language-resolved or bare)")
    ap.add_argument("-l", "--language", type=str, default=None)
    ap.add_argument("--skip-penalty", type=float, default=0.5)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--audio-out", type=pathlib.Path, default=None,
                    help="also write the resampled mono wav used as input")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.tifa_root.resolve()))
    from inference.api import load_inference_model          # noqa: E402
    from lib.config.schema import ConfigurationScope        # noqa: E402
    from modules.decoding import decode_alignment_flat      # noqa: E402
    from modules.functional import cross_cosine_similarity  # noqa: E402

    backend, vocabulary, _ = load_inference_model(args.model_dir / "model.pt",
                                                  scope=ConfigurationScope.FA)
    backend.eval()

    # ---- audio (resample to the model rate) --------------------------------
    import librosa
    wav, sr = librosa.load(str(args.audio), sr=None, mono=True)
    target = backend.sample_rate
    if sr != target:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target)
    if args.audio_out is not None:
        args.audio_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.audio_out), wav, target, subtype="PCM_16")

    waveform = torch.from_numpy(wav).float()[None]
    duration = torch.tensor([len(wav) / target], dtype=torch.float32)

    # ---- tokens ------------------------------------------------------------
    languages = [args.language] if args.language else []
    phones = args.phones.split()
    tokens = []
    for ph in phones:
        resolved = vocabulary.resolve(ph, languages)
        if resolved is None and args.language:
            resolved = vocabulary.resolve(ph, [])
        if resolved is None:
            raise SystemExit(f"phone {ph!r} not in vocabulary")
        tokens.append(int(resolved[0]))
    tokens_t = torch.tensor([tokens], dtype=torch.int64)

    # ---- forward -----------------------------------------------------------
    spec = backend.spectrogram(waveform, duration)

    layer_x, layer_token = [], []
    handles = []
    for mod in backend.model.backbone.layers:
        def hook(_m, _inp, out, _i=len(layer_x)):
            out_x, out_tok = (out if isinstance(out, tuple) else (out, None))
            layer_x.append(out_x.detach())
            layer_token.append(out_tok.detach() if out_tok is not None else None)
        handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        frame_feats, frame_logits, token_feats, token_logits = backend.model(
            spec.features, tokens_t, spec.mask, tokens_t != 0)
        similarity = cross_cosine_similarity(frame_feats, token_feats)
        spans = decode_alignment_flat(
            similarity,
            spec.mask.sum(dim=-1),
            (tokens_t != 0).sum(dim=-1),
            groups=None,
            skip_penalty=args.skip_penalty,
        )
        probs = torch.softmax(token_logits.float(), dim=-1)
        token_prob = probs.gather(-1, tokens_t.unsqueeze(-1)).squeeze(-1)
        agreement = token_prob.mean(dim=-1)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def save(name: str, tensor: torch.Tensor) -> None:
        arr = tensor.detach().cpu().float().numpy()
        np.save(out / name, arr)
        return arr

    # ---- layer-0 sub-block trace (matches ops_jebf.cpp taps) --------------
    taps = {}
    with torch.no_grad():
        bb = backend.model.backbone
        nmask = (tokens_t != 0)
        x = bb.audio_input_proj(spec.features) * spec.mask.unsqueeze(-1)
        tok = bb.text_input_proj(backend.model.token_embedding(tokens_t)) * nmask.unsqueeze(-1)
        L0 = bb.layers[0]
        tok = L0.layer_scale_ffn1_token(L0.ffn1_token(L0.norm_ffn1_token(tok))) + tok
        x   = L0.layer_scale_ffn1_x(L0.ffn1_x(L0.norm_ffn1_x(x))) + x
        taps["ffn1_token"] = tok.clone()
        taps["ffn1_x"] = x.clone()
        x = x * spec.mask.unsqueeze(-1)
        tok = tok * nmask.unsqueeze(-1)
        j = L0.attn.jattn
        tight_t = j.token_norm(tok); tight_x = j.x_norm(x)
        taps["attn_norm_token"] = tight_t.clone(); taps["attn_norm_x"] = tight_x.clone()
        taps["qkv_token"] = j.token_qkv(tight_t).clone(); taps["qkv_x"] = j.x_qkv(tight_x).clone()
        tq, tk, tv, xq, xk, xv = j._project_qkv(tok, x)
        tq, tk, xq, xk = j._apply_rope(tq, tk, xq, xk)
        taps["q_token"] = tq[0].permute(1, 0, 2).reshape(tq.shape[2], -1).clone()
        taps["k_token"] = tk[0].permute(1, 0, 2).reshape(tk.shape[2], -1).clone()
        taps["v_token"] = tv[0].permute(1, 0, 2).reshape(tv.shape[2], -1).clone()
        taps["q_x"] = xq[0].permute(1, 0, 2).reshape(xq.shape[2], -1).clone()
        taps["k_x"] = xk[0].permute(1, 0, 2).reshape(xk.shape[2], -1).clone()
        taps["v_x"] = xv[0].permute(1, 0, 2).reshape(xv.shape[2], -1).clone()
        a_tok, a_x = L0.attn.jattn(tok, x, spec.mask, nmask)
        taps["attn_token"] = a_tok.clone()
        taps["attn_x"] = a_x.clone()
        c_tok = L0.attn.c_token(L0.attn.c_norm_token(tok))
        c_x = L0.attn.c_x(L0.attn.c_norm_x(x))
        m_tok = torch.cat([a_tok, c_tok], dim=-1)
        m_x = torch.cat([a_x, c_x], dim=-1)
        m_tok = L0.attn.merge_dw_conv_token(m_tok.transpose(1, 2)).transpose(1, 2) + m_tok
        m_tok = L0.attn.merge_linear_token(m_tok)
        m_x = L0.attn.merge_dw_conv_x(m_x.transpose(1, 2)).transpose(1, 2) + m_x
        m_x = L0.attn.merge_linear_x(m_x)
        taps["pjac_token"] = L0.layer_scale_jpac_token(m_tok)
        taps["pjac_x"] = L0.layer_scale_jpac_x(m_x)

    for h in handles:
        h.remove()
    for key, val in taps.items():
        # qkv taps are already batch-indexed in the trace above.
        save(f"tap0_{key}.npy", val[0] if val.ndim == 4 else val)
    for i, (lx, lt) in enumerate(zip(layer_x, layer_token)):
        save(f"layer{i + 1}_x.npy", lx[0])
        save(f"layer{i + 1}_token.npy", lt[0])
    save("mel.npy", spec.features[0])
    np.save(out / "tokens.npy", tokens_t[0].numpy())
    save("frame_features.npy", frame_feats[0])
    save("token_features.npy", token_feats[0])
    save("frame_logits.npy", frame_logits[0])
    save("token_logits.npy", token_logits[0])
    save("similarity.npy", similarity[0])
    np.save(out / "spans.npy", spans[0].numpy())
    save("agreement.npy", agreement)

    meta = {
        "sample_rate": target,
        "timestep": backend.timestep,
        "num_frames": int(spec.features.shape[1]),
        "num_tokens": int(len(tokens)),
        "phones": phones,
        "language": args.language,
        "skip_penalty": args.skip_penalty,
        "mel_shape": list(spec.features[0].shape),
        "mask_true": int(spec.mask.sum().item()),
        "agreement": float(agreement.item()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
