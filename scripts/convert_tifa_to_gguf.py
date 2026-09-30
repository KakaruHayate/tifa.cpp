#!/usr/bin/env python3
"""Convert a TIFA PyTorch checkpoint directory into a single GGUF file.

Usage:
    python convert_tifa_to_gguf.py --model-dir TIFA-1.0-ST -o tifa-1.0-st.gguf

Inputs:
    <model-dir>/model.pt        required; TIFA release checkpoint
    <model-dir>/config.yaml     required; model + inference configuration
    <model-dir>/vocabulary.json required; symbol table

Output:
    one GGUF file containing
      * architecture metadata         (general.architecture = "tifa-fa")
      * model / backbone hyperparams  (tifa.model.*, tifa.backbone.*)
      * feature + inference metadata  (tifa.features.*, tifa.inference.*)
      * the vocabulary, G2P config and global/stop symbols, packed as JSON
        (tifa.vocab.json)
      * every parameter tensor, keyed by its PyTorch module path with the
        training `model.` prefix stripped (e.g. backbone.layers.0.attn.jattn.token_qkv.weight)

The converter is deliberately self-contained: it needs only torch, numpy, gguf
and pyyaml, does not quantize by default, and does not import the TIFA package.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
from typing import Any

import numpy as np
import torch
import yaml

try:
    import gguf  # type: ignore
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "error: the 'gguf' Python package is required "
        "(pip install -r scripts/requirements.txt).\n"
    )
    raise

log = logging.getLogger("convert_tifa_to_gguf")

TIFA_ARCH = "tifa-fa"
TIFA_ARCH_VERSION = 1

# ---------------------------------------------------------------------------
# dtype policy
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "q6_k": gguf.GGMLQuantizationType.Q6_K,
}

_BLOCK = {
    gguf.GGMLQuantizationType.Q4_0: 32,
    gguf.GGMLQuantizationType.Q8_0: 32,
    gguf.GGMLQuantizationType.Q4_K: 256,
    gguf.GGMLQuantizationType.Q5_K: 256,
    gguf.GGMLQuantizationType.Q6_K: 256,
}


def f32_only(name: str) -> bool:
    """Tensors that must stay F32 for numerical stability or kernel support."""
    if name.endswith(".bias"):
        return True
    if "norm" in name and name.endswith(".weight"):
        return True          # RMSNorm weights (incl. token_q_norm / x_norm / c.norm)
    if name.endswith(".scale"):
        return True          # LayerScale
    if ".dw." in name and name.endswith(".weight"):
        return True          # CgMLP depthwise conv
    if "dw_conv" in name and name.endswith(".weight"):
        return True          # PJAC merge depthwise conv
    return False


def resolve_dtype(name: str, base: str, rules: list[tuple[str, str]]) -> gguf.GGMLQuantizationType:
    """Per-tensor dtype: F32-only roles win, then explicit rules, then base.

    F32-only roles (norms, LayerScale, depthwise kernels, biases) are checked
    FIRST: a rule that matches them would otherwise hand the graph an F16/F16
    tensor where a binary op expects F32 (ggml's CPU backend aborts with
    "binary_op: unsupported types ... src1: f16").
    """
    import fnmatch

    if f32_only(name):
        return gguf.GGMLQuantizationType.F32
    for pattern, dtype in rules:
        if fnmatch.fnmatch(name, pattern):
            return _DTYPE_MAP[dtype]
    return _DTYPE_MAP[base]


def quantize(arr: np.ndarray, qtype: gguf.GGMLQuantizationType) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    """Quantize when the requested type is a block quant; F16 is a cast."""
    if qtype == gguf.GGMLQuantizationType.F32:
        return arr.astype(np.float32), qtype
    if qtype == gguf.GGMLQuantizationType.F16:
        return arr.astype(np.float16), qtype
    if arr.ndim != 2:
        log.warning("skipping quantization of %d-D tensor", arr.ndim)
        return arr.astype(np.float16), gguf.GGMLQuantizationType.F16
    block = _BLOCK[qtype]
    if arr.shape[-1] % block != 0:
        log.warning("skipping quantization: last dim %d not a multiple of %d",
                    arr.shape[-1], block)
        return arr.astype(np.float16), gguf.GGMLQuantizationType.F16
    qs = gguf.quants.quantize(arr.astype(np.float32), qtype)
    return qs, qtype


# ---------------------------------------------------------------------------
# expected key audit
# ---------------------------------------------------------------------------

def expected_keys(cfg: dict[str, Any]) -> set[str]:
    """Every tensor name the C++ binder expects (post prefix-strip)."""
    m = cfg["model"]
    bb = m["backbone"]["kwargs"]
    keys: set[str] = {"token_embedding.weight",
                      "backbone.audio_input_proj.weight", "backbone.audio_input_proj.bias",
                      "backbone.text_input_proj.weight", "backbone.text_input_proj.bias"}

    for i in range(bb["num_layers"]):
        p = f"backbone.layers.{i}"
        for stream in ("x", "token"):
            for ffn in ("ffn1", "ffn2"):
                for part in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"):
                    keys.add(f"{p}.{ffn}_{stream}.{part}")
                keys.add(f"{p}.norm_{ffn}_{stream}.weight")
                keys.add(f"{p}.layer_scale_{ffn}_{stream}.scale")
            keys.add(f"{p}.attn.c_{stream}.pw1.weight")
            keys.add(f"{p}.attn.c_{stream}.pw1.bias")
            keys.add(f"{p}.attn.c_{stream}.norm.weight")
            keys.add(f"{p}.attn.c_{stream}.dw.weight")
            keys.add(f"{p}.attn.c_{stream}.dw.bias")
            keys.add(f"{p}.attn.c_{stream}.pw2.weight")
            keys.add(f"{p}.attn.c_{stream}.pw2.bias")
            keys.add(f"{p}.attn.jattn.{stream}_norm.weight")
            keys.add(f"{p}.attn.jattn.{stream}_qkv.weight")
            keys.add(f"{p}.attn.jattn.{stream}_qkv.bias")
            keys.add(f"{p}.attn.jattn.{stream}_q_norm.weight")
            keys.add(f"{p}.attn.jattn.{stream}_k_norm.weight")
            keys.add(f"{p}.attn.jattn.{stream}_out.weight")
            keys.add(f"{p}.attn.jattn.{stream}_out.bias")
            keys.add(f"{p}.attn.merge_linear_{stream}.weight")
            keys.add(f"{p}.attn.merge_linear_{stream}.bias")
            keys.add(f"{p}.attn.merge_dw_conv_{stream}.weight")
            keys.add(f"{p}.attn.merge_dw_conv_{stream}.bias")
        keys.add(f"{p}.attn.c_norm_x.weight")
        keys.add(f"{p}.attn.c_norm_token.weight")
        keys.add(f"{p}.layer_scale_jpac_x.scale")
        keys.add(f"{p}.layer_scale_jpac_token.scale")

    keys.add("backbone.output_norm_x.weight")
    keys.add("backbone.output_norm_token.weight")
    for stream in ("x", "token"):
        keys.add(f"backbone.output_proj_{stream}.weight")
        keys.add(f"backbone.output_proj_{stream}.bias")
    return keys


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def load_state_dict(path: pathlib.Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        sd = dict(ckpt)
    else:  # pragma: no cover - a bare nn.Module checkpoint
        sd = dict(ckpt.state_dict())
    # EMA weights (if present) win: the release ships merged weights already.
    if isinstance(ckpt, dict) and ckpt.get("ema_state_dict"):
        ema = {k: v for k, v in ckpt["ema_state_dict"].items() if k in sd}
        sd.update(ema)
    return sd


def strip_prefix(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    prefixes = {k.split(".")[0] for k in sd}
    for prefix in ("model",):
        if prefixes == {prefix}:
            return {k[len(prefix) + 1:]: v for k, v in sd.items()}, prefix
    return sd, ""


def add_metadata(writer: gguf.GGUFWriter, cfg: dict[str, Any], model_dir: pathlib.Path) -> None:
    m = cfg["model"]
    bb = m["backbone"]["kwargs"]
    inf = cfg["inference"]
    feat = inf["features"]
    spec = feat["spectrogram"]

    writer.add_name(cfg.get("name", model_dir.name))
    writer.add_string("general.version", str(TIFA_ARCH_VERSION))

    def put(key: str, value: Any) -> None:
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_int32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        else:
            writer.add_string(key, str(value))

    put("tifa.model.arch", m.get("arch", "ForcedAlignmentModel"))
    put("tifa.model.max_vocab_size", int(m["max_vocab_size"]))
    put("tifa.model.in_dim", int(m["in_dim"]))
    put("tifa.model.embedding_dim", int(m["embedding_dim"]))
    put("tifa.model.out_dim", int(m["out_dim"]))

    put("tifa.backbone.cls", m["backbone"].get("cls", ""))
    for key in ("dim", "num_layers", "num_heads", "head_dim"):
        put(f"tifa.backbone.{key}", int(bb[key]))
    for key in ("attn_type", "ffn_type"):
        put(f"tifa.backbone.{key}", bb.get(key, "joint" if key == "attn_type" else "glu"))
    for key in ("qk_norm", "use_rope", "use_ls", "use_out_norm", "skip_first_ffn", "skip_out_ffn"):
        put(f"tifa.backbone.{key}", bool(bb.get(key, key in ("qk_norm", "use_rope", "use_ls", "use_out_norm"))))
    put("tifa.backbone.theta", float(bb.get("theta", 10000.0)))
    put("tifa.backbone.c_kernel_size_token", int(bb.get("c_kernel_size_token", 7)))
    put("tifa.backbone.m_kernel_size_token", int(bb.get("m_kernel_size_token", 5)))
    put("tifa.backbone.c_kernel_size_x", int(bb.get("c_kernel_size_x", 31)))
    put("tifa.backbone.m_kernel_size_x", int(bb.get("m_kernel_size_x", 31)))

    put("tifa.features.audio_sample_rate", int(feat["audio_sample_rate"]))
    put("tifa.features.hop_size", int(feat["hop_size"]))
    put("tifa.features.fft_size", int(feat["fft_size"]))
    put("tifa.features.win_size", int(feat.get("win_size", feat["fft_size"])))
    put("tifa.features.num_bins", int(spec["num_bins"]))
    put("tifa.features.fmin", float(spec.get("fmin", 0.0)))
    put("tifa.features.fmax", float(spec.get("fmax", 8000.0)))
    put("tifa.features.clip_val", float(spec.get("clip_val", 1e-5)))

    g2p = inf.get("g2p")
    vocab_path = model_dir / "vocabulary.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf8"))
    merged = []
    if g2p:
        merged = g2p.get("vocabulary", {}).get("merged_groups", []) or []
    globals_ = ["AP", "SP", "EP", "GS", "sil", "br", "pau"]
    stops = ["SP", "sil", "pau"]
    if g2p:
        vcfg = g2p.get("vocabulary", {})
        globals_ = vcfg.get("global_symbols", globals_)
        stops = vcfg.get("stop_symbols", stops)
        merged = vcfg.get("merged_groups", merged)

    packed = {
        "symbols": vocab["symbols"],
        "global_symbols": globals_,
        "stop_symbols": stops,
        "merged_groups": merged,
        "g2p": g2p,
    }
    writer.add_string("tifa.vocab.json", json.dumps(packed, ensure_ascii=False, separators=(",", ":")))

    put("tifa.inference.skip_penalty", float(inf.get("skip_penalty", 0.5)))
    put("tifa.inference.score_unit", str(inf.get("score_unit", "levenshtein")))


def convert(model_dir: pathlib.Path, output: pathlib.Path, *, base_dtype: str,
            rules: list[tuple[str, str]], strict: bool) -> None:
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf8"))
    sd, prefix = strip_prefix(load_state_dict(model_dir / "model.pt"))
    if prefix:
        log.info("stripped '%s.' prefix from %d tensors", prefix, len(sd))

    expect = expected_keys(cfg)
    missing = sorted(expect - set(sd))
    extra = sorted(set(sd) - expect)
    if missing:
        log.error("checkpoint is missing %d expected tensors", len(missing))
        for name in missing[:20]:
            log.error("  missing: %s", name)
        if strict:
            raise SystemExit("missing tensors (use --no-strict to continue)")
    if extra:
        log.warning("checkpoint has %d unexpected tensors:", len(extra))
        for name in extra[:20]:
            log.warning("  extra: %s", name)
        if strict:
            raise SystemExit("unexpected tensors (use --no-strict to continue)")

    writer = gguf.GGUFWriter(output, TIFA_ARCH)
    add_metadata(writer, cfg, model_dir)

    total_bytes = 0
    hist: dict[str, int] = {}
    for name in sorted(sd):
        tensor = sd[name].detach().cpu().float().contiguous()
        arr = tensor.numpy()
        qtype = resolve_dtype(name, base_dtype, rules)
        payload, eff = quantize(arr, qtype)
        writer.add_tensor(name, payload, raw_dtype=eff)
        total_bytes += payload.nbytes
        hist[eff.name] = hist.get(eff.name, 0) + 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    log.info("wrote %s (%.1f MB, %d tensors: %s)",
             output, output.stat().st_size / 1e6, len(sd),
             ", ".join(f"{k}={v}" for k, v in sorted(hist.items())))


def parse_rules(path: pathlib.Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf8"))
    out: list[tuple[str, str]] = []
    for entry in data.get("rules", []):
        out.append((entry["pattern"], entry["dtype"].lower()))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=pathlib.Path, required=True,
                        help="directory with model.pt, config.yaml, vocabulary.json")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True,
                        help="output .gguf path")
    parser.add_argument("--dtype", choices=sorted(_DTYPE_MAP), default="f16",
                        help="base dtype for 2-D weights (default: f16)")
    parser.add_argument("--quant-config", type=pathlib.Path, default=None,
                        help="JSON with per-tensor rules: {\"rules\":[{\"pattern\":...,\"dtype\":...}]}")
    parser.add_argument("--no-strict", dest="strict", action="store_false",
                        help="warn instead of failing on key-set mismatches")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    convert(args.model_dir, args.output, base_dtype=args.dtype,
            rules=parse_rules(args.quant_config), strict=args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
