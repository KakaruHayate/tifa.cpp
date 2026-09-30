#!/usr/bin/env python3
"""Convert a BreathLab ONNX breath/AP detector into a single GGUF file.

Usage:
    python convert_breath_to_gguf.py --model-dir ../../breathlab/models_dml \\
        --name v5_24k -o breath-v5-24k.gguf

Inputs:
    <model-dir>/<name>.onnx       required; exported BreathLab checkpoint
    <model-dir>/<name>.meta.json  required; feature / postprocess metadata

Output:
    one GGUF file containing
      * architecture metadata        (general.architecture = "breath-ap")
      * graph hyper-parameters       (breath.model.*)
      * feature + audio settings     (breath.feature.*, breath.audio.*)
      * post-processing constants    (breath.fps, breath.threshold,
                                      breath.postprocess.*, breath.eval.*,
                                      breath.mean/std, breath.head.*)
      * every Conv / MatMul / LayerNormalization weight, keyed by its position
        in the graph with the exporter's node prefixes stripped
        (e.g. blocks.0.attn.qkv.weight, front.conv.0.weight)

The converter is self-contained: it needs only numpy, onnx and gguf and does
not import the BreathLab package.  Tensors stay F32 by default (the model is
only ~5.2 M params); `--dtype f16` halves the file at a small parity cost and
never touches norms, biases or depthwise kernels (see `f32_only`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import re
import sys
from typing import Any

import numpy as np

try:
    import gguf  # type: ignore
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "error: the 'gguf' Python package is required "
        "(pip install -r scripts/requirements.txt).\n"
    )
    raise

try:
    import onnx  # type: ignore
    from onnx import numpy_helper  # type: ignore
except ImportError:  # pragma: no cover
    sys.stderr.write("error: the 'onnx' Python package is required (pip install onnx).\n")
    raise

log = logging.getLogger("convert_breath_to_gguf")

BREATH_ARCH = "breath-ap"
BREATH_ARCH_VERSION = 1

# ---------------------------------------------------------------------------
# dtype policy
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
}


def f32_only(name: str) -> bool:
    """Tensors that must stay F32 for numerical stability or kernel support.

    Mirrors scripts/convert_tifa_to_gguf.py: every bias, every LayerNorm
    weight and every depthwise kernel.  The depthwise kernels are read as
    `float *` by ggml's dedicated conv_2d_dw kernel and a norm/bias that
    arrives as F16 aborts the CPU binary ops with
    "binary_op: unsupported types: dst: f32, src0: f32, src1: f16".
    """
    if name.endswith(".bias"):
        return True
    if name.endswith(".norm.weight"):
        return True
    if name.endswith(".ln.weight") or name.endswith(".ln_out.weight"):
        return True
    if name.endswith(".dw.weight"):
        return True
    return False


# ---------------------------------------------------------------------------
# ONNX graph -> stable tensor names
# ---------------------------------------------------------------------------

# Node output path (with the exporter's `model.` prefix and the trailing
# `_output_N` aliasing removed) -> GGUF tensor base name.  Everything else is
# a structural detail of the export (Shape/Gather/Slice/Range chains) that the
# C++ graph rebuilds directly, so only the parameter-bearing nodes are listed.
_TENSOR_PATTERNS: list[tuple[str, str]] = [
    (r"^/model/front/net/net\.\d+/Conv$",                       "front.conv.{i}"),
    (r"^/model/proj/MatMul$",                                   "proj"),
    (r"^/model/blocks\.(\d+)/ffn1/net/net\.1/MatMul$",          "blocks.{0}.ffn1.fc1"),
    (r"^/model/blocks\.(\d+)/ffn1/net/net\.4/MatMul$",          "blocks.{0}.ffn1.fc2"),
    (r"^/model/blocks\.(\d+)/ffn1/net/net\.0/LayerNormalization$", "blocks.{0}.ffn1.norm"),
    (r"^/model/blocks\.(\d+)/ln/LayerNormalization$",           "blocks.{0}.ln"),
    (r"^/model/blocks\.(\d+)/attn/qkv/MatMul$",                 "blocks.{0}.attn.qkv"),
    (r"^/model/blocks\.(\d+)/attn/out/MatMul$",                 "blocks.{0}.attn.out"),
    (r"^/model/blocks\.(\d+)/conv/pw1/Conv$",                   "blocks.{0}.conv.pw1"),
    (r"^/model/blocks\.(\d+)/conv/dw/Conv$",                    "blocks.{0}.conv.dw"),
    (r"^/model/blocks\.(\d+)/conv/pw2/Conv$",                   "blocks.{0}.conv.pw2"),
    (r"^/model/blocks\.(\d+)/merge/dw/Conv$",                   "blocks.{0}.merge.dw"),
    (r"^/model/blocks\.(\d+)/merge/proj/MatMul$",               "blocks.{0}.merge.proj"),
    (r"^/model/blocks\.(\d+)/ffn2/net/net\.1/MatMul$",          "blocks.{0}.ffn2.fc1"),
    (r"^/model/blocks\.(\d+)/ffn2/net/net\.4/MatMul$",          "blocks.{0}.ffn2.fc2"),
    (r"^/model/blocks\.(\d+)/ffn2/net/net\.0/LayerNormalization$", "blocks.{0}.ffn2.norm"),
    (r"^/model/blocks\.(\d+)/ln_out/LayerNormalization$",       "blocks.{0}.ln_out"),
    (r"^/model/head/head\.1/MatMul$",                           "head"),
]

# strip everything from the last '/' and any `_output_\d+` suffix
_OUTPUT_ALIAS = re.compile(r"_output_\d+$")


def node_key(output: str) -> str:
    """`/model/blocks.0/attn/qkv/MatMul_output_0` -> `/model/blocks.0/attn/qkv/MatMul`."""
    path = output.split(":", 1)[0]
    path = _OUTPUT_ALIAS.sub("", path)
    return path


def base_name(path: str, front_counter: list[int]) -> str | None:
    """Map a node path to its GGUF base name (None when it carries no weight)."""
    for pattern, fmt in _TENSOR_PATTERNS:
        m = re.match(pattern, path)
        if not m:
            continue
        if "{i}" in fmt:
            name = fmt.format(i=front_counter[0])
            front_counter[0] += 1
            return name
        return fmt.format(*m.groups())
    return None


def collect_weights(graph, inits: dict[str, Any]) -> dict[str, np.ndarray]:
    """Walk the graph and return {gguf_name: ndarray} for every parameter.

    MatMul weights are transposed on the way out: ONNX stores them as the
    right operand of `A @ W`, i.e. [in, out], while ggml/`nn.Linear` want the
    row-major [out, in] layout (ne0 = in, ne1 = out).  Conv kernels are left
    alone — ONNX [OC, IC, KH, KW] row-major is already ggml ne=[KW, KH, IC, OC].
    """
    # MatMul biases sit in an `Add` node right after the MatMul, not in the
    # MatMul node itself.
    bias_of: dict[str, str] = {}
    for node in graph.node:
        if node.op_type == "Add" and len(node.input) == 2:
            for a, b in (node.input, node.input[::-1]):
                if b in inits:
                    bias_of[a] = b
                    break

    out: dict[str, np.ndarray] = {}
    used: set[str] = set()
    front_counter = [0]

    for node in graph.node:
        if node.op_type not in ("Conv", "MatMul", "LayerNormalization"):
            continue
        path = node_key(node.output[0])
        base = base_name(path, front_counter)
        if base is None:
            continue
        # Conv / MatMul / LayerNormalization all take the weight as input[1].
        w_name = node.input[1]
        if w_name not in inits:
            raise SystemExit(f"error: node {path} has no initializer weight")
        w = numpy_helper.to_array(inits[w_name])
        if node.op_type == "MatMul":
            if w.ndim != 2:
                raise SystemExit(f"error: {base}.weight is not 2-D ({w.shape})")
            w = np.ascontiguousarray(w.T)
        out[base + ".weight"] = w
        used.add(w_name)

        b_name = node.input[2] if node.op_type in ("Conv", "LayerNormalization") else None
        if b_name is None:
            b_name = bias_of.get(node.output[0])
        if b_name is not None and b_name in inits:
            out[base + ".bias"] = numpy_helper.to_array(inits[b_name])
            used.add(b_name)
        else:
            log.warning("no bias found for %s", base)

    leftover = sorted(set(inits) - used)
    if leftover:
        # Any of these would be silently dropped from the GGUF; fail loudly.
        raise SystemExit(
            "error: %d initializers were not bound to a node: %s"
            % (len(leftover), ", ".join(leftover[:10]))
        )
    return out


# ---------------------------------------------------------------------------
# graph hyper-parameters (read off the exported graph, never guessed)
# ---------------------------------------------------------------------------

def graph_params(graph, weights: dict[str, np.ndarray], meta: dict) -> dict[str, Any]:
    f = meta["feature"]
    # shapes are ggml-style by now: a linear weight is [in, out] with `out`
    # on shape[0].
    n_blocks = 1 + max(int(re.match(r"blocks\.(\d+)\.", k).group(1))
                       for k in weights if k.startswith("blocks."))
    dim = int(weights["blocks.0.ln.weight"].shape[0])
    ffn_hidden = int(weights["blocks.0.ffn1.fc1.weight"].shape[0])
    n_heads_q, head_dim = 0, 0

    # head layout + rope base: read the qkv reshape shape [B, T, 3, H, D] and
    # the `cat(freq, freq)` rope table straight out of the Constant nodes.
    const_val: dict[str, np.ndarray] = {}
    shape_const: dict[str, list[int]] = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    shape_const[node.output[0]] = list(a.t.dims)
                    const_val[node.output[0]] = numpy_helper.to_array(a.t)
    for node in graph.node:
        if node.op_type == "Concat" and node_key(node.output[0]).endswith("/attn/Concat"):
            # [B, T, 3, H, D]: the two leading entries are dynamic (Unsqueeze
            # of Shape outputs), the trailing three are constants.
            dims = [int(const_val[i].ravel()[0]) for i in node.input
                    if i in const_val and const_val[i].size == 1]
            if len(dims) == 3:
                n_heads_q, head_dim = dims[1], dims[2]
    if head_dim == 0:
        raise SystemExit("error: qkv reshape shape not found in the graph")

    # the two Q/K scaling constants: the factors feeding `/attn/Mul_4` (scores)
    # and its transpose.  Both carry the same value c with c*c == 1/sqrt(D).
    rope_scale = None
    for node in graph.node:
        if node.op_type != "Mul":
            continue
        if not node_key(node.output[0]).endswith("/attn/Mul_4"):
            continue
        for name in node.input:
            if name in const_val and const_val[name].size == 1:
                rope_scale = float(const_val[name])
    expect = head_dim ** -0.25
    if rope_scale is None or abs(rope_scale - expect) > 1e-5:
        raise SystemExit(
            f"error: attention scale {rope_scale} does not match D^-1/4={expect}")

    # the rope table is duplicated head-wise: inv_freq[i] = theta ** (-i / D2)
    inv_freq = None
    for node in graph.node:
        if node.op_type == "Constant":
            arr = numpy_helper.to_array(node.attribute[0].t)
            if arr.ndim == 1 and arr.size == head_dim // 2:
                inv_freq = arr.astype(np.float64)
                break
    if inv_freq is None:
        raise SystemExit("error: rope frequency table not found in the graph")
    theta = float(round((1.0 / inv_freq[1]) ** (head_dim // 2)))
    if abs(theta - 10000.0) > 1.0:
        raise SystemExit(f"error: unexpected rope theta {theta}")

    # front CNN: 3x3 convs with a (2,1) max-pool after every second layer
    n_front = sum(1 for k in weights if re.match(r"front\.conv\.\d+\.weight$", k))
    front_out_ch: list[int] = []
    for i in range(n_front):
        w = weights[f"front.conv.{i}.weight"]
        front_out_ch.append(int(w.shape[0]))
        if list(w.shape[2:]) != [3, 3]:
            raise SystemExit(f"error: front.conv.{i} is not 3x3 ({w.shape})")
    pools = [k for k in range(len(front_out_ch)) if k % 2 == 1]
    # conv kernel sizes are identical across blocks
    conv_k = int(weights["blocks.0.conv.dw.weight"].shape[2])
    merge_k = int(weights["blocks.0.merge.dw.weight"].shape[2])

    if int(weights["blocks.0.attn.qkv.weight"].shape[0]) != 3 * n_heads_q * head_dim:
        raise SystemExit("error: qkv weight does not match the head layout")

    # the front flattens [C, F] into a single row per frame -> projection in dim
    proj_in = int(weights["proj.weight"].shape[1])
    freq_bins = int(f["input_rows"])
    for _ in pools:
        freq_bins //= 2
    if front_out_ch[-1] * freq_bins != proj_in:
        raise SystemExit(
            f"error: front output {front_out_ch[-1]}x{freq_bins} != proj input {proj_in}")

    return {
        "blocks": n_blocks,
        "dim": dim,
        "ffn_hidden": ffn_hidden,
        "n_heads": n_heads_q,
        "head_dim": head_dim,
        "rope_theta": theta,
        # the graph scales Q and K each by D^-1/4, i.e. the scores by 1/sqrt(D)
        "attn_scale": head_dim ** -0.5,
        "front_channels": front_out_ch,
        "front_pools": pools,
        "front_freq_bins": freq_bins,
        "conv_kernel": conv_k,
        "merge_kernel": merge_k,
        "proj_in": proj_in,
        "n_out": int(weights["head.weight"].shape[0]),   # 4 = AP/SP/start/end
    }


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def add_metadata(writer: gguf.GGUFWriter, meta: dict, gp: dict, name: str,
                 source: str) -> None:
    f = meta["feature"]
    pp = meta["postprocess"]
    ev = meta["eval"]

    writer.add_name(name)
    writer.add_string("general.version", str(BREATH_ARCH_VERSION))
    writer.add_string("breath.source", source)

    def put(key: str, value: Any) -> None:
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_int32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        else:
            writer.add_string(key, str(value))

    for key, value in gp.items():
        if isinstance(value, (list, tuple)):
            # lists have no GGUF scalar type; index them one key per entry
            for i, item in enumerate(value):
                put(f"breath.model.{key}.{i}", item)
        else:
            put(f"breath.model.{key}", value)

    put("breath.audio.sample_rate", int(meta["audio"]["sample_rate"]))
    put("breath.audio.mono", bool(meta["audio"].get("mono", True)))

    put("breath.feature.n_fft", int(f["n_fft"]))
    put("breath.feature.win_length", int(f["win_length"]))
    put("breath.feature.hop_length", int(f["hop_length"]))
    put("breath.feature.n_mels", int(f["n_mels"]))
    put("breath.feature.fmin", float(f["fmin"]))
    put("breath.feature.fmax", float(f["fmax"]))
    put("breath.feature.log_offset", float(f["log_offset"]))
    put("breath.feature.f0_channels", int(f.get("f0_channels", 0)))
    put("breath.feature.input_rows", int(f["input_rows"]))
    put("breath.feature.norm_mode", str(f.get("norm_mode", "global")))
    put("breath.feature.pitch_frame_samples", int(f.get("pitch_frame_samples", 400)))
    put("breath.feature.pitch_fmin_hz", float(f.get("pitch_fmin_hz", 70.0)))
    put("breath.feature.pitch_fmax_hz", float(f.get("pitch_fmax_hz", 1000.0)))

    put("breath.fps", int(meta["fps"]))
    put("breath.threshold", float(meta["threshold"]))

    for key in ("threshold", "median_filter_frames", "min_dur_ms", "merge_gap_ms",
                "sp_threshold", "sp_floor_percentile", "sp_floor_margin_db",
                "sp_hf_guard_db", "sp_min_dur_ms"):
        if key in pp:
            put(f"breath.postprocess.{key}", pp[key])

    put("breath.eval.infer_window_sec", float(ev.get("infer_window_sec", 12.0)))
    put("breath.eval.infer_hop_sec", float(ev.get("infer_hop_sec", 6.0)))
    put("breath.eval.event_collar_ms", float(ev.get("event_collar_ms", 200.0)))

    head = meta.get("head_index", {"ap": 0})
    for key, value in head.items():
        put(f"breath.head.{key}", int(value))

    # CMVN models ignore mean/std but the training ruler is still shipped so a
    # `norm_mode=global` model converts through the same code path.
    if "mean" in meta:
        writer.add_array("breath.mean", [float(v) for v in meta["mean"]])
    if "std" in meta:
        writer.add_array("breath.std", [float(v) for v in meta["std"]])


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def convert(model_dir: pathlib.Path, name: str, output: pathlib.Path, *,
            base_dtype: str) -> None:
    onnx_path = model_dir / f"{name}.onnx"
    meta_path = model_dir / f"{name}.meta.json"
    for p in (onnx_path, meta_path):
        if not p.exists():
            raise SystemExit(f"error: {p} not found")

    model = onnx.load(str(onnx_path))
    graph = model.graph
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    inits = {t.name: t for t in graph.initializer}

    qtype = _DTYPE_MAP[base_dtype]
    weights = collect_weights(graph, inits)
    gp = graph_params(graph, weights, meta)

    # the model consumes 64 mel rows plus the 2 pitch/periodicity rows
    rows = int(meta["feature"]["input_rows"])
    if rows != int(meta["feature"]["n_mels"]) + int(meta["feature"].get("f0_channels", 0)):
        raise SystemExit(f"error: input_rows {rows} != n_mels + f0_channels")
    if model.graph.input[0].name != "logmel" or model.graph.output[0].name != "prob":
        log.warning("unexpected onnx io names: %s -> %s",
                    model.graph.input[0].name, model.graph.output[0].name)

    writer = gguf.GGUFWriter(output, BREATH_ARCH)
    add_metadata(writer, meta, gp, name, onnx_path.name)

    total = 0
    f16 = 0
    for key in sorted(weights):
        arr = np.ascontiguousarray(weights[key], dtype=np.float32)
        eff = gguf.GGMLQuantizationType.F32
        if qtype == gguf.GGMLQuantizationType.F16 and not f32_only(key):
            arr = arr.astype(np.float16)
            eff = gguf.GGMLQuantizationType.F16
            f16 += 1
        writer.add_tensor(key, arr, raw_dtype=eff)
        total += arr.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    log.info("graph: %d blocks, dim=%d, ffn=%d, heads=%dx%d, front=%s pools=%s",
             gp["blocks"], gp["dim"], gp["ffn_hidden"], gp["n_heads"], gp["head_dim"],
             gp["front_channels"], gp["front_pools"])
    log.info("wrote %s (%.1f MB, %d tensors, %d f16)",
             output, output.stat().st_size / 1e6, len(weights), f16)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=pathlib.Path, required=True,
                        help="directory holding <name>.onnx + <name>.meta.json")
    parser.add_argument("--name", required=True,
                        help="model stem, e.g. v5_24k")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True,
                        help="output .gguf path")
    parser.add_argument("--dtype", choices=sorted(_DTYPE_MAP), default="f32",
                        help="base dtype for the weight tensors (default: f32)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    convert(args.model_dir, args.name, args.output, base_dtype=args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
