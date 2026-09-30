#!/usr/bin/env python3
"""Per-role quantization study for the TIFA GGUF (mirrors game.cpp's work).

For every recipe it
  1. converts the checkpoint to a GGUF with that per-tensor dtype policy,
  2. aligns a subset of the sample dataset with the produced model,
  3. scores the TextGrids against the ground truth (boundary MAE / hit rates),
  4. records model size and per-file latency,
and finally writes a markdown matrix so the size/accuracy trade-off is explicit.

Usage:
    python quant_matrix.py --model-dir ../models/TIFA-1.0-ST \
        --cli ../tifa.cpp/build/bin/tifa_ggml_cli.exe \
        --dataset ../../dataset --limit 40 \
        --backend cpu --out-dir ../eval/quant

Recipes are (name, base dtype, extra fnmatch rules) — rules override the base
for matching tensors; norms, LayerScale and depthwise-conv kernels always stay
F32 (see convert_tifa_to_gguf.f32_only).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

# Tensor-name patterns; names are the stripped (no `model.`) module paths.
ATTN = ["backbone.layers.*.attn.jattn.*.weight",
        "backbone.layers.*.attn.merge_linear_*.weight"]
FFN = ["backbone.layers.*.ffn1_*.ln*.weight", "backbone.layers.*.ffn2_*.ln*.weight"]
CGMLP = ["backbone.layers.*.attn.c_*.pw*.weight"]
EMB = ["token_embedding.weight"]
PROJ = ["backbone.audio_input_proj.weight", "backbone.text_input_proj.weight",
        "backbone.output_proj_*.weight"]
LAYERS = ["backbone.layers.*.ffn1_*.ln*.weight", "backbone.layers.*.ffn2_*.ln*.weight",
          "backbone.layers.*.attn.jattn.*.weight",
          "backbone.layers.*.attn.merge_linear_*.weight",
          "backbone.layers.*.attn.c_*.pw*.weight"]


def recipe(name: str, base: str, rules):
    return {"name": name, "base": base, "rules": rules}


RECIPES = [
    recipe("f32", "f32", []),
    recipe("f16", "f16", []),
    recipe("q8_0-all", "f16", [(p, "q8_0") for p in LAYERS + PROJ]),
    recipe("q8_0-attn", "f16", [(p, "q8_0") for p in ATTN]),
    recipe("q8_0-ffn", "f16", [(p, "q8_0") for p in FFN]),
    recipe("q8_0-cgmlp", "f16", [(p, "q8_0") for p in CGMLP]),
    recipe("q8_0-proj", "f16", [(p, "q8_0") for p in PROJ] + [(p, "q8_0") for p in EMB]),
    recipe("q8_0-layers", "f16", [(p, "q8_0") for p in LAYERS]),
    recipe("q4_0-all", "f16", [(p, "q4_0") for p in LAYERS + PROJ]),
    recipe("q5_k-all", "f16", [(p, "q5_k") for p in LAYERS + PROJ]),
    recipe("q6_k-all", "f16", [(p, "q6_k") for p in LAYERS + PROJ]),
    # Mixed: attention in Q8_0, FFN in Q4_0 (the FFN dominates the parameter count)
    recipe("mix-q8attn-q4ffn", "f16",
           [(p, "q8_0") for p in ATTN] + [(p, "q4_0") for p in FFN]),
]


def convert(model_dir: pathlib.Path, out: pathlib.Path, base: str, rules, python: str) -> None:
    cfg = {"rules": [{"pattern": p, "dtype": d} for p, d in rules]}
    cfg_path = out.with_suffix(".rules.json")
    cfg_path.write_text(json.dumps(cfg), encoding="utf8")
    cmd = [python, str(HERE / "convert_tifa_to_gguf.py"),
           "--model-dir", str(model_dir), "-o", str(out),
           "--dtype", base, "--quant-config", str(cfg_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def align_and_score(cli: pathlib.Path, model: pathlib.Path, dataset: pathlib.Path,
                    out_dir: pathlib.Path, backend: str, limit: int,
                    env_extra=None) -> tuple[float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    import os
    env = dict(os.environ)
    env["TIFA_GGML_BACKEND"] = backend
    if env_extra:
        env.update(env_extra)

    wavs = sorted((dataset / "wavs").glob("*.wav"))[:limit]
    t0 = time.time()
    cmd = [str(cli), "align", str(dataset / "wavs"),
           "-m", str(model),
           "--transcriptions-csv", str(dataset / "transcriptions.csv"),
           "-l", "zh", "-o", str(out_dir), "-q"]
    # Align only the subset: run per file so the limit applies.
    for wav in wavs:
        subprocess.run([str(cli), "align", str(wav), "-m", str(model),
                        "--transcriptions-csv", str(dataset / "transcriptions.csv"),
                        "-l", "zh", "-o", str(out_dir), "-q"],
                       env=env, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    elapsed = time.time() - t0

    eval_cmd = [sys.executable, str(HERE / "eval_textgrid.py"),
                "--pred", str(out_dir), "--gt", str(dataset / "wavs"),
                "--tier", "phones", "--drop-gt-marks", "SP,sil,pau", "--worst", "0"]
    res = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
    mae = None
    hit = None
    import re
    for line in res.stdout.splitlines():
        if "onset error" in line:
            mae = float(line.split("mean")[1].split("ms")[0])
        if "hit@20ms" in line:
            m = re.search(r"([0-9.]+)\s*%", line)
            if m:
                hit = float(m.group(1))
    return (mae if mae is not None else float("nan"),
            hit if hit is not None else float("nan")), elapsed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=pathlib.Path, required=True)
    ap.add_argument("--cli", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--backend", default="vulkan")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only", default=None, help="comma separated recipe names")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else None

    rows = []
    for rec in RECIPES:
        if wanted and rec["name"] not in wanted:
            continue
        gguf = args.out_dir / f"tifa-{rec['name']}.gguf"
        print(f"== {rec['name']} ==", flush=True)
        convert(args.model_dir, gguf, rec["base"], rec["rules"], args.python)
        size_mb = gguf.stat().st_size / 1e6
        pred_dir = args.out_dir / f"pred-{rec['name']}"
        (mae, hit), elapsed = align_and_score(
            args.cli, gguf, args.dataset, pred_dir, args.backend, args.limit)
        per_file = elapsed / max(1, args.limit)
        rows.append((rec["name"], size_mb, mae, hit, per_file))
        print(f"   size {size_mb:6.1f} MB   onset MAE {mae:6.2f} ms   "
              f"hit@20 {hit:5.1f}%   {per_file * 1000:6.0f} ms/file", flush=True)

    md = args.out_dir / "quant-matrix.md"
    with md.open("w", encoding="utf8") as fh:
        fh.write("# TIFA-1.0-ST quantization matrix\n\n")
        fh.write(f"dataset: {args.dataset} (first {args.limit} files), "
                 f"backend: {args.backend}\n\n")
        fh.write("| recipe | size (MB) | onset MAE (ms) | hit@20ms (%) | ms/file |\n")
        fh.write("|---|---|---|---|---|\n")
        for name, size, mae, hit, per_file in rows:
            fh.write(f"| {name} | {size:.1f} | {mae:.2f} | {hit:.1f} | "
                     f"{per_file * 1000:.0f} |\n")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
