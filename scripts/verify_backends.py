#!/usr/bin/env python3
"""Cross-backend / cross-dtype verification: align the same clips with several
(model, backend) combinations and report how many phone boundaries differ.

Usage:
    python verify_backends.py --cli build/bin/tifa_ggml_cli.exe \
        --models "f32=../models/tifa-1.0-st-f32.gguf,f16=../models/tifa-1.0-st-f16.gguf" \
        --dataset ../dataset --limit 20 \
        --backends "cpu,vulkan" --out-dir ../eval/backend

The first combination is the reference; every other one reports
  * files with identical TextGrid text,
  * max/mean absolute onset & offset difference (ms) over matched intervals,
  * label mismatches.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def parse_pairs(spec: str):
    out = []
    for item in spec.split(","):
        if not item.strip():
            continue
        name, _, value = item.partition("=")
        out.append((name.strip(), value.strip()))
    return out


def run(cli: pathlib.Path, model: pathlib.Path, backend: str, wav: pathlib.Path,
        csv: pathlib.Path, out_dir: pathlib.Path) -> None:
    import os
    env = dict(os.environ)
    env["TIFA_GGML_BACKEND"] = backend
    subprocess.run([str(cli), "align", str(wav), "-m", str(model),
                    "--transcriptions-csv", str(csv), "-l", "zh",
                    "-o", str(out_dir), "-q"],
                   env=env, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def intervals(path: pathlib.Path):
    import textgrid
    tg = textgrid.TextGrid.fromFile(str(path))
    for tier in tg:
        if tier.name == "phones" and isinstance(tier, textgrid.IntervalTier):
            return [(float(iv.minTime), float(iv.maxTime), iv.mark.strip())
                    for iv in tier if iv.mark.strip()]
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cli", type=pathlib.Path, required=True)
    ap.add_argument("--models", required=True, help="name=path[,name=path...]")
    ap.add_argument("--backends", default="cpu,vulkan")
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)

    models = parse_pairs(args.models)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    csv = args.dataset / "transcriptions.csv"
    wavs = sorted((args.dataset / "wavs").glob("*.wav"))[:args.limit]

    combos = list(itertools.product(models, backends))
    results = {}
    for (mname, mpath), backend in combos:
        tag = f"{mname}-{backend}"
        out_dir = args.out_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        for wav in wavs:
            run(args.cli, pathlib.Path(mpath), backend, wav, csv, out_dir)
        results[tag] = {w.stem: intervals(out_dir / f"{w.stem}.TextGrid") for w in wavs}
        print(f"ran {tag}: {len(wavs)} files")

    ref_tag = f"{models[0][0]}-{backends[0]}"
    ref = results[ref_tag]
    print(f"\nreference: {ref_tag}")
    print(f"{'combo':28s} {'identical':>10s} {'onset_max':>10s} {'onset_mean':>11s} "
          f"{'offset_max':>11s} {'label_mismatch':>15s}")
    for tag, res in results.items():
        if tag == ref_tag:
            continue
        same = 0
        on_err, off_err, label_bad, count = [], [], 0, 0
        for ident, r in ref.items():
            p = res.get(ident, [])
            if len(p) == len(r) and all(
                    abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 and a[2] == b[2]
                    for a, b in zip(p, r)):
                same += 1
            for a, b in zip(p, r):
                on_err.append(abs(a[0] - b[0]) * 1000.0)
                off_err.append(abs(a[1] - b[1]) * 1000.0)
                if a[2] != b[2]:
                    label_bad += 1
                count += 1
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f"{tag:28s} {same:4d}/{len(ref):<5d} {max(on_err or [0]):10.3f} "
              f"{mean(on_err):11.3f} {max(off_err or [0]):11.3f} {label_bad:15d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
