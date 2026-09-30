#!/usr/bin/env python3
"""Score predicted TextGrids against ground truth (dataset validation).

Usage:
    python eval_textgrid.py --pred <dir-or-file> --gt <dir-or-file> \
        [--tier phones] [--tolerance-ms 20,50] [--csv report.csv]

Reports, over all matched phone intervals:
  * boundary MAE (onset / offset, ms)
  * boundary hit rate within N ms (default 20 and 50)
  * label match rate
  * counts of files / intervals, plus the worst files

Identifiers are the file stems (subdirectories are searched recursively).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys

import textgrid


def load_intervals(path: pathlib.Path, tier_name: str):
    tg = textgrid.TextGrid.fromFile(str(path))
    for tier in tg:
        if tier.name == tier_name and isinstance(tier, textgrid.IntervalTier):
            return [(float(iv.minTime), float(iv.maxTime), iv.mark.strip())
                    for iv in tier if iv.mark.strip()]
    raise KeyError(f"tier '{tier_name}' not found in {path}")


def collect(root: pathlib.Path, tier: str):
    if root.is_file():
        return {root.stem: load_intervals(root, tier)}
    out = {}
    for path in sorted(root.rglob("*.TextGrid")):
        ident = path.relative_to(root).with_suffix("").as_posix()
        try:
            out[ident] = load_intervals(path, tier)
        except Exception as exc:  # noqa: BLE001
            print(f"warn: {path}: {exc}", file=sys.stderr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", type=pathlib.Path, required=True)
    ap.add_argument("--gt", type=pathlib.Path, required=True)
    ap.add_argument("--tier", default="phones")
    ap.add_argument("--tolerance-ms", default="20,50")
    ap.add_argument("--csv", type=pathlib.Path, default=None)
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--drop-gt-marks", default="",
                    help="comma separated labels removed from the ground truth "
                         "before comparison (e.g. stop symbols the aligner drops)")
    args = ap.parse_args(argv)

    tol = [float(x) for x in str(args.tolerance_ms).split(",")]
    pred = collect(args.pred, args.tier)
    gt = collect(args.gt, args.tier)
    drop = {m.strip() for m in str(args.drop_gt_marks).split(",") if m.strip()}
    if drop:
        gt = {k: [iv for iv in v if iv[2] not in drop] for k, v in gt.items()}

    common = sorted(set(pred) & set(gt))
    if not common:
        print("no matching identifiers", file=sys.stderr)
        return 1

    onsets, offsets, hits = [], [], {t: 0 for t in tol}
    total, label_ok, per_file = 0, 0, []
    for ident in common:
        p, g = pred[ident], gt[ident]
        n = min(len(p), len(g))
        if len(p) != len(g):
            print(f"warn: {ident}: {len(p)} predicted vs {len(g)} gt intervals",
                  file=sys.stderr)
        f_on, f_off, f_hit, f_lab = [], [], {t: 0 for t in tol}, 0
        for i in range(n):
            d_on = abs(p[i][0] - g[i][0]) * 1000.0
            d_off = abs(p[i][1] - g[i][1]) * 1000.0
            onsets.append(d_on)
            offsets.append(d_off)
            f_on.append(d_on)
            f_off.append(d_off)
            for t in tol:
                if d_on <= t and d_off <= t:
                    hits[t] += 1
                    f_hit[t] += 1
            if p[i][2] == g[i][2]:
                label_ok += 1
                f_lab += 1
            total += 1
        per_file.append((statistics.mean(f_on) if f_on else 0.0, ident, n))

    def stat(name, xs):
        print(f"  {name:22s} mean {statistics.mean(xs):7.2f} ms   "
              f"median {statistics.median(xs):7.2f}   p90 {sorted(xs)[int(0.9*len(xs))]:7.2f}   "
              f"max {max(xs):8.2f}")

    print(f"files: {len(common)}   intervals: {total}")
    stat("onset error", onsets)
    stat("offset error", offsets)
    for t in tol:
        print(f"  hit@{t:.0f}ms (both edges)  {100.0 * hits[t] / max(1, total):6.2f} %")
    print(f"  label match              {100.0 * label_ok / max(1, total):6.2f} %")

    per_file.sort(reverse=True)
    if args.worst:
        print(f"worst {args.worst} files by onset MAE:")
        for mae, ident, n in per_file[:args.worst]:
            print(f"  {mae:8.2f} ms  {ident}  ({n} intervals)")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow(["identifier", "intervals", "onset_mae_ms"])
            for mae, ident, n in sorted(per_file, key=lambda x: x[1]):
                w.writerow([ident, n, f"{mae:.3f}"])
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
