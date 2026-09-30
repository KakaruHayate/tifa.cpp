#!/usr/bin/env python3
"""Compare C++ (tifa_ggml) raw dumps against the PyTorch reference dump.

Usage:
    python compare_golden.py --ref golden/sample_0_0 --cpp golden/cpp_sample_0_0
                             [--tolerance 1e-4]

`--ref` holds the .npy files written by dump_tifa_reference.py; `--cpp` holds
the raw .bin files written by `tifa_ggml_cli align --dump-dir`.  Exits non-zero
when any stage exceeds the tolerance.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def load_stage(ref_dir: pathlib.Path, cpp_dir: pathlib.Path, name: str):
    ref = np.load(ref_dir / f"{name}.npy")
    meta = json.loads((cpp_dir / "meta.json").read_text(encoding="utf8"))
    if name in ("mel", "frame_features", "frame_logits", "similarity"):
        cols = {"mel": meta["in_dim"], "frame_features": meta["out_dim"],
                "frame_logits": meta["vocab_size"], "similarity": meta["num_tokens"]}[name]
        cpp = np.fromfile(cpp_dir / f"{name}.bin", dtype=np.float32)
        if name == "similarity":
            cpp = cpp.reshape(meta["num_frames"], cols)
        else:
            cpp = cpp.reshape(meta["num_frames"], cols)
    else:
        cols = {"token_features": meta["out_dim"], "token_logits": meta["vocab_size"]}[name]
        cpp = np.fromfile(cpp_dir / f"{name}.bin", dtype=np.float32).reshape(meta["num_tokens"], cols)
    return ref, cpp


def report(name: str, ref: np.ndarray, cpp: np.ndarray, tol: float) -> bool:
    if ref.shape != cpp.shape:
        print(f"  {name:16s} SHAPE MISMATCH ref{ref.shape} cpp{cpp.shape}")
        return False
    diff = np.abs(ref.astype(np.float64) - cpp.astype(np.float64))
    scale = max(1e-12, float(np.abs(ref).max()))
    ok = diff.max() <= tol * max(1.0, scale)
    print(f"  {name:16s} max_abs {diff.max():.3e}  mean_abs {diff.mean():.3e}  "
          f"ref_max {scale:.3f}  {'OK' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", type=pathlib.Path, required=True)
    ap.add_argument("--cpp", type=pathlib.Path, required=True)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args(argv)

    print(f"comparing {args.ref} vs {args.cpp} (tolerance {args.tolerance})")
    ok = True
    for name in ("mel", "frame_features", "token_features",
                 "frame_logits", "token_logits", "similarity"):
        ref, cpp = load_stage(args.ref, args.cpp, name)
        ok &= report(name, ref, cpp, args.tolerance)

    spans_ref = np.load(args.ref / "spans.npy")
    spans_cpp = np.fromfile(args.cpp / "spans.bin", dtype=np.int32).reshape(-1, 2).astype(np.int64)
    if spans_ref.shape == spans_cpp.shape:
        n_diff = int((spans_ref != spans_cpp).any(axis=1).sum())
        print(f"  {'spans':16s} {n_diff} differing rows of {len(spans_ref)}  "
              f"{'OK' if n_diff == 0 else 'FAIL'}")
        ok &= n_diff == 0
    else:
        print(f"  {'spans':16s} SHAPE MISMATCH ref{spans_ref.shape} cpp{spans_cpp.shape}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
