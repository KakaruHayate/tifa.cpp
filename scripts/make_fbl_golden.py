#!/usr/bin/env python3
"""Golden for the FoxBreatheLabeler ggml port (tests/test_fbl_net.cpp).

Runs the reference ONNX on a deterministic frame matrix and records its AP
probabilities.  The matrix is generated from a formula both sides share, so the
golden only has to carry the output:

    frames[k][t] = sin(0.017*k + 0.11*t) * cos(0.003*(k+t))

Usage:
    python make_fbl_golden.py --onnx fbl02_1cls.onnx -o tests/golden/fbl_ap.json
"""
import argparse
import json
import math
import pathlib

import numpy as np
import onnxruntime as ort

SPEC_WIN = 1024
T = 64


def waveform(n):
    """Deterministic signal; the graph frames it internally (pad 71, hop 882)."""
    t = np.arange(n, dtype=np.float32)
    return (np.sin(0.017 * t) + 0.3 * np.cos(0.003 * t)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    # T frames come out of N samples: (N + 2*71 - 1024)//882 + 1 == T
    n = T * 882
    x = waveform(n)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x[None, :]})[0]              # [1, 1, T]
    logits = np.asarray(out).reshape(-1)[:T]
    # the exported graph stops at the raw logits; the net applies sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))

    doc = {
        "spec_win": SPEC_WIN,
        "frames": T,
        "samples": T * 882,
        "formula": "sin(0.017*t) + 0.3*cos(0.003*t)",
        "source": pathlib.Path(args.onnx).name,
        "ap_prob": [float(v) for v in probs],
        "min": float(probs.min()),
        "max": float(probs.max()),
        "mean": float(probs.mean()),
    }
    pathlib.Path(args.output).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {args.output}: min={doc['min']:.4f} max={doc['max']:.4f} "
          f"mean={doc['mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
