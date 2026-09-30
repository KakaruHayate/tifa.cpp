#!/usr/bin/env python3
"""Compute-efficiency benchmark: tifa_ggml (CPU / Vulkan, F16 / Q4_0) against
the PyTorch reference (CPU / CUDA).

Both sides run the *same* pipeline — wav (44.1 kHz dataset files) + the known
phone sequence from `transcriptions.csv` -> TextGrid — so the numbers include
the audio decode + resample + mel front end, exactly like a real dataset run.

Usage:
    python benchmark_efficiency.py --cli ../tifa.cpp/build/bin/tifa_ggml_cli.exe \
        --dataset ../../dataset --limit 20 \
        --model ../models/tifa-1.0-st-f16.gguf --model-cpu ../models/tifa-1.0-st-q4_0.gguf \
        --report ../docs/benchmark-efficiency.md

Notes
- The ggml side is timed with one CLI process per configuration: a 1-file run
  gives (model load + 1 file), an N-file run gives (model load + N files), so
  per-file latency is their difference divided by N-1 — that removes the load
  time instead of hiding it inside the average.
- The PyTorch side loads once, warms up, then times the loop.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys
import time


def stage_wavs(dataset: pathlib.Path, limit: int, dest: pathlib.Path) -> list:
    """Copy the first `limit` clips into a scratch dir (the CLI scans dirs)."""
    import shutil
    dest.mkdir(parents=True, exist_ok=True)
    wavs = sorted((dataset / "wavs").glob("*.wav"))[:limit]
    out = []
    for w in wavs:
        target = dest / w.name
        if not target.exists():
            shutil.copy2(w, target)
        out.append(w)
    return out


def ggml_per_file(cli: pathlib.Path, model: pathlib.Path, backend: str,
                  dataset: pathlib.Path, limit: int,
                  scratch: pathlib.Path) -> tuple[float, float]:
    """Return (per_file_seconds, model_load_seconds)."""
    import os
    env = dict(os.environ)
    env["TIFA_GGML_BACKEND"] = backend
    wavs = stage_wavs(dataset, limit, scratch)
    csv = dataset / "transcriptions.csv"
    one_dir = scratch / "one"
    one_dir.mkdir(parents=True, exist_ok=True)
    import shutil as _sh
    _sh.copy2(wavs[0], one_dir / wavs[0].name)

    def run(folder):
        t0 = time.perf_counter()
        subprocess.run([str(cli), "align", str(folder),
                        "-m", str(model), "--transcriptions-csv", str(csv),
                        "-l", "zh", "-o", str(scratch / "out"), "-q"],
                       env=env, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return time.perf_counter() - t0

    t_one = run(one_dir)
    t_all = run(scratch)
    per_file = (t_all - t_one) / max(1, len(wavs) - 1)
    return per_file, t_one - per_file


def torch_per_file(model_dir: pathlib.Path, tifa_root: pathlib.Path,
                   dataset: pathlib.Path, limit: int,
                   device: str) -> tuple[float, float]:
    """Return (per_file_seconds, load_seconds) for the PyTorch reference."""
    import csv as _csv
    import numpy as np
    import torch
    import librosa
    sys.path.insert(0, str(tifa_root.resolve()))
    from inference.api import load_inference_model
    from lib.config.schema import ConfigurationScope

    t0 = time.perf_counter()
    backend, vocabulary, _ = load_inference_model(model_dir / "model.pt",
                                                  scope=ConfigurationScope.FA)
    backend = backend.to(device).eval()
    load = time.perf_counter() - t0

    rows = {r["name"]: r for r in _csv.DictReader(
        (dataset / "transcriptions.csv").open(encoding="utf8"))}
    wavs = sorted((dataset / "wavs").glob("*.wav"))[:limit]

    def one(path):
        wav, sr = librosa.load(str(path), sr=None, mono=True)
        target = backend.sample_rate
        if sr != target:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target)
        w = torch.from_numpy(wav).float()[None].to(device)
        dur = torch.tensor([len(wav) / target], dtype=torch.float32, device=device)
        phones = [p for p in rows[path.stem]["ph_seq"].split()
                  if p not in ("SP", "sil", "pau")]
        langs = ["zh"]
        toks = []
        for ph in phones:
            r = vocabulary.resolve(ph, langs) or vocabulary.resolve(ph, [])
            toks.append(int(r[0]))
        tokens = torch.tensor([toks], dtype=torch.int64, device=device)
        with torch.no_grad():
            spec = backend.spectrogram(w, dur)
            backend.align(spec, tokens=tokens, unit="frame", skip_penalty=0.5)

    if device == "cuda":
        torch.cuda.synchronize()
    for p in wavs[:2]:      # warmup (shader/allocator/kernel autotune)
        one(p)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for p in wavs:
        one(p)
    if device == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return total / len(wavs), load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cli", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--model", type=pathlib.Path, required=True, help="F16 GGUF")
    ap.add_argument("--model-q4", type=pathlib.Path, default=None)
    ap.add_argument("--model-dir", type=pathlib.Path, default=None,
                    help="TIFA-1.0-ST checkpoint dir for the PyTorch reference")
    ap.add_argument("--tifa-root", type=pathlib.Path, default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--report", type=pathlib.Path, default=None)
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args(argv)

    scratch = pathlib.Path("tifa_bench_scratch").resolve()
    wavs = sorted((args.dataset / "wavs").glob("*.wav"))[:args.limit]
    secs = []
    for w in wavs:
        import wave
        with wave.open(str(w)) as fh:
            secs.append(fh.getnframes() / fh.getframerate())
    audio_seconds = sum(secs)

    rows = []
    for backend in ("cpu", "vulkan"):
        for name, model in (("F16", args.model), ("Q4_0", args.model_q4)):
            if model is None:
                continue
            per_file, load = ggml_per_file(args.cli, model, backend, args.dataset,
                                           args.limit, scratch)
            rows.append((f"tifa_ggml · {backend} · {name}", per_file, load))
            print(f"ggml {backend:7s} {name:5s} {per_file * 1000:8.1f} ms/file "
                  f"(load {load:.2f} s)")

    if not args.skip_torch and args.model_dir and args.tifa_root:
        for device in ("cpu", "cuda"):
            try:
                per_file, load = torch_per_file(args.model_dir, args.tifa_root,
                                                args.dataset, args.limit, device)
                rows.append((f"PyTorch · {device} · F32", per_file, load))
                print(f"torch  {device:7s} F32   {per_file * 1000:8.1f} ms/file "
                      f"(load {load:.2f} s)")
            except Exception as exc:  # noqa: BLE001
                print(f"torch {device}: failed: {exc}", file=sys.stderr)

    print(f"\naudio: {len(wavs)} files, {audio_seconds:.1f} s total")
    if args.report:
        lines = ["# Compute efficiency: tifa_ggml vs PyTorch\n",
                 f"{len(wavs)} clips ({audio_seconds:.1f} s of audio), warm start, "
                 "the full pipeline per file (decode + resample + mel + network + Viterbi).\n",
                 "| engine | backend | dtype | ms / file | × realtime | model load |",
                 "|---|---|---|---|---|---|"]
        for name, per_file, load in rows:
            rtf = (audio_seconds / len(wavs)) / per_file
            engine, backend, dtype = name.split(" · ")
            lines.append(f"| {engine} | {backend} | {dtype} | {per_file * 1000:.0f} | "
                         f"{rtf:.1f}× | {load:.2f} s |")
        lines.append("\n× realtime = audio seconds per wall-clock second "
                     "(higher is better).\n")
        args.report.write_text("\n".join(lines), encoding="utf8")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
