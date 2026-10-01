#!/usr/bin/env python3
"""Write the synthetic smoke input used by CI (and any quick local check).

    python scripts/make_smoke_wav.py --out-dir e2e-data [--seconds 3]

Produces
    <out-dir>/wavs/smoke.wav           3 s, 44.1 kHz mono, 220 Hz sine
    <out-dir>/transcriptions.csv       one row: name,ph_seq,ph_dur

The phone sequence is ``AP SP``: both symbols exist in every TIFA-1.0
vocabulary (they are the global annotation symbols), so the aligner runs a
real forward pass without needing G2P dictionaries or an LSTM model.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import struct
import wave


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--sample-rate", type=int, default=44100)
    args = ap.parse_args()

    wav_dir = args.out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    sr = args.sample_rate
    frames = bytearray()
    for i in range(int(sr * args.seconds)):
        frames += struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * 220 * i / sr)))
    with wave.open(str(wav_dir / "smoke.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))

    (args.out_dir / "transcriptions.csv").write_text(
        "name,ph_seq,ph_dur\nsmoke,AP SP,0.1 0.1\n", encoding="utf-8")
    print(f"wrote {args.out_dir}/wavs/smoke.wav and transcriptions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
