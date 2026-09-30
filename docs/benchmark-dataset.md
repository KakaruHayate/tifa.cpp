# Accuracy & speed on the ZH-sample dataset

Dataset: `path\to\dataset` — 1063 singing clips (44.1 kHz mono,
≈7 s each, 31 864 phone intervals) with human-verified phone boundaries; the
same data was part of TIFA-1.0-ST's training mix.

Procedure: feed each clip together with its known phone sequence
(`transcriptions.csv` → `ph_seq`, stop symbols dropped) and compare the
predicted `phones` tier against the ground-truth TextGrid.

## Accuracy (all 1063 files)

| metric | value |
|---|---|
| onset error, mean / median / p90 / max | **5.37 / 3.32 / 10.33 / 287.01 ms** |
| offset error, mean / median / p90 / max | **5.28 / 3.33 / 10.05 / 270.00 ms** |
| boundary hit @20 ms (both edges) | **94.46 %** |
| boundary hit @50 ms (both edges) | **98.82 %** |
| phone label agreement | **100 %** |

One frame is 10 ms (hop 480 @ 48 kHz), so the median error is a third of a
frame.  The outliers are short clips with long silences (`one long-silence clip`, 12
intervals: 30.9 ms) where a pause boundary is genuinely ambiguous.

## Parity with the PyTorch reference

`scripts/compare_golden.py` against the official implementation (48 kHz input,
batch size 1, same phone sequence):

| stage | max abs. difference |
|---|---|
| log-mel | 3.2e-04 |
| frame / token features | 1.7e-03 / 1.1e-03 |
| frame / token logits | 3.9e-03 / 1.5e-03 |
| cosine similarity | 1.6e-04 |
| **decoded spans** | **0 differing rows (exact)** |

Diagnosis metrics match to ~3e-6 (agreement 0.999755, confidence 0.847871,
determinacy 0.905443, monotonicity 0.921436 vs the reference's
0.847874 / 0.905445 / 0.921439).

## Speed (RTX 2070 + Ryzen, Release build, 8.7 s clip)

| configuration | model size | latency | realtime factor |
|---|---|---|---|
| CPU, F32 | 158 MB | 5.7 s | 0.7× |
| CPU, F16 | 80 MB | 2.9 s | 3.0× |
| Vulkan, F32 | 158 MB | 0.19 s | 45× |
| Vulkan, F16 | 80 MB | 0.19 s | 45× |

All four produce byte-identical TextGrids on the checked clips.

## Reproducing

```bash
# align the whole dataset (Vulkan, F16)
TIFA_GGML_BACKEND=vulkan build/bin/tifa_ggml_cli.exe align \
    ../../dataset/wavs -m ../models/tifa-1.0-st-f16.gguf \
    --transcriptions-csv ../../dataset/transcriptions.csv \
    -l zh -o ../eval/pred --output-formats textgrid,json -q

# score it
python scripts/eval_textgrid.py --pred ../eval/pred \
    --gt ../../dataset/wavs --tier phones \
    --drop-gt-marks SP,sil,pau --csv ../eval/eval.csv
```
