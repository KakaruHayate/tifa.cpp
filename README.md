# tifa.cpp — TIFA (Token-Imputing Forced Aligner) on ggml

Native C++ inference for [openvpi/TIFA](https://github.com/openvpi/TIFA), the
multilingual forced aligner used to build singing-voice datasets.  Runs on CPU,
Vulkan, Metal or CUDA with no Python at runtime, and ships with **TIFA Label**,
a desktop tool for the full dataset pipeline: TIFA align → FBL breath AP/SP →
2PASS re-align.

> [中文说明](README_CN.md) · dataset workflow: [docs/dataset-workflow.md](docs/dataset-workflow.md)

## What it does

Given a recording and its text (or an already known phoneme sequence), TIFA
predicts phone-level time boundaries and writes a three-tier Praat TextGrid
(`texts` / `words` / `phones`), plus optional self-check metrics that flag
alignments worth a human look.

```
audio ──► log-mel (48 kHz, hop 480, 80 bins)
      ──► JEBF backbone (8 layers, dim 256, joint attention, RoPE, qk-norm)
      ──► frame/token features + token logits
text  ──► G2P ──► candidate grid ──► (optional pronunciation scoring)
      ──► cosine similarity [T, N] ──► Viterbi decode ──► spans ──► TextGrid
```

## Highlights

- **Parity-tested** — layer-by-layer identical to the PyTorch reference
  (spans match exactly; float differences ≤ 1e-3).
- **Fast** — 0.15 s per file on an RTX 2070 (Vulkan, F16) over the 20-clip
  benchmark, 42× realtime; CPU F16 11.5× realtime.  The CgMLP depthwise convs
  run through the dedicated `GGML_OP_CONV_2D_DW` kernel (no im2col) — the same
  direct path game.cpp uses.
- **Multilingual G2P** — Chinese (pinyin, hanzi), Cantonese (jyutping),
  Japanese (kana), English (dictionary + LSTM OOV inference), with the
  candidate/pronunciation grid the model was trained with.
- **PFML input** — the upstream Pronunciation Flow Markup Language: final
  phonemes (`<word phonemes="zh ong">重</word>`) and language scopes
  (`<scope language="ja">東京</scope>`) embed directly in the transcript.
- **Full dataset workflow** — align → `breathe --merge` (FBL AP/SP folded into
  the phones tier) → 2PASS re-align, in the CLI and in the GUI.
- **Quantization matrix** — F32/F16/Q8_0/Q4_0 per-tensor recipes with measured
  boundary-error impact (`scripts/quant_matrix.py`).

## Build

```bash
# Windows (MSVC): edit the VS path in the .bat if needed
scripts\configure.bat
scripts\build.bat
# Linux/macOS
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIFA_GGML_VULKAN=ON
cmake --build build -j
```

Options: `TIFA_GGML_VULKAN`, `TIFA_GGML_CUDA`, `TIFA_GGML_METAL`,
`TIFA_GGML_BUILD_CLI`, `TIFA_GGML_BUILD_TESTS`.

## Model

Download `TIFA-1.0-ST.zip` from the [TIFA releases](https://github.com/openvpi/TIFA/releases)
and convert the checkpoint:

```bash
python scripts/convert_tifa_to_gguf.py \
    --model-dir models/TIFA-1.0-ST -o models/tifa-1.0-st-f16.gguf --dtype f16
```

The GGUF embeds the vocabulary and the G2P pipeline configuration; the
pronunciation dictionaries stay as files next to the model
(`ds-zh-pinyin-lite.txt`, `jyutping_dict.txt`, `japanese_dict_full.txt`,
`ds_cmudict-07b.txt`).

## CLI

```bash
# known phoneme sequence (DiffSinger dataset)
tifa_ggml_cli align wavs/ -m models/tifa-1.0-st-f16.gguf \
    --transcriptions-csv transcriptions.csv -l zh -o out --output-formats textgrid,json

# phones from a TextGrid / inline
tifa_ggml_cli align song.wav -m model.gguf --textgrid song.TextGrid -l zh
tifa_ggml_cli align song.wav -m model.gguf --phones "AP zh e n a"

# text + audio (G2P) — sidecar .txt/.lab beside the audio
tifa_ggml_cli align song.wav -m model.gguf -l zh

tifa_ggml_cli inspect model.gguf
```

```bash
# breath/AP-SP detection; --merge folds the result into a first-pass
# alignment's phones tier (step 2 of the dataset 2PASS workflow)
tifa_ggml_cli breathe song.wav -m models/breath-v5-24k.gguf --merge out -o out
# step 3: re-align with the breath annotation in place
tifa_ggml_cli align song.wav -m models/tifa.gguf --textgrid out -o out2
```

`--skip-handling discard|omit|preserve` controls what happens to zero-width
(skipped) phones; `--skip-penalty` is the raw cosine cost of a skip (0.5
matches the reference); `--oov-handling raise|discard|force` selects the G2P
out-of-vocabulary policy; `--backend cpu|vulkan|cuda|metal|auto` and
`--output-formats textgrid,json` are also available.

## TIFA Label (desktop tool)

`ui/` is an Electron app (see `ui/README.md`) that drives the dataset
pipeline in one batch run per file:

1. **Import audio** — files or folders (scanned recursively).
2. **Annotation source** — text/G2P (sidecar `.txt`/`.lab` or a uniform
   transcript; PFML accepted) or existing phones (same-name TextGrid, a
   DiffSinger `transcriptions.csv`, a TextGrid folder, an inline list).  The
   old `.lab + wav` workflow never touches G2P.
3. **Pipeline** — align → (optional) FBL breath detection merged into the
   phones tier → (optional) 2PASS re-align.

Per-file status/agreement, progress, streaming log, cancel.  The UI is in
Chinese.

## Layout

```
src/                 engine: backend / gguf_io / tensor_utils / mel / ops_* / model_tifa
src/g2p/             pinyin engine, converters (English LSTM, PFML), candidate grid
src/breath/          FBL breath AP/SP detection on ggml
src/cli/             tifa_ggml_cli (align / breathe / inspect)
include/tifa_ggml/   public C++ API (PIMPL, ggml-free headers)
scripts/             converter, reference dumps, golden comparison, quant matrix, eval
ui/                  Electron desktop app
AGENT.md             hard-won constraints — read before changing the graph
```

## License

MPL-2.0 (this port).  TIFA's model weights and dictionaries keep their own
licenses — see the upstream release.
