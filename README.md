# tifa.cpp — TIFA (Token-Imputing Forced Aligner) on ggml

Native C++ inference for [openvpi/TIFA](https://github.com/openvpi/TIFA), the
multilingual forced aligner used to build singing-voice datasets.  Runs on CPU,
Vulkan, Metal or CUDA with no Python at runtime, and ships with a desktop UI
for "import audio → export TextGrid annotations".

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
- **Fast** — 0.19 s for an 8.7 s clip on an RTX 2070 (Vulkan, F16), ~45×
  realtime; CPU F16 ≈ 2.9 s (≈3× realtime).
- **Multilingual G2P** — Chinese (pinyin, hanzi), Cantonese (jyutping),
  Japanese (kana), English (dictionary), with the candidate/pronunciation grid
  the model was trained with.
- **Two input modes** — text+audio (G2P) or a known phoneme sequence
  (TextGrid / DiffSinger `transcriptions.csv` / a phone list).
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

`--skip-handling discard|omit|preserve` controls what happens to zero-width
(skipped) phones; `--skip-penalty` is the raw cosine cost of a skip (0.5
matches the reference); `--backend cpu|vulkan|cuda|metal|auto` and
`--output-formats textgrid,json` are also available.

## UI

`ui/` is an Electron app (see `ui/README.md`): pick a model, import audio
(files or a folder), choose the text/phones source, run, then inspect the
waveform with the three annotation tiers and export the TextGrids.

## Layout

```
src/                 engine: backend / gguf_io / tensor_utils / mel / ops_* / model_tifa
src/g2p/             pinyin engine, converters, candidate grid, selection
src/cli/             tifa_ggml_cli
include/tifa_ggml/   public C++ API (PIMPL, ggml-free headers)
scripts/             converter, reference dumps, golden comparison, quant matrix, eval
ui/                  Electron desktop app
AGENT.md             hard-won constraints — read before changing the graph
```

## License

MPL-2.0 (this port).  TIFA's model weights and dictionaries keep their own
licenses — see the upstream release.
