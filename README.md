# tifa.cpp — TIFA (Token-Imputing Forced Aligner) inference engine on ggml

Native C++ inference for [openvpi/TIFA](https://github.com/openvpi/TIFA), the
multilingual forced aligner used to build singing-voice datasets.  Runs on CPU,
Vulkan, Metal or CUDA with no Python at runtime.

**This is an engine, not a pipeline.**  It takes audio plus an already-known
phone sequence and predicts phone-level boundaries.  Text-to-phoneme (G2P),
PFML, breath detection, pronunciation dictionaries and the Electron annotator
were removed in 0.2.0 — see [docs/scope-change.md](docs/scope-change.md) for
why, and for the migration guide.

> [中文说明](README_CN.md) · scope change: [docs/scope-change.md](docs/scope-change.md) · bundled usage guide (bilingual): [docs/usage.md](docs/usage.md)

## Download

Prebuilt packages live in [Releases](https://github.com/KakaruHayate/tifa.cpp/releases):

| package | contents |
|---|---|
| `tifa-cli-<platform>-full.tar.gz` | `tifa` engine + F16 full-precision model |
| `tifa-cli-<platform>-q4.tar.gz` | `tifa` engine + Q4_0 smallest model |

`<platform>` is `windows-x64`, `linux-x64` or `macos-arm64`.  Each archive
already contains the model and the runtime libraries, so one download is all it
takes.  See `USAGE.md` inside it (Chinese + English).  To build from source see
*Build* below.

> The macOS build is unsigned: right-click > Open the first time.

## What it does

Given a recording and its phone sequence, TIFA predicts phone-level time
boundaries and writes a three-tier Praat TextGrid (`texts` / `words` /
`phones`), plus optional self-check metrics that flag alignments worth a human
look.

Each tier is written continuously: a stretch that no phone or word claims — the
decoder's gap states — becomes an interval with an empty label.  This mirrors
the Python `textgrid` package's `_fillInTheGaps`, which upstream TIFA serialises
through, and it is what strict consumers such as vLabeler's `continuous`
TextGrid labeler require.

```
audio ──► log-mel (48 kHz, hop 480, 80 bins)
      ──► JEBF backbone (8 layers, dim 256, joint attention, RoPE, qk-norm)
      ──► frame/token features + token logits
phones ──► cosine similarity [T, N] ──► Viterbi decode ──► spans ──► TextGrid
```

## Highlights

- **Parity-tested** — layer-by-layer identical to the PyTorch reference
  (spans match exactly; float differences ≤ 1e-3).
- **Fast** — 0.15 s per file on an RTX 2070 (Vulkan, F16) over the 20-clip
  benchmark, 42× realtime; CPU F16 11.5× realtime.  The CgMLP depthwise convs
  run through the dedicated `GGML_OP_CONV_2D_DW` kernel (no im2col) — the same
  direct path game.cpp uses.
- **Stable contract** — audio + phones + language in, spans/TextGrid/JSON out.
  No dictionaries, no tokenizer, no model-side text preprocessing.
- **Phone inputs for existing tools** — inline list, `.txt`, Praat TextGrid,
  or a DiffSinger `transcriptions.csv`; `--textgrid` also re-attaches the
  first pass's `texts`/`words` tiers for a 2PASS re-align.
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
`TIFA_GGML_BUILD_CLI`, `TIFA_GGML_BUILD_TESTS`.  The CLI binary is
`build/bin/tifa` (`tifa.exe` on Windows).

## Model

Download `TIFA-1.0-ST.zip` from the [TIFA releases](https://github.com/openvpi/TIFA/releases)
and convert the checkpoint:

```bash
python scripts/convert_tifa_to_gguf.py \
    --model-dir models/TIFA-1.0-ST -o models/tifa-1.0-st-f16.gguf --dtype f16
```

The GGUF embeds the vocabulary and the global/stop symbol sets.  It no longer
embeds a G2P pipeline configuration; the engine has no use for one.

## CLI

`tifa` needs a phone source — it will not invent phones from text.

```bash
# known phoneme sequence (DiffSinger dataset)
tifa align wavs/ -m models/tifa-1.0-st-f16.gguf \
    --transcriptions-csv transcriptions.csv -l zh -o out --output-formats textgrid,json

# phones from a TextGrid / inline / a .txt file
tifa align song.wav -m model.gguf --textgrid song.TextGrid -l zh
tifa align song.wav -m model.gguf --phones "AP zh e n a"
tifa align song.wav -m model.gguf --phones-file phones.txt -l zh

tifa inspect model.gguf
```

`--skip-handling discard|omit|preserve` controls what happens to zero-width
(skipped) phones; `--skip-penalty` is the raw cosine cost of a skip (0.5
matches the reference); `--backend cpu|vulkan|cuda|metal|auto`,
`--output-formats textgrid,json`, `--fill-gaps` and `--max-frames` are also
available.  `-l` is required unless every phone is already language-qualified:
the model is multilingual and its symbols are stored as `zh/zh`, `ja/a`, ...

## 2PASS

The engine still supports the second pass of a dataset workflow: run `align`
once, edit the resulting `phones` tier (for example insert `AP`/`SP` from an
external breath detector), then re-run with `--textgrid`.  The first pass's
`texts`/`words` grouping is inherited onto the new phone boundaries.

```bash
tifa align song.wav -m models/tifa.gguf -l zh -o pass1
# edit pass1/song.TextGrid's phones tier with any tool
tifa align song.wav -m models/tifa.gguf --textgrid pass1 -l zh -o pass2
```

## Layout

```
src/                 engine: backend / gguf_io / tensor_utils / mel / ops_* / model_tifa
src/cli/             tifa (align / inspect)
include/tifa_ggml/   public C++ API (PIMPL, ggml-free headers)
scripts/             converter, reference dumps, golden comparison, quant matrix
docs/scope-change.md why 0.2.0 removed G2P / breath / UI
AGENT.md             hard-won constraints — read before changing the graph
```

## License

MPL-2.0 (this port).  TIFA's model weights keep their own license — see the
upstream release.
