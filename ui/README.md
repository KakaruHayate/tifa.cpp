# TIFA Label

Electron desktop tool for **tifa.cpp** — a ggml port of the
[openvpi/TIFA](https://github.com/openvpi/TIFA) forced aligner — built for
**dataset annotation**, not for creative editing (that is what game.cpp's
studio is for).

It drives the full dataset pipeline in batches:

1. **导入音频 / Import audio** — files or folders (scanned recursively).
2. **标注来源 / Annotation source** — text/G2P (sidecar `<name>.txt`/`.lab`,
   a uniform transcript, PFML fragments) or existing phones (same-name
   TextGrid, a DiffSinger `transcriptions.csv`, a TextGrid folder, an inline
   list).  The old `.lab + wav` workflow never touches G2P.
3. **数据集流程 / Pipeline** — TIFA align → (optional) BreathLab breath AP/SP
   detection merged into the phones tier → (optional) 2PASS re-align.

Per-file status with agreement, progress bar, streaming log, cancel.  The UI
is in Chinese.  See `docs/dataset-workflow.md` for the pipeline rationale.

The per-file output is `<name>.TextGrid` (with `texts` / `words` / `phones`
tiers, plus a `breath` tier after the merge stage) and optionally
`<name>.diagnosis.json`.

```
npm install
npm start
```

`npm run smoke` runs the headless logic test and `npm run e2e` drives the real
renderer headlessly (see *Verification* below).

If `npm install` stalls on Electron's ~110 MB binary (GitHub release downloads
can be throttled to a crawl), point it at a mirror — the postinstall honours
`ELECTRON_MIRROR`:

```
ELECTRON_MIRROR=https://cdn.npmmirror.com/binaries/electron/ npm install
```

(or re-run just the binary step:
`ELECTRON_MIRROR=… node node_modules/electron/install.js`).

## Requirements

- Node 20+ and npm (developed on Node 24 / npm 11).
- A TIFA GGUF model, e.g. `models/tifa-1.0-st-f16.gguf`.
- The engine CLI, `tifa_ggml_cli` / `tifa_ggml_cli.exe`. The app looks for it,
  in order, at:
  1. the path stored in its config (`app.getPath('userData')/config.json`),
  2. `$TIFA_GGML_CLI`,
  3. `<repo>/build/bin/`, `../build/bin/`, `./bin/`, `<ui>/bin/`,
  4. the current working directory and the directory of the app executable.

  Use **Locate CLI…** if it is somewhere else; the choice is remembered.

## Layout

```
ui/
├── package.json
├── src/
│   ├── main.js              main process: dialogs, fs, spawn, config, IPC
│   ├── preload.js           contextBridge surface (window.bridge)
│   ├── lib/parse.js         pure logic, shared by main + renderer + tests
│   └── renderer/
│       ├── index.html
│       ├── app.js
│       └── style.css
└── scripts/
    ├── smoke-test.js        headless test of src/lib/parse.js (+ --cli)
    └── e2e-electron.js      drives the real renderer headlessly
```

`contextIsolation: true`, `nodeIntegration: false`; the renderer only ever
touches `window.bridge`. Reads of audio/TextGrid/JSON from the renderer are
restricted in the main process to roots the user actually picked (input
directories, output directory, model, CSV, TextGrid folder).

## Workflow

1. **Engine / models** — the CLI is detected at startup (**定位引擎…** overrides
   it).  **选择对齐模型…** picks the TIFA `.gguf`; **选择呼吸模型…** picks the
   BreathLab breath `.gguf` required by the breath/2PASS stages.  Both are
   remembered across restarts.
2. **① 导入音频** — **添加文件…**, **添加文件夹…** (recursive) or drag-drop;
   the list shows each file's size, the sidecar it found and its status.
3. **② 标注来源 / annotation source**
   - *文本转录（自动 G2P）* — text per file (`<name>.txt` / `.lab` beside the
     audio) or one uniform transcript; PFML fragments are accepted verbatim
     (`<word phonemes="zh ong">重</word>` skips the dictionary entirely).
   - *已有音素标注* — same-name `<name>.TextGrid` (auto), a DiffSinger
     `transcriptions.csv`, a TextGrid folder, or one inline phone sequence
     applied to every file.  This path never invokes G2P, so the classic
     `.lab + wav` workflow stays untouched.
4. **③ 数据集流程 / pipeline** — first-pass align (always), BreathLab breath
   detection merged into the phones tier (optional, needs the breath model),
   2PASS re-align (optional, needs the breath stage).  Advanced options:
   diagnosis JSON, backend, zero-width handling, skip penalty, quiet.
5. **Run** — per file, the enabled stages run in order as separate CLI
   processes; the list shows live status and agreement, a progress bar covers
   the batch, the log pane streams stdout/stderr and **取消** kills the
   running child.

Outputs per file: `<name>.TextGrid` (plus the `breath` tier after the merge
stage) and, when enabled, `<name>.diagnosis.json`.

## CLI contract used

```
tifa_ggml_cli align <audio> -m <model.gguf> -o <outdir> -l zh
  [--text STR | --text-file <f>]                  # text/G2P (PFML accepted)
  [--textgrid <file|dir>] [--phones-tier phones]
  [--transcriptions-csv <csv>] [--key <id>]
  [--phones "AP zh e n"] [--phones-file <f>]
  [--oov-handling raise|discard|force]
  [--skip-handling discard|omit|preserve] [--skip-penalty F]
  [--output-formats textgrid,json] [--backend auto|cpu|vulkan|cuda|metal] [-q]

tifa_ggml_cli breathe <audio> -m <breath.gguf> -o <outdir>
  [--merge <file|dir>] [--min-insert-ms 50] [-q]
```

Writes `<outdir>/<stem>.TextGrid` and, with `json`, `<outdir>/<stem>.diagnosis.json`:

```json
{ "identifier": "sample_0_0", "num_frames": 870, "num_tokens": 49,
  "agreement": 0.999755, "confidence": 0, "determinacy": 0, "monotonicity": 0 }
```

The process exit code is `1` when *any* file was skipped or failed, so the
result of a run is decided by the presence of the output files, not by the exit
code alone. `TIFA_GGML_BACKEND` is exported for the child as well as passed via
`--backend`. Lines from `ggml_vulkan:` are counted rather than logged, since the
CLI prints them for every file.

## Verification

**1. Pure logic** (fast, no Electron, no CLI):

```
node scripts/smoke-test.js [datasetDir]      # default path/to/dataset
node scripts/smoke-test.js --cli             # + a live align run through the real engine
```

It exercises `src/lib/parse.js` against the real files in the dataset:
`wavs/sample_0_0.TextGrid` (49 `phones` intervals), a `transcriptions.csv` row
(`ph_seq` / `ph_dur` consistency), the 3-tier TextGrid shape the CLI writes
(incl. the short `ooTextFile` variant and `""` escaping), diagnosis-JSON
parsing, `inspect` parsing, sidecar resolution, CLI argument construction,
worst-first table sorting and CSV export.  With `--cli` it additionally runs
`tifa_ggml_cli` on `sample_0_0.wav` through `buildAlignArgs`, checks the exact
output file names and the diagnosis schema, builds the results row and confirms
the text-mode path (`--text` + sidecar .txt/`.lab`).

**2. Renderer, headless** (real Electron window, driven through
`executeJavaScript` — no clicking required):

```
npm run e2e [-- --user-data-dir=<dir>]
```

Boots the app and drives the dataset workflow: engine detection, model load,
adding a wav, configuring the CSV source, running stage 1, then (when
`TIFA_E2E_BREATH` is set) the full align -> `breathe --merge` -> 2PASS chain,
checking per-file status/agreement, the progress bar, the written TextGrid and
the cancel path.  It writes to explicit temp output directories and leaves the
dataset untouched.  Options: `TIFA_E2E_DATASET`, `TIFA_E2E_SAMPLE`,
`TIFA_GGML_MODEL`, `TIFA_E2E_BREATH`, `TIFA_E2E_BACKEND`.

**3. GUI** — `npm start` opens the window; there is no screenshot/automation
harness here for mouse interaction, so button clicks and the file dialogs are
the only parts not covered by the two scripts above.

`TIFA_UI_LOG=1` mirrors the renderer console into the terminal (via
`console-message` in `src/main.js`), which is how the app can be smoke-checked
without looking at the window; the renderer logs a `[tifa] ready · …` line with
the detected CLI, the restored model and whether Align is enabled.

## Packaging

`npm run dist:win` / `dist:mac` / `dist:linux` (electron-builder, `appId
com.kakaru.tifa-aligner`). The package ships `src/**` only; the model and the
CLI are supplied by the user and are not bundled.

## License

MIT.
