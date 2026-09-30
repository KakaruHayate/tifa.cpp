# TIFA Aligner Studio

Electron desktop UI for **tifa.cpp** — a ggml port of the
[openvpi/TIFA](https://github.com/openvpi/TIFA) forced aligner.

It is an *import files → export annotations* tool: pick a model, add audio,
choose where the phones come from, run the compiled `tifa_ggml_cli` and inspect
what came out (`<name>.TextGrid` with `texts` / `words` / `phones` tiers, plus an
optional `<name>.diagnosis.json`).

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

1. **Model / Engine** — the CLI is detected at startup; **Choose model…** picks a
   `.gguf` (drag-and-drop onto the window also works) and its `inspect` output
   (architecture, feature config, vocab size, timestep, backbone) is shown. The
   last model is remembered across restarts.
2. **Inputs** — **Add files…**, **Add folder…** (recursive) or drag-drop.
   `wav` / `flac` / `mp3` are collected; the list shows each file's size, the
   sidecar that was found next to it and its per-file status.
3. **Phones source**
   - *Auto (sidecar)* — `<name>.TextGrid` beside the audio (tier `phones`), else
     `<name>.txt` / `.lab`, which means **text mode**.
   - *DiffSinger CSV* — a `transcriptions.csv`; rows are matched by file stem
     (the `name` column). Files without a row are reported before running.
   - *TextGrid folder* — the CLI's `--textgrid <dir>` scan.
   - *Inline phones* — one sequence applied to every file.
4. **Settings** — language (`-l`, default `zh`), skip handling, skip penalty,
   output directory (default `<input dir>/out`), backend (`auto|cpu|vulkan`) and
   the diagnosis-JSON toggle (`--output-formats textgrid,json`). All of it is
   persisted.
5. **Run** — files are aligned sequentially, one CLI process per file. A progress
   bar, a live stderr/stdout log pane and **Cancel** (which kills the running
   child) are provided; the renderer is never blocked.
6. **Results** — table of frames, phones, agreement, confidence, determinacy and
   monotonicity, sorted worst-first by a chosen metric (`—` when a value is
   missing). Clicking a row shows a waveform + 3-tier annotation strip: the wav
   is decoded in the renderer with Web Audio, drawn on a canvas with the TextGrid
   intervals overlaid, hovering a phone shows its label and start/end, and
   **Play** plays from the clicked position.
7. **Export** — **Open output folder** (`shell.openPath`) and **Save table as
   CSV**.

Startup never depends on a configured model: with no model the window opens and
**Align** stays disabled.

## Text mode

When a `<name>.txt` / `.lab` sits beside the audio, the UI calls the CLI with no
phone flags (`align <file> -m <model> -l <lang> …`) so that a future `--text` /
sidecar text mode is picked up automatically. The CLI in this tree has no
`--text` yet, so such a file fails with `no phone source: pass --phones/…`; that
is reported as *"text mode not available in this CLI build"* with the CLI's
stderr shown verbatim in the log.

## CLI contract used

```
tifa_ggml_cli align <audio> -m <model.gguf> -o <outdir> -l zh
  [--textgrid <file|dir>] [--phones-tier phones]
  [--transcriptions-csv <csv>] [--key <id>]
  [--phones "AP zh e n"] [--phones-file <f>]
  [--skip-handling discard|omit|preserve] [--skip-penalty F]
  [--output-formats textgrid,json] [--backend auto|cpu|vulkan|cuda|metal] [-q]
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
the text-mode failure path (`no phone source`).

**2. Renderer, headless** (real Electron window, driven through
`executeJavaScript` — no clicking required):

```
npm run e2e [-- --user-data-dir=<dir>]
```

Boots the app, then adds the sample wav, indexes the CSV, aligns it, and checks
the resulting row, the progress bar, the decoded waveform/peaks, the three
annotation lanes, the hover tooltip (`label + start/end`), play/pause and the
CSV export.  It writes to an explicit temp output directory and leaves the
dataset untouched.  Options: `TIFA_E2E_DATASET`, `TIFA_E2E_SAMPLE`,
`TIFA_GGML_MODEL`.

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
