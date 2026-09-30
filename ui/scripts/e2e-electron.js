'use strict';
// ---------------------------------------------------------------------------
// Headless end-to-end check of the renderer.
//
//   npx electron scripts/e2e-electron.js --user-data-dir=<dir>
//
// Boots the real app (src/main.js registers the IPC surface), then drives the
// renderer through executeJavaScript: add an input file, align it, build the
// result row, render the waveform + annotation strip, hover a phone interval
// and start playback.  Prints a JSON report and exits non-zero on failure.
// ---------------------------------------------------------------------------
const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');

const DATASET = process.env.TIFA_E2E_DATASET || 'path/to/dataset';
const SAMPLE = process.env.TIFA_E2E_SAMPLE || 'sample_0_0';
const MODEL = process.env.TIFA_GGML_MODEL ||
  path.join(__dirname, '..', '..', '..', 'models', 'tifa-1.0-st-f16.gguf');
// Never let the harness write next to the dataset: use an explicit output dir.
const OUT_DIR = path.join(os.tmpdir(), 'tifa-ui-e2e-out-' + Date.now()).replace(/\\/g, '/');

// Keep the harness out of the real user profile unless the caller overrides it.
if (!process.argv.some(a => a.startsWith('--user-data-dir'))) {
  app.setPath('userData', path.join(os.tmpdir(), 'tifa-ui-e2e-userdata'));
}

require('../src/main.js');                       // the real app

const checks = [];
function check(name, cond, detail) {
  checks.push({ name, ok: !!cond, detail: cond ? undefined : detail });
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (cond || detail === undefined ? '' : '  -> ' + JSON.stringify(detail)));
}

const DRIVER = `(async () => {
  const out = { steps: [] };
  const step = (s, v) => out.steps.push({ step: s, value: v });

  // --- engine + model (a fresh profile has neither configured) ------------
  step('cliPath', S.cli.path);
  if (!S.cli.path) await detectCli(true);
  step('cliPath2', S.cli.path);
  if (!S.model.path) await loadModel('${MODEL.replace(/\\/g, '/')}');
  step('model', S.model.path);
  step('inspectFields', S.model.fields.map(f => f.key));

  // --- inputs -------------------------------------------------------------
  await addPaths(['${DATASET.replace(/\\/g, '/')}/wavs/${SAMPLE}.wav']);
  step('inputs', S.inputs.length);
  step('sidecarKind', S.inputs[0] && S.inputs[0].sidecarKind);
  step('sidecarPath', S.inputs[0] && S.inputs[0].sidecarPath);

  // --- settings: DiffSinger CSV ------------------------------------------
  S.settings.phonesMode = 'csv';
  S.settings.csvPath = '${DATASET.replace(/\\/g, '/')}/transcriptions.csv';
  S.settings.language = 'zh';
  S.settings.backend = 'vulkan';
  S.settings.exportJson = true;
  // Same as choosing it in the picker: persist it, then index it.
  await window.bridge.setConfig({ csvPath: S.settings.csvPath });
  step('csvIndexed', await loadCsvIndex(S.settings.csvPath));
  step('csvRows', S.csv.count);
  step('csvHasSample', csvHas('${SAMPLE}'));
  step('plan', planFor(S.inputs[0]));
  step('argv', P.buildAlignArgs({ path: S.inputs[0].path, kind: 'csv', key: '${SAMPLE}' },
                                Object.assign({}, S.settings, { modelPath: S.model.path })));

  // --- run ----------------------------------------------------------------
  S.settings.outDir = '${OUT_DIR}';
  await runAll();
  step('outDirUsed', S.run.outDir);
  const row = S.results[0] || {};
  step('results', S.results.length);
  step('row', { name: row.name, frames: row.frames, phones: row.phones,
                agreement: row.agreement, status: row.status, error: row.error && row.error.slice(0, 120),
                tgPath: row.tgPath });
  step('progressWidth', document.getElementById('progress-bar').style.width);
  step('inputStatus', S.inputs[0].status);

  // --- preview ------------------------------------------------------------
  if (S.results[0]) await selectRow(S.results[0], true);
  step('audio', S.audio && S.audio.buffer ? {
    duration: Number(S.audio.duration.toFixed(3)), sampleRate: S.audio.buffer.sampleRate,
    peaks: S.audio.peaks ? S.audio.peaks.length / 2 : 0 } : null);
  step('gridPath', S.audio && S.audio.gridPath);
  step('gridTiers', S.audio && S.audio.grid ? S.audio.grid.tiers.map(t => t.name + ':' + t.intervals.length) : null);
  step('stripLanes', S.strip ? S.strip.lanes.map(l => l.name + ':' + l.intervals.length) : null);
  step('waveformCanvasPx', [document.getElementById('waveform').width, document.getElementById('waveform').height]);
  step('stripCanvasPx', [document.getElementById('strip').width, document.getElementById('strip').height]);
  step('playLabel', document.getElementById('btn-play').textContent);

  // hover a phone interval (the middle of the 3rd interval of the phones lane)
  const cv = document.getElementById('strip');
  const rect = cv.getBoundingClientRect();
  const phones = S.strip && S.strip.lanes.find(l => l.name === 'phones');
  let hover = null;
  if (phones && phones.intervals.length > 3) {
    const bar = phones.intervals[3];
    const cx = rect.left + (bar.xmin + bar.xmax) / 2 / S.strip.duration * rect.width;
    const cy = rect.top + phones.y + phones.h / 2;
    cv.dispatchEvent(new MouseEvent('mousemove', { clientX: cx, clientY: cy, bubbles: true }));
    const tip = document.getElementById('tooltip');
    hover = { hidden: tip.classList.contains('hidden'), text: tip.textContent,
              highlight: S.hover && S.hover.id, expected: bar.id, label: bar.text,
              t0: bar.xmin, t1: bar.xmax };
  }
  step('hover', hover);

  // --- click-to-seek at 50% of the strip ----------------------------------
  let seek = null;
  if (phones) {
    const cx = rect.left + rect.width * 0.5;
    const cy = rect.top + phones.y + phones.h / 2;
    cv.dispatchEvent(new MouseEvent('mousedown', { clientX: cx, clientY: cy, bubbles: true }));
    seek = { offset: S.audio.playOffset, expected: S.audio.duration * 0.5 };
    step('seek', seek);
  }

  // --- playback -----------------------------------------------------------
  let playback = null;
  try {
    await togglePlay();
    await new Promise(r => setTimeout(r, 350));
    const t = playheadTime();
    playback = { playing: S.audio.playing, advanced: t != null && t > 0.05,
                 startedNearSeek: seek ? Math.abs(t - seek.offset) < 0.3 : null,
                 t: t == null ? null : Number(t.toFixed(3)),
                 ctxState: S.audio.ctx ? S.audio.ctx.state : null,
                 label: document.getElementById('btn-play').textContent };
    await togglePlay();
    playback.paused = !S.audio.playing;
  } catch (e) {
    playback = { error: String(e && e.message || e) };
  }
  step('playback', playback);

  // Is the AudioContext clock itself running?  Without an audio device the
  // clock can stall, which makes the playhead look frozen.
  if (S.audio && S.audio.ctx) {
    const c0 = S.audio.ctx.currentTime;
    await new Promise(r => setTimeout(r, 500));
    const c1 = S.audio.ctx.currentTime;
    step('ctxClock', { c0: Number(c0.toFixed(3)), c1: Number(c1.toFixed(3)), delta: Number((c1 - c0).toFixed(3)) });
  }

  // --- sorting + csv export ----------------------------------------------
  step('sortMetric', S.settings.sortMetric);
  step('ordered', orderResults().map(r => r.name));
  step('csvExport', P.toCsv(orderResults()).split('\\n').slice(0, 2));

  // --- cancel kills the running child -------------------------------------
  let cancel = null;
  try {
    const args2 = P.buildAlignArgs({ path: S.inputs[0].path, kind: 'csv', key: '${SAMPLE}' },
      Object.assign({}, S.settings, { modelPath: S.model.path, outDir: '${OUT_DIR}' }));
    const pending = window.bridge.runAlign({ cliPath: S.cli.path, args: args2,
      item: { stem: 'cancel-probe' }, outDir: '${OUT_DIR}', backend: S.settings.backend });
    await new Promise(r => setTimeout(r, 1500));
    const c = await window.bridge.cancelRun();
    const res = await pending;
    cancel = { killed: c.killed, cancelled: res.cancelled, ok: res.ok, code: res.code, ms: res.durationMs };
    await window.bridge.resetCancel();
  } catch (e) {
    cancel = { error: String(e && e.message || e) };
  }
  step('cancel', cancel);
  return out;
})()`;

app.whenReady().then(async () => {
  const wait = (ms) => new Promise(r => setTimeout(r, ms));
  await wait(2500);                                   // let the renderer boot
  const win = BrowserWindow.getAllWindows()[0];
  if (!win) { console.log('no window'); app.exit(2); return; }
  let out = null, err = null;
  try {
    out = await win.webContents.executeJavaScript(DRIVER, true);
  } catch (e) {
    err = String((e && e.message) || e);
  }
  console.log('\n--- renderer report ---');
  console.log(JSON.stringify(out, null, 2));
  if (err) console.log('driver error:', err);

  const steps = {};
  for (const s of (out && out.steps) || []) steps[s.step] = s.value;

  console.log('\n--- checks ---');
  check('driver ran', !err, err);
  check('CLI detected', !!steps.cliPath2, steps.cliPath2 || steps.cliPath);
  check('model loaded', !!steps.model, steps.model);
  check('input scanned', steps.inputs === 1, steps);
  check('sidecar TextGrid found', steps.sidecarKind === 'textgrid', steps.sidecarKind);
  check('csv index loaded', steps.csvRows > 1000 && steps.csvHasSample === true, steps);
  check('csv plan', steps.plan && steps.plan.kind === 'csv', steps.plan);
  check('argv has csv + backend', Array.isArray(steps.argv) &&
        steps.argv.includes('--transcriptions-csv') && steps.argv.includes('vulkan') &&
        steps.argv.includes('--output-formats'), steps.argv);
  check('run produced a row', steps.results === 1, steps.results);
  check('row parsed', steps.row && steps.row.frames > 0 && steps.row.phones > 0, steps.row);
  check('row agreement parsed', steps.row && steps.row.agreement > 0 && steps.row.agreement <= 1, steps.row);
  check('row status ok', steps.row && /^ok/.test(steps.row.status), steps.row);
  check('input list status ok', steps.inputStatus === 'ok', steps.inputStatus);
  check('progress bar 100%', steps.progressWidth === '100%', steps.progressWidth);
  check('output written to the configured dir', steps.outDirUsed === OUT_DIR, steps.outDirUsed);
  check('no stray output next to the dataset',
        !fs.existsSync(path.join(path.dirname(path.join(DATASET, 'wavs', SAMPLE + '.wav')), 'out')),
        path.join(DATASET, 'wavs', 'out'));
  check('audio decoded', steps.audio && steps.audio.duration > 1, steps.audio);
  check('waveform peaks computed', steps.audio && steps.audio.peaks > 100, steps.audio);
  check('grid loaded for preview', steps.gridTiers && steps.gridTiers.length >= 1, steps.gridTiers);
  check('strip has 3 lanes', steps.stripLanes && steps.stripLanes.length === 3, steps.stripLanes);
  check('canvases sized', steps.waveformCanvasPx && steps.waveformCanvasPx[0] > 0 &&
        steps.stripCanvasPx && steps.stripCanvasPx[1] > 0, [steps.waveformCanvasPx, steps.stripCanvasPx]);
  check('hover shows label + times', steps.hover && !steps.hover.hidden &&
        /s/.test(steps.hover.text || ''), steps.hover);
  check('hover highlights the same bar', steps.hover && steps.hover.highlight === steps.hover.expected, steps.hover);
  const stalledClock = steps.ctxClock && steps.ctxClock.delta < 0.05;
  check('playback starts a source', steps.playback && steps.playback.playing === true, steps.playback);
  check('playhead advances (or the audio clock itself is stalled -> not verifiable here)',
        steps.playback && (steps.playback.advanced || stalledClock),
        { playback: steps.playback, ctxClock: steps.ctxClock });
  check('pause works', steps.playback && steps.playback.paused !== false, steps.playback);
  check('inspect parsed at startup', Array.isArray(steps.inspectFields) && steps.inspectFields.length > 3, steps.inspectFields);
  check('csv export rows', Array.isArray(steps.csvExport) && steps.csvExport.length === 2, steps.csvExport);
  check('click on the strip seeks', steps.seek && Math.abs(steps.seek.offset - steps.seek.expected) < 0.05, steps.seek);
  check('playback starts from the clicked position', steps.playback && steps.playback.startedNearSeek === true, steps.playback);
  check('cancel kills the child', steps.cancel && steps.cancel.killed === true &&
        steps.cancel.cancelled === true && steps.cancel.ok === false, steps.cancel);

  const failed = checks.filter(c => !c.ok).length;
  console.log('\n' + (failed ? 'E2E FAILED' : 'E2E PASSED') + ' · ' + (checks.length - failed) + ' passed, ' + failed + ' failed');
  try { fs.rmSync(OUT_DIR, { recursive: true, force: true }); } catch { /* ignore */ }
  app.exit(failed ? 1 : 0);
}).catch((e) => { console.error('e2e boot error:', e); app.exit(2); });
