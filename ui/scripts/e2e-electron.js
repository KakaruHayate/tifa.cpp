'use strict';
// ---------------------------------------------------------------------------
// Headless end-to-end check of the TIFA Label renderer.
//
//   npx electron scripts/e2e-electron.js --user-data-dir=<dir>
//
// Boots the real app (src/main.js registers the IPC surface) and drives the
// dataset workflow through executeJavaScript: detect the engine, load the
// model, import a wav from the dataset, run the CSV (known-phones) mode, and
// — when TIFA_E2E_BREATH points at a breath GGUF — the full
// align -> breathe --merge -> 2PASS chain.  Prints a JSON report and exits
// non-zero on failure.
// ---------------------------------------------------------------------------
const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');

const DATASET = process.env.TIFA_E2E_DATASET || 'path/to/dataset';
const SAMPLE = process.env.TIFA_E2E_SAMPLE || 'sample_0_0';
const MODEL = process.env.TIFA_GGML_MODEL ||
  path.join(__dirname, '..', '..', '..', 'models', 'tifa-1.0-st-f16.gguf');
const BREATH_MODEL = process.env.TIFA_E2E_BREATH || '';
const BACKEND = process.env.TIFA_E2E_BACKEND || 'vulkan';
// Never let the harness write next to the dataset: use explicit output dirs.
const OUT_DIR = path.join(os.tmpdir(), 'tifa-label-e2e-out-' + Date.now()).replace(/\\/g, '/');
const OUT_DIR_2 = OUT_DIR + '-2pass';

// A FRESH profile per run: Chromium's disk cache would otherwise serve the
// previous run's renderer bundle and silently drive dead code.
if (!process.argv.some(a => a.startsWith('--user-data-dir'))) {
  app.setPath('userData', path.join(os.tmpdir(), 'tifa-label-e2e-userdata-' + Date.now()));
}

require('../src/main.js');                       // the real app

// A deterministic long input for the cancel probe: 30 s is several seconds of
// CPU align work, so the cancel reliably lands mid-run whatever the dataset
// size is (the CI dataset is a single 3 s clip).
const CANCEL_WAV = path.join(os.tmpdir(), 'tifa-e2e-cancel-probe.wav');
function writeSineWav(file, seconds = 30, sr = 44100, hz = 220) {
  const n = Math.floor(sr * seconds);
  const data = Buffer.alloc(44 + n * 2);
  data.write('RIFF', 0); data.writeUInt32LE(36 + n * 2, 4); data.write('WAVE', 8);
  data.write('fmt ', 12); data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20);
  data.writeUInt16LE(1, 22); data.writeUInt32LE(sr, 24); data.writeUInt32LE(sr * 2, 28);
  data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
  data.write('data', 36); data.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    data.writeInt16LE(Math.round(0.3 * 32767 * Math.sin(2 * Math.PI * hz * i / sr)), 44 + i * 2);
  }
  fs.writeFileSync(file, data);
}

const checks = [];
function check(name, cond, detail) {
  checks.push({ name, ok: !!cond, detail: cond ? undefined : detail });
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (cond ? '' : '  -> ' + JSON.stringify(detail)));
}

const DRIVER = `(async () => {
  const out = { steps: [] };
  const L = window.TifaLabel;
  const S = L.S;
  const step = (s, v) => out.steps.push({ step: s, value: v });
  const $ = (id) => document.getElementById(id);
  const pause = (ms) => new Promise(r => setTimeout(r, ms));

  // --- engine + model ------------------------------------------------------
  if (!S.cliPath) await L.detectCli(true);
  step('cliPath', S.cliPath);
  await L.loadModel('${MODEL.replace(/\\/g, '/')}');
  step('model', S.modelPath);

  // --- inputs --------------------------------------------------------------
  await L.addPaths(['${DATASET.replace(/\\/g, '/')}/wavs/${SAMPLE}.wav']);
  step('inputs', S.inputs.length);
  step('sidecarKind', S.inputs[0] && S.inputs[0].sidecarKind);

  // --- source: known phones from the DiffSinger CSV ------------------------
  document.querySelector('input[name="src-mode"][value="phones"]').checked = true;
  $('sel-phones-mode').value = 'csv';
  $('csv-status').dataset.path = '${DATASET.replace(/\\/g, '/')}/transcriptions.csv';
  $('in-language').value = 'zh';
  $('sel-backend').value = '${BACKEND}';
  $('in-outdir').value = '${OUT_DIR}';
  L.refreshSourceUI();
  L.refreshRunReady();
  step('runEnabled', !$('btn-run').disabled);

  // --- run stage 1 (align) -------------------------------------------------
  await L.runAll();
  const item = S.inputs[0] || {};
  step('status', item.status);
  step('frames', item.frames);
  step('phones', item.phones);
  step('agreement', item.agreement);
  step('progressWidth', $('progress-fill').style.width);
  step('outputExists', await window.bridge.exists('${OUT_DIR}/${SAMPLE}.TextGrid').then(r => r.exists === true));

  // --- stage 2+3 (breathe --merge, then 2PASS) when a breath model is set --
  ${BREATH_MODEL ? `
  $('in-outdir').value = '${OUT_DIR_2}';
  await L.loadBreathModel('${BREATH_MODEL.replace(/\\/g, '/')}');
  $('chk-breath').checked = true;
  $('chk-pass2').checked = true;
  L.refreshSourceUI();
  L.refreshRunReady();
  step('domOutdir2', $('in-outdir').value);
  step('settingsOutdir2', L.settings().outDir);
  step('runEnabled2', !$('btn-run').disabled);
  await L.runAll();
  step('status2', S.inputs[0] && S.inputs[0].status);
  step('note2', S.inputs[0] && S.inputs[0].note);
  step('logTail', $('log').textContent.slice(-2600));
  step('frames2', S.inputs[0] && S.inputs[0].frames);
  step('agreement2', S.inputs[0] && S.inputs[0].agreement);
  step('output2Exists', await window.bridge.exists('${OUT_DIR_2}/${SAMPLE}.TextGrid').then(r => r.exists === true));
  step('output2HasBreathTier', await window.bridge.readText('${OUT_DIR_2}/${SAMPLE}.TextGrid')
      .then(t => t.ok && t.text.includes('name = "phones"')).catch(() => false));
  ` : `step('breathSkipped', 'TIFA_E2E_BREATH not set');`}

  // --- cancel kills the running child --------------------------------------
  let cancel = null;
  try {
    // CPU backend on a purpose-built 30 s wave: seconds of work, so the
    // cancel below lands mid-run instead of racing a fast finish (the CI
    // dataset is a single 3 s file, where a dir-align finishes first).
    const args = ['align', '${CANCEL_WAV.replace(/\\/g, '/')}',
                  '-m', '${MODEL.replace(/\\/g, '/')}', '--phones', 'AP SP',
                  '-o', '${OUT_DIR}', '--backend', 'cpu'];
    const pending = window.bridge.runAlign({ cliPath: S.cliPath, args,
      item: { stem: 'cancel-probe' }, outDir: '${OUT_DIR}', backend: 'cpu' });
    await pause(1200);
    const c = await window.bridge.cancelRun();
    const res = await pending;
    cancel = { killed: c.killed, cancelled: res.cancelled, ok: res.ok, code: res.code };
    await window.bridge.resetCancel();
  } catch (e) {
    cancel = { error: String(e && e.message || e) };
  }
  step('cancel', cancel);

  return out;
})()`;

app.whenReady().then(async () => {
  const wait = (ms) => new Promise(r => setTimeout(r, ms));
  if (!fs.existsSync(CANCEL_WAV)) writeSineWav(CANCEL_WAV);
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
  check('CLI detected', !!steps.cliPath, steps.cliPath);
  check('model loaded', !!steps.model, steps.model);
  check('input scanned', steps.inputs === 1, steps);
  check('sidecar detected', steps.sidecarKind === 'textgrid' || steps.sidecarKind === 'text' || steps.sidecarKind === 'none', steps.sidecarKind);
  check('run button enabled', steps.runEnabled === true, steps.runEnabled);
  check('stage 1 completed', steps.status === '完成', steps.status);
  check('frames parsed', steps.frames > 0, steps.frames);
  check('phones parsed', steps.phones > 0, steps.phones);
  check('agreement parsed', steps.agreement > 0 && steps.agreement <= 1, steps.agreement);
  check('progress bar 100%', steps.progressWidth === '100%', steps.progressWidth);
  check('output written', steps.outputExists === true, steps.outputExists);

  if (BREATH_MODEL) {
    check('breath run button enabled', steps.runEnabled2 === true, steps.runEnabled2);
    check('2PASS chain completed', steps.status2 === '完成', steps.status2);
    check('2PASS output written', steps.output2Exists === true, steps.output2Exists);
    check('2PASS output is a phones TextGrid', steps.output2HasBreathTier === true, steps.output2HasBreathTier);
  } else {
    console.log('  note  TIFA_E2E_BREATH not set - breath/2PASS stages skipped');
  }

  check('cancel killed the child', steps.cancel && steps.cancel.killed === true && steps.cancel.cancelled === true, steps.cancel);
  check('no stray output next to the dataset',
        !fs.existsSync(path.join(path.dirname(path.join(DATASET, 'wavs', SAMPLE + '.wav')), 'out')),
        path.join(DATASET, 'wavs', 'out'));

  const failed = checks.filter(c => !c.ok);
  console.log('\n' + (checks.length - failed.length) + ' passed, ' + failed.length + ' failed');
  app.exit(failed.length ? 1 : 0);
}).catch((e) => {
  console.log('harness error:', e);
  app.exit(3);
});
