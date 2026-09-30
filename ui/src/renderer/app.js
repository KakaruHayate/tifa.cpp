'use strict';
// ---------------------------------------------------------------------------
// TIFA Aligner Studio — renderer.
// No Node APIs here: everything goes through window.bridge (see preload.js).
// Pure helpers come from ../lib/parse.js (globalThis.TifaParse).
// ---------------------------------------------------------------------------
const P = window.TifaParse;
const $ = (id) => document.getElementById(id);

const MAX_INPUT_ROWS = 400;
const MAX_LOG_LINES = 400;

const S = {
  cli: { path: null, version: null },
  model: { path: null, name: null, size: null, fields: [], raw: '' },
  inputs: [],
  csv: { path: null, keys: null, count: 0, header: null },
  settings: {
    phonesMode: 'auto',
    phonesTier: 'phones',
    csvPath: null,
    textgridDir: null,
    inlinePhones: '',
    language: 'zh',
    skipHandling: 'omit',
    skipPenalty: '0.5',
    outDir: '',
    backend: 'auto',
    exportJson: true,
    sortMetric: 'agreement',
    sortDir: 'asc',
  },
  run: { running: false, cancel: false, total: 0, done: 0, outDir: null },
  results: [],
  sel: null,                                // selected result row
  audio: null,                              // { path, ctx, buffer, duration, peaks, playing, ... }
  logLines: [],
};

// ---------------------------------------------------------------------------
// small UI helpers
// ---------------------------------------------------------------------------
function setDot(elId, state) {
  const el = $(elId);
  if (el) el.className = 'dot' + (state === 'ok' ? ' ok' : state === 'bad' ? ' bad' : state === 'run' ? ' running' : '');
}

function log(msg) {
  const stamp = new Date().toLocaleTimeString();
  for (const line of String(msg).split(/\r?\n/)) {
    if (!line.length) continue;
    S.logLines.unshift(`[${stamp}] ${line}`);
  }
  if (S.logLines.length > MAX_LOG_LINES) S.logLines.length = MAX_LOG_LINES;
  const el = $('log');
  if (el) el.textContent = S.logLines.join('\n');
}

function toast(id, text, dotState) {
  const el = $(id);
  if (el) el.textContent = text;
}

function fmtBytes(n) {
  if (n == null || !isFinite(n)) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0, v = Number(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? v : v.toFixed(1)) + ' ' + u[i];
}

function pill(text, cls) {
  const s = document.createElement('span');
  s.className = 'pill' + (cls ? ' ' + cls : '');
  s.textContent = text;
  return s;
}

// ---------------------------------------------------------------------------
// settings persistence
// ---------------------------------------------------------------------------
let saveTimer = null;
function persist() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    window.bridge.setConfig({
      modelPath: S.model.path || undefined,
      cliPath: S.cli.path || undefined,
      csvPath: S.settings.csvPath || undefined,
      textgridDir: S.settings.textgridDir || undefined,
      settings: S.settings,
    }).catch(() => {});
  }, 400);
}

function applyConfig(cfg) {
  const s = cfg && cfg.settings ? cfg.settings : {};
  Object.assign(S.settings, s);
  S.settings.exportJson = S.settings.exportJson !== false;
  S.csv.path = S.settings.csvPath || null;
  if (S.settings.textgridDir) $('tgdir-status').textContent = S.settings.textgridDir;
  if (S.settings.csvPath) $('csv-status').textContent = P.baseName(S.settings.csvPath);
}

function bindSetting(elId, key, kind) {
  const el = $(elId);
  if (!el) return;
  el.addEventListener(kind || 'change', () => {
    S.settings[key] = el.type === 'checkbox' ? el.checked : el.value;
    if (key === 'phonesTier') $('in-phones-tier-2').value = el.value;
    persist();
    if (key === 'phonesMode') refreshPhonesUI();
    refreshRunReady();
  });
}

// ---------------------------------------------------------------------------
// CLI + model
// ---------------------------------------------------------------------------
async function detectCli(quiet) {
  const r = await window.bridge.detectCli();
  if (r && r.ok) {
    S.cli.path = r.cliPath;
    S.cli.version = r.version;
    $('cli-status').textContent = `${r.cliPath}${r.version ? '  ·  ' + r.version : ''}`;
    setDot('cliDot', r.versionOk === false ? '' : 'ok');
    if (!quiet) log('engine: ' + r.cliPath);
  } else {
    S.cli.path = null;
    $('cli-status').textContent = (r && r.error) || 'tifa_ggml_cli not found — use “Locate CLI…”';
    setDot('cliDot', 'bad');
    if (!quiet) log('engine not found: ' + ((r && r.error) || 'unknown'));
  }
  refreshRunReady();
}

async function loadModel(modelPath, quiet) {
  if (!modelPath) return;
  const info = await window.bridge.modelInfo(modelPath);
  if (!info || !info.ok) {
    $('model-status').textContent = (info && info.error) || 'model not found';
    setDot('modelDot', 'bad');
    S.model.path = null;
    $('btn-inspect').disabled = true;
    refreshRunReady();
    return;
  }
  S.model.path = info.modelPath;
  S.model.name = info.name;
  S.model.size = info.size;
  $('model-status').textContent = `${info.name}  ·  ${fmtBytes(info.size)}`;
  setDot('modelDot', 'ok');
  $('btn-inspect').disabled = false;
  await window.bridge.setModel(info.modelPath);
  await runInspect(quiet);
  refreshRunReady();
}

async function runInspect(quiet) {
  if (!S.model.path) return;
  $('btn-inspect').disabled = true;
  toast('model-status', 'inspecting ' + S.model.name + '…');
  const r = await window.bridge.inspectModel(S.model.path);
  $('btn-inspect').disabled = false;
  const box = $('inspect-box');
  if (!r || !r.ok) {
    if (!quiet) log('inspect failed: ' + ((r && r.error) || 'unknown'));
    return;
  }
  S.model.fields = r.fields || [];
  S.model.raw = r.raw || '';
  const dl = $('inspect-fields');
  dl.innerHTML = '';
  for (const f of S.model.fields) {
    const dt = document.createElement('dt'); dt.textContent = f.key;
    const dd = document.createElement('dd'); dd.textContent = f.value;
    dl.appendChild(dt); dl.appendChild(dd);
  }
  $('inspect-raw').textContent = S.model.raw.trim();
  box.classList.remove('hidden');
  const arch = S.model.fields.find(f => /^architecture$/i.test(f.key));
  $('model-status').textContent =
    `${S.model.name}  ·  ${fmtBytes(S.model.size)}${arch ? '  ·  ' + arch.value : ''}`;
  setDot('modelDot', 'ok');
  if (!quiet) log('inspect: ' + S.model.fields.map(f => f.key + '=' + f.value).slice(0, 4).join('  '));
}

function modelTimestep() {
  const f = (S.model.fields || []).find(x => /^timestep$/i.test(x.key));
  if (!f) return 0.01;
  const m = String(f.value).match(/([\d.]+)/);
  const v = m ? Number(m[1]) : 0.01;
  return isFinite(v) && v > 0 ? v : 0.01;
}

// ---------------------------------------------------------------------------
// inputs
// ---------------------------------------------------------------------------
function mergeInputs(recs) {
  const seen = new Set(S.inputs.map(i => i.path.toLowerCase()));
  let added = 0;
  for (const r of recs || []) {
    if (seen.has(r.path.toLowerCase())) continue;
    seen.add(r.path.toLowerCase());
    r.status = 'pending';
    r.message = '';
    S.inputs.push(r);
    added++;
  }
  return added;
}

function renderInputs() {
  const tb = $('input-list').querySelector('tbody');
  tb.innerHTML = '';
  const rows = S.inputs.slice(0, MAX_INPUT_ROWS);
  for (const it of rows) {
    const tr = document.createElement('tr');
    tr.title = it.path;

    const tdName = document.createElement('td');
    tdName.className = 'cell-name';
    tdName.textContent = it.name;

    const tdSize = document.createElement('td');
    tdSize.className = 'num';
    tdSize.textContent = fmtBytes(it.sizeBytes);

    const tdSide = document.createElement('td');
    const side = it.sidecarKind === 'textgrid' ? 'TextGrid'
      : it.sidecarKind === 'text' ? (P.baseName(it.sidecarPath || '') + ' (text mode)')
        : '—';
    tdSide.appendChild(pill(side, it.sidecarKind === 'none' ? 'skip' : ''));

    const tdStatus = document.createElement('td');
    const map = {
      pending: ['pending', 'skip'], running: ['running…', 'run'], ok: ['ok', 'ok'],
      failed: ['failed', 'bad'], skipped: ['skipped', 'skip'], cancelled: ['cancelled', 'skip'],
    };
    const m = map[it.status] || map.pending;
    tdStatus.appendChild(pill(it.message || m[0], m[1]));

    tr.appendChild(tdName); tr.appendChild(tdSize); tr.appendChild(tdSide); tr.appendChild(tdStatus);
    tb.appendChild(tr);
  }
  const more = $('input-more');
  if (S.inputs.length > MAX_INPUT_ROWS) {
    more.classList.remove('hidden');
    more.textContent = `… and ${S.inputs.length - MAX_INPUT_ROWS} more (all files are still processed)`;
  } else {
    more.classList.add('hidden');
  }
  $('input-count').textContent = `${S.inputs.length} file${S.inputs.length === 1 ? '' : 's'}`;
  refreshRunReady();
}

async function addPaths(paths) {
  if (!paths || !paths.length) return;
  const r = await window.bridge.scanPaths(paths);
  if (!r || !r.ok) { log('scan failed: ' + ((r && r.error) || 'unknown')); return; }
  const added = mergeInputs(r.list);
  if (r.truncated) log('scan truncated at the file limit');
  log(`added ${added} file(s) (${S.inputs.length} total)`);
  renderInputs();
}

async function addFileObjects(files) {
  const paths = [];
  for (const f of files || []) {
    const p = window.bridge.pathForFile(f);
    if (p) paths.push(p);
  }
  await addPaths(paths);
}

// ---------------------------------------------------------------------------
// phone source UI
// ---------------------------------------------------------------------------
function refreshPhonesUI() {
  const mode = S.settings.phonesMode;
  $('pane-csv').classList.toggle('hidden', mode !== 'csv');
  $('pane-textgrid-dir').classList.toggle('hidden', mode !== 'textgrid-dir');
  $('pane-inline').classList.toggle('hidden', mode !== 'inline');
  const notes = {
    auto: 'looks for <name>.TextGrid beside each audio file (tier “phones”), then <name>.txt / .lab for text mode',
    csv: 'rows matched to each file stem',
    'textgrid-dir': 'a folder scanned as <name>.TextGrid per file',
    inline: 'one phone sequence applied to every file',
  };
  $('phones-status').textContent = notes[mode] || '';
  setDot('phonesDot', mode === 'auto' ? '' : 'ok');
  $('textmode-note').classList.toggle('hidden', mode !== 'auto');
}

async function pickCsv(path) {
  if (path) { S.settings.csvPath = path; S.csv.path = path; }
  const r = await window.bridge.pickCsv();
  if (!r || !r.ok) {
    if (r && !r.canceled) log('csv: ' + r.error);
    return;
  }
  S.settings.csvPath = r.csvPath;
  S.csv = { path: r.csvPath, keys: r.keys, count: r.count, header: r.header };
  $('csv-status').textContent = `${P.baseName(r.csvPath)} · ${r.count} rows`;
  log(`csv loaded: ${r.csvPath} (${r.count} rows, columns: ${(r.header || []).join(', ')})`);
  persist();
  // refresh the sidecar column so users see matches immediately
  for (const it of S.inputs) it.message = '';
  renderInputs();
  refreshRunReady();
}

// Rebuild the CSV index for a path already known to the config (no dialog).
async function loadCsvIndex(csvPath) {
  const r = await window.bridge.readText(csvPath);
  if (!r || !r.ok) return false;
  const idx = P.csvIndex(r.text, 'name');
  S.csv = { path: csvPath, keys: idx.keys, count: idx.count, header: idx.header };
  $('csv-status').textContent = `${P.baseName(csvPath)} · ${idx.count} rows`;
  log(`csv index: ${idx.count} rows from ${csvPath}`);
  return true;
}

async function pickTextgridDir() {
  const r = await window.bridge.pickTextgridDir();
  if (!r || !r.ok) {
    if (r && !r.canceled) log('textgrid dir: ' + r.error);
    return;
  }
  S.settings.textgridDir = r.textgridDir;
  $('tgdir-status').textContent = `${r.textgridDir} · ${r.count} .TextGrid`;
  log(`textgrid folder: ${r.textgridDir} (${r.count} files)`);
  persist();
  refreshRunReady();
}

// ---------------------------------------------------------------------------
// per-file plan + run
// ---------------------------------------------------------------------------
function csvHas(stem) {
  const keys = S.csv.keys;
  if (!keys) return true;                       // unknown -> let the CLI decide
  if (keys[stem]) return true;
  const lower = stem.toLowerCase();
  return Object.keys(keys).some(k => k.toLowerCase() === lower);
}

function planFor(it) {
  const s = S.settings;
  switch (s.phonesMode) {
    case 'auto':
      if (it.sidecarKind === 'textgrid') return { kind: 'textgrid', tgPath: it.sidecarPath };
      if (it.sidecarKind === 'text') return { kind: 'text' };
      return { kind: 'none', error: 'no phone source beside the audio' };
    case 'csv':
      if (!s.csvPath) return { kind: 'csv', error: 'no transcriptions.csv selected' };
      if (!csvHas(it.stem)) return { kind: 'csv', error: `no row “${it.stem}” in the CSV` };
      return { kind: 'csv' };
    case 'textgrid-dir':
      if (!s.textgridDir) return { kind: 'textgrid-dir', error: 'no TextGrid folder selected' };
      return { kind: 'textgrid-dir' };
    case 'inline':
      if (!String(s.inlinePhones || '').trim()) return { kind: 'inline', error: 'inline phones are empty' };
      return { kind: 'inline' };
    default:
      return { kind: 'none', error: 'unknown phone source mode' };
  }
}

function refreshRunReady() {
  const ok = !!(S.cli.path && S.model.path && S.inputs.length && !S.run.running);
  $('btn-run').disabled = !ok;
  $('btn-cancel').disabled = !S.run.running;
  return ok;
}

function setProgress(done, total) {
  const pct = total ? Math.round((done / total) * 100) : 0;
  $('progress-bar').style.width = pct + '%';
}

function stderrTail(text, n) {
  const lines = String(text || '').split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  return lines.slice(-(n || 2)).join(' ');
}

async function runAll() {
  if (!refreshRunReady() || S.run.running) return;
  const files = S.inputs.slice();
  const s = S.settings;
  const outDir = s.outDir && s.outDir.trim() ? s.outDir.trim() : P.joinPath(files[0].dir, 'out');

  const ensured = await window.bridge.ensureDir(outDir);
  if (!ensured || !ensured.ok) { log('cannot use output dir: ' + ((ensured && ensured.error) || 'unknown')); return; }

  S.run.running = true;
  S.run.cancel = false;
  S.run.total = files.length;
  S.run.done = 0;
  S.run.outDir = outDir;
  S.results = [];
  S.sel = null;
  renderResults();
  setProgress(0, files.length);
  refreshRunReady();
  await window.bridge.resetCancel();
  $('btn-open-out').disabled = false;
  log(`run start · ${files.length} file(s) · out=${outDir}`);
  setDot('runDot', 'run');
  toast('run-status', `running 0/${files.length}`);

  for (let i = 0; i < files.length; i++) {
    if (S.run.cancel) break;
    const it = files[i];
    // Re-rendering the tables for every one of 1000s of files would dominate
    // the runtime, so throttle to every 5th file (plus every failure).
    const refresh = (i % 5 === 0) || (i + 1 === files.length);
    it.status = 'running';
    it.message = '';
    if (refresh) renderInputs();
    toast('run-status', `running ${i + 1}/${files.length} · ${it.name}`);

    const plan = planFor(it);
    if (plan.error) {
      it.status = 'failed';
      it.message = plan.error;
      S.results.push({ name: it.name, path: it.path, stem: it.stem, status: 'failed', error: plan.error, outDir });
      S.run.done = i + 1;
      setProgress(i + 1, files.length);
      log(`${it.name}: ${plan.error}`);
      renderResults(); renderInputs();
      continue;
    }

    const args = P.buildAlignArgs(
      { path: it.path, kind: plan.kind, tgPath: plan.tgPath, key: it.stem },
      Object.assign({}, s, { modelPath: S.model.path, outDir }),
    );
    if (i === 0) log('argv: ' + args.join(' '));

    // Remember what was already there so a leftover output from an earlier run
    // cannot be mistaken for this run's result.
    const tgPath = P.joinPath(outDir, it.stem + '.TextGrid');
    const jsonPath = P.joinPath(outDir, it.stem + '.diagnosis.json');
    const tgBefore = await window.bridge.exists(tgPath);
    const startedMs = Date.now();

    const res = await window.bridge.runAlign({
      cliPath: S.cli.path,
      args,
      item: { stem: it.stem, path: it.path },
      outDir,
      backend: s.backend,
    });

    // Collect what the CLI wrote (exit code 1 can also mean "skipped", so the
    // output files — not the exit code — decide the outcome).
    const row = {
      name: it.name, path: it.path, stem: it.stem, outDir,
      status: 'failed', error: '', tgPath, jsonPath,
      frames: null, phones: null,
      agreement: null, confidence: null, determinacy: null, monotonicity: null,
      durationMs: res && res.durationMs,
    };

    const exists = await window.bridge.exists(tgPath);
    const fresh = !!(exists && exists.exists &&
      (!tgBefore.exists || (exists.mtimeMs != null && exists.mtimeMs >= startedMs - 1000)));
    if (fresh) {
      const txt = await window.bridge.readText(tgPath);
      if (txt && txt.ok) {
        const tg = P.parseTextGrid(txt.text);
        const ph = P.tierByName(tg, S.settings.phonesTier || 'phones') || tg.tiers[tg.tiers.length - 1];
        if (ph && ph.intervals) row.phones = ph.intervals.filter(v => v.text).length;
        if (tg.xmax != null) row.frames = Math.round(tg.xmax / modelTimestep());
      }
    } else if (exists && exists.exists) {
      log(`${it.name}: output is stale (not rewritten by this run)`);
    }

    const jr = await window.bridge.readText(jsonPath);
    if (jr && jr.ok) {
      const diag = P.parseDiagnosis(jr.text);
      if (diag) {
        row.frames = diag.num_frames != null ? diag.num_frames : row.frames;
        row.phones = diag.num_tokens != null ? diag.num_tokens : row.phones;
        row.agreement = diag.agreement;
        row.confidence = diag.confidence;
        row.determinacy = diag.determinacy;
        row.monotonicity = diag.monotonicity;
      }
    }

    const err = (res && (res.stderr || res.error)) || '';
    if (res && res.cancelled) {
      it.status = 'cancelled'; it.message = 'cancelled';
      row.status = 'cancelled'; row.error = 'cancelled';
    } else if (fresh) {
      it.status = 'ok';
      it.message = row.agreement != null ? 'agreement ' + P.fmtMetric(row.agreement) : 'ok';
      row.status = res && res.ok ? 'ok'
        : (res && res.code != null ? 'ok (exit ' + res.code + ')' : 'ok');
      log(`${it.name}: ok · ${P.fmtInt(row.frames)} frames · ${P.fmtInt(row.phones)} phones · ` +
          `agreement ${P.fmtMetric(row.agreement)} · ${(row.durationMs / 1000).toFixed(1)}s`);
    } else {
      const textMode = /no phone source/i.test(err);
      const msg = textMode
        ? 'text mode not available in this CLI build (no --text yet)'
        : (stderrTail(err, 2) || 'failed');
      it.status = 'failed';
      it.message = msg;
      row.status = 'failed';
      row.error = err.slice(-1500) || msg;
      log(`${it.name}: FAILED — ${msg}`);
      if (textMode) log('  cli stderr: ' + stderrTail(err, 1));
    }

    S.results.push(row);
    S.run.done = i + 1;
    setProgress(i + 1, files.length);
    if (refresh || it.status !== 'ok') { renderResults(); renderInputs(); }
  }
  renderResults();
  renderInputs();

  S.run.running = false;
  S.run.cancel = false;
  await window.bridge.resetCancel();
  refreshRunReady();
  const okN = S.results.filter(r => r.status.startsWith('ok')).length;
  const badN = S.results.length - okN;
  setDot('runDot', badN ? 'bad' : 'ok');
  toast('run-status', `done · ${okN} ok · ${badN} failed/skipped`);
  log(`run done · ${okN} ok · ${badN} failed/skipped`);

  // auto-select the worst row so the preview shows something useful
  if (S.results.length) {
    const sorted = orderResults();
    const first = sorted.find(r => r.status.startsWith('ok')) || sorted[0];
    if (first) selectRow(first, false);
  }
}

// ---------------------------------------------------------------------------
// results table
// ---------------------------------------------------------------------------
function orderResults() {
  return P.sortRows(S.results, S.settings.sortMetric, S.settings.sortDir);
}

function renderResults() {
  const tb = $('result-table').querySelector('tbody');
  tb.innerHTML = '';
  const rows = orderResults();
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.title = r.path + (r.error ? '\n' + r.error : '');
    const cells = [
      [r.name, 'cell-name'],
      [P.fmtInt(r.frames), 'num'],
      [P.fmtInt(r.phones), 'num'],
      [P.fmtMetric(r.agreement), 'num'],
      [P.fmtMetric(r.confidence), 'num'],
      [P.fmtMetric(r.determinacy), 'num'],
      [P.fmtMetric(r.monotonicity), 'num'],
    ];
    for (const [text, cls] of cells) {
      const td = document.createElement('td');
      td.className = cls;
      td.textContent = text;
      tr.appendChild(td);
    }
    const td = document.createElement('td');
    td.appendChild(pill(r.status, r.status.startsWith('ok') ? 'ok' : r.status === 'failed' ? 'bad' : 'skip'));
    tr.appendChild(td);
    tr.onclick = () => selectRow(r, true);
    if (S.sel && S.sel === r) tr.classList.add('sel');
    tb.appendChild(tr);
  }
  $('btn-save-csv').disabled = !S.results.length;
  $('btn-open-out').disabled = !S.run.outDir;
}

// ---------------------------------------------------------------------------
// preview: waveform + 3-tier annotation strip + playback
// ---------------------------------------------------------------------------
function audioCtx() {
  if (!S.audio) S.audio = { ctx: null };
  if (!S.audio.ctx) S.audio.ctx = new (window.AudioContext || window.webkitAudioContext)();
  return S.audio.ctx;
}

function stopPlayback() {
  const a = S.audio;
  if (!a) return;
  if (a.src) { try { a.src.onended = null; a.src.stop(); } catch { /* already stopped */ } a.src = null; }
  a.playing = false;
}

function computePeaks(buffer, buckets) {
  const ch = buffer.getChannelData(0);
  const n = ch.length;
  const nb = Math.max(1, Math.min(buckets, n));
  const step = n / nb;
  const peaks = new Float32Array(nb * 2);
  for (let i = 0; i < nb; i++) {
    const a = Math.floor(i * step);
    const b = Math.min(n, Math.floor((i + 1) * step));
    let mn = Infinity, mx = -Infinity;
    for (let j = a; j < b; j++) { const v = ch[j]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (!isFinite(mn)) { mn = 0; mx = 0; }
    peaks[i * 2] = mn; peaks[i * 2 + 1] = mx;
  }
  return peaks;
}

async function selectRow(row, fromClick) {
  S.sel = row;
  renderResults();
  $('preview-status').textContent = row.name + ' — loading…';
  $('btn-play').disabled = true;
  stopPlayback();
  try {
    await ensureAudio(row.path);
  } catch (err) {
    $('preview-status').textContent = 'audio decode failed: ' + (err.message || err);
    log('decode error: ' + (err.message || err));
  }
  await loadAnnotation(row);
  if (fromClick && S.audio) S.audio.playOffset = 0;
  drawPreview();
  updatePlayBtn();
}

async function ensureAudio(path) {
  const a = S.audio || (S.audio = {});
  if (a.path === path && a.buffer) return a;
  stopPlayback();
  a.path = path; a.buffer = null; a.peaks = null; a.playing = false; a.playOffset = 0;
  const r = await window.bridge.readBytes(path);
  if (!r || !r.ok) throw new Error((r && r.error) || 'read failed');
  const u8 = r.bytes;
  const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
  const ctx = audioCtx();
  const buf = await ctx.decodeAudioData(ab);
  a.buffer = buf;
  a.duration = buf.duration;
  a.peaks = computePeaks(buf, 2400);
  a.playOffset = 0;
  $('btn-play').disabled = false;
  $('preview-status').textContent =
    `${P.baseName(path)} · ${buf.sampleRate} Hz · ${buf.numberOfChannels}ch · ${buf.duration.toFixed(2)} s`;
  return a;
}

async function loadAnnotation(row) {
  const a = S.audio;
  if (!a) return;
  a.grid = null;
  a.gridPath = null;
  const candidates = [row.tgPath, row.path.replace(/\.[^.]+$/, '.TextGrid')];
  for (const c of candidates) {
    if (!c) continue;
    const r = await window.bridge.readText(c);
    if (r && r.ok) {
      const grid = P.parseTextGrid(r.text);
      if (grid.tiers.length) { a.grid = grid; a.gridPath = c; break; }
    }
  }
}

// --- drawing ---------------------------------------------------------------
function prepCanvas(cv, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(320, Math.floor(cv.clientWidth || 900));
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(cssH * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h: cssH };
}

function playheadTime() {
  const a = S.audio;
  if (!a || !a.playing || !a.ctx) return a ? (a.playOffset || 0) : null;
  const t = a.ctx.currentTime - a.startedAt;
  return Math.max(0, Math.min(a.duration, t));
}

function drawWaveformCanvas() {
  const cv = $('waveform');
  if (!cv) return;
  const H = 110;
  const { ctx, w, h } = prepCanvas(cv, H);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fbfbfb'; ctx.fillRect(0, 0, w, h);
  const a = S.audio;
  const duration = (a && a.duration) || 0;
  const mid = h / 2;
  ctx.strokeStyle = '#e8e8e8';
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  if (!a || !a.peaks || !duration) {
    ctx.fillStyle = '#aaa'; ctx.font = '12px system-ui';
    ctx.fillText('no audio loaded', 10, mid + 4);
    return;
  }
  const nb = a.peaks.length / 2;
  ctx.fillStyle = '#3a3a3a';
  for (let x = 0; x < w; x++) {
    const i0 = Math.floor((x / w) * nb);
    const i1 = Math.max(i0 + 1, Math.floor(((x + 1) / w) * nb));
    let mn = Infinity, mx = -Infinity;
    for (let i = i0; i < i1 && i < nb; i++) {
      mn = Math.min(mn, a.peaks[i * 2]);
      mx = Math.max(mx, a.peaks[i * 2 + 1]);
    }
    if (!isFinite(mn)) continue;
    const y0 = mid - mx * mid * 0.95;
    const y1 = mid - mn * mid * 0.95;
    ctx.fillRect(x, y0, 1, Math.max(1, y1 - y0));
  }
  // time ruler
  ctx.fillStyle = '#999'; ctx.font = '10px ui-monospace, monospace';
  const step = niceStep(duration, w);
  for (let t = 0; t <= duration + 1e-6; t += step) {
    const x = (t / duration) * w;
    ctx.strokeStyle = '#ececec';
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillText(t.toFixed(t < 10 ? 1 : 0) + 's', Math.min(x + 3, w - 30), 11);
  }
  drawPlayhead(ctx, w, h, duration);
}

function niceStep(duration, w) {
  const target = duration / Math.max(2, Math.floor(w / 90));
  for (const s of [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120]) {
    if (s >= target) return s;
  }
  return Math.ceil(target / 60) * 60;
}

function drawPlayhead(ctx, w, h, duration) {
  const t = playheadTime();
  if (t == null || !duration) return;
  const x = (t / duration) * w;
  ctx.strokeStyle = '#c23';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  ctx.lineWidth = 1;
}

const TIER_COLORS = { texts: '#8a7fd4', words: '#4a90d9', phones: '#2b7a3c' };
const LANE_ORDER = ['texts', 'words', 'phones'];

function stripTiers() {
  const a = S.audio;
  if (!a || !a.grid) return [];
  const out = [];
  for (const name of LANE_ORDER) {
    const t = P.tierByName(a.grid, name);
    if (t) out.push(t);
  }
  // any extra tiers the CLI did not write under a known name
  for (const t of a.grid.tiers) {
    if (!LANE_ORDER.some(n => n.toLowerCase() === String(t.name).toLowerCase())) out.push(t);
  }
  return out.slice(0, 4);
}

function drawStripCanvas() {
  const cv = $('strip');
  if (!cv) return;
  const H = 132;
  const { ctx, w, h } = prepCanvas(cv, H);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fbfbfb'; ctx.fillRect(0, 0, w, h);
  const a = S.audio;
  const duration = (a && a.duration) || (a && a.grid && a.grid.xmax) || 0;
  if (!a || !a.grid || !duration) {
    ctx.fillStyle = '#aaa'; ctx.font = '12px system-ui';
    ctx.fillText(a && a.buffer ? 'no annotation for this file' : 'no audio loaded', 10, h / 2);
    S.strip = null;
    return;
  }
  const tiers = stripTiers();
  const RULER = 16;
  const laneH = Math.max(18, (h - RULER - 4) / Math.max(1, tiers.length));
  const xOf = t => (t / duration) * w;

  // ruler
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, RULER);
  ctx.fillStyle = '#888'; ctx.font = '10px ui-monospace, monospace';
  const step = niceStep(duration, w);
  ctx.strokeStyle = '#eee';
  for (let t = 0; t <= duration + 1e-6; t += step) {
    const x = xOf(t);
    ctx.beginPath(); ctx.moveTo(x, RULER); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillText(t.toFixed(t < 10 ? 1 : 0), Math.min(x + 3, w - 26), 11);
  }

  const layout = { w, duration, lanes: [], xOf };
  tiers.forEach((tier, li) => {
    const y = RULER + li * laneH;
    const color = TIER_COLORS[String(tier.name).toLowerCase()] || '#7a7a7a';
    // lane background + name tag
    ctx.fillStyle = li % 2 ? '#fafafa' : '#f4f4f4';
    ctx.fillRect(0, y, w, laneH - 2);
    ctx.fillStyle = '#999'; ctx.font = '9px system-ui';
    ctx.fillText(String(tier.name), 4, y + 9);

    const lane = { name: tier.name, y, h: laneH - 2, intervals: [] };
    for (let vi = 0; vi < tier.intervals.length; vi++) {
      const iv = tier.intervals[vi];
      if (iv.xmax == null || iv.xmin == null) continue;
      const x0 = xOf(iv.xmin), x1 = xOf(iv.xmax);
      const bw = Math.max(1, x1 - x0);
      // Bars are rebuilt on every redraw, so hover state is tracked by a
      // stable id rather than by object identity.
      const bar = { id: tier.name + '#' + vi, x0, x1, w: bw, xmin: iv.xmin, xmax: iv.xmax, text: iv.text, lane: lane };
      lane.intervals.push(bar);
      const hovered = S.hover && S.hover.id === bar.id;
      ctx.fillStyle = iv.text ? color : '#ddd';
      ctx.globalAlpha = hovered ? 1 : 0.82;
      ctx.fillRect(x0, y + 3, Math.max(1, bw - 1), laneH - 8);
      ctx.globalAlpha = 1;
      if (hovered) {
        ctx.strokeStyle = '#111'; ctx.lineWidth = 1.5;
        ctx.strokeRect(x0 - 0.5, y + 2.5, Math.max(1, bw), laneH - 9);
        ctx.lineWidth = 1;
      }
      const label = String(iv.text || '');
      if (label && bw > 16) {
        ctx.save();
        ctx.beginPath(); ctx.rect(x0, y + 3, Math.max(1, bw - 1), laneH - 8); ctx.clip();
        ctx.fillStyle = '#fff';
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillText(label, x0 + 3, y + laneH - 8);
        ctx.restore();
      }
    }
    layout.lanes.push(lane);
  });
  S.strip = layout;
  drawPlayhead(ctx, w, h, duration);
}

function drawPreview() {
  drawWaveformCanvas();
  drawStripCanvas();
}

function tickPlayhead() {
  const a = S.audio;
  if (a && a.playing) {
    drawPreview();
    requestAnimationFrame(tickPlayhead);
  }
}

function updatePlayBtn() {
  const a = S.audio;
  $('btn-play').textContent = a && a.playing ? 'Pause' : 'Play';
}

async function togglePlay() {
  const a = S.audio;
  if (!a || !a.buffer) return;
  if (a.playing) { a.playOffset = playheadTime(); stopPlayback(); updatePlayBtn(); drawPreview(); return; }
  const ctx = audioCtx();
  if (ctx.state === 'suspended') { try { await ctx.resume(); } catch { /* ignore */ } }
  const start = Math.max(0, Math.min(a.playOffset || 0, Math.max(0, a.duration - 0.02)));
  const src = ctx.createBufferSource();
  src.buffer = a.buffer;
  src.connect(ctx.destination);
  src.start(0, start);
  a.src = src;
  a.startedAt = ctx.currentTime - start;
  a.playing = true;
  src.onended = () => {
    if (a.src !== src) return;
    a.playing = false; a.src = null; a.playOffset = 0;
    updatePlayBtn(); drawPreview();
  };
  updatePlayBtn();
  requestAnimationFrame(tickPlayhead);
}

function timeFromEvent(e, cv, duration) {
  const rect = cv.getBoundingClientRect();
  const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
  return { t: (x / rect.width) * duration, x };
}

function seekFromEvent(e, cv) {
  const a = S.audio;
  if (!a || !a.buffer) return;
  const { t } = timeFromEvent(e, cv, a.duration);
  const wasPlaying = a.playing;
  if (wasPlaying) {
    stopPlayback();
    a.playOffset = t;
    togglePlay();
  } else {
    a.playOffset = t;
    drawPreview();
  }
}

function bindPreviewInteractions() {
  const wave = $('waveform');
  const strip = $('strip');
  const tip = $('tooltip');

  wave.addEventListener('mousedown', (e) => seekFromEvent(e, wave));

  strip.addEventListener('mousemove', (e) => {
    const a = S.audio;
    if (!a || !S.strip) { tip.classList.add('hidden'); return; }
    const { t, x } = timeFromEvent(e, strip, S.strip.duration);
    // hit-test top-most lane whose y-range contains the cursor
    const rect = strip.getBoundingClientRect();
    const my = e.clientY - rect.top;
    let found = null;
    for (const lane of S.strip.lanes) {
      if (my < lane.y || my > lane.y + lane.h) continue;
      found = lane.intervals.find(b => t >= b.xmin && t < b.xmax) || null;
      break;
    }
    const prevId = S.hover && S.hover.id;
    const nextId = found && found.id;
    if (nextId !== prevId) { S.hover = found ? { id: found.id } : null; drawStripCanvas(); }
    if (found) {
      const dur = found.xmax - found.xmin;
      tip.textContent =
        `${found.text || '(empty)'}\n` +
        `${found.lane.name}  ${P.fmtSeconds(found.xmin)} – ${P.fmtSeconds(found.xmax)}  (${dur.toFixed(3)} s)`;
      tip.style.left = x + 'px';
      tip.style.top = (found.lane.y + 2) + 'px';
      tip.classList.remove('hidden');
    } else {
      tip.classList.add('hidden');
    }
  });
  strip.addEventListener('mouseleave', () => {
    tip.classList.add('hidden');
    if (S.hover) { S.hover = null; drawStripCanvas(); }
  });
  strip.addEventListener('mousedown', (e) => seekFromEvent(e, strip));
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
function bindInputs() {
  const dz = $('dropZone');
  dz.onclick = () => $('file-audio').click();
  $('file-audio').onchange = (e) => addFileObjects(e.target.files);
  ['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault(); dz.classList.remove('drag');
    addFileObjects(e.dataTransfer.files);
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());

  $('btn-add-files').onclick = async () => {
    const r = await window.bridge.pickFiles();
    if (!r || !r.ok) { if (r && !r.canceled) log('pick files: ' + r.error); return; }
    mergeInputs(r.list);
    log(`added ${(r.list || []).length} file(s)`);
    renderInputs();
  };
  $('btn-add-folder').onclick = async () => {
    const r = await window.bridge.pickFolder();
    if (!r || !r.ok) { if (r && !r.canceled) log('pick folder: ' + r.error); return; }
    mergeInputs(r.list);
    log(`scanned folder: ${(r.list || []).length} audio file(s)`);
    renderInputs();
  };
  $('btn-clear-inputs').onclick = () => {
    S.inputs = [];
    S.results = [];
    S.sel = null;
    renderInputs();
    renderResults();
    $('preview-status').textContent = 'select a result row';
    stopPlayback();
    S.audio = null;
    drawPreview();
    log('input list cleared');
  };
}

function bindModel() {
  $('btn-model').onclick = async () => {
    const r = await window.bridge.pickModel();
    if (!r || !r.ok) { if (r && !r.canceled) log('model: ' + r.error); return; }
    await loadModel(r.modelPath);
  };
  $('btn-inspect').onclick = () => runInspect(false);
  $('btn-cli').onclick = async () => {
    const r = await window.bridge.pickCli();
    if (!r || !r.ok) { if (r && !r.canceled) log('cli: ' + (r.error || 'not selected')); return; }
    S.cli.path = r.cliPath;
    S.cli.version = r.version;
    $('cli-status').textContent = `${r.cliPath}${r.version ? '  ·  ' + r.version : ''}`;
    setDot('cliDot', r.versionOk === false ? '' : 'ok');
    log('engine set: ' + r.cliPath);
    refreshRunReady();
  };
}

function bindActions() {
  $('btn-pick-csv').onclick = () => pickCsv();
  $('btn-pick-tgdir').onclick = pickTextgridDir;
  $('btn-pick-outdir').onclick = async () => {
    const r = await window.bridge.pickOutDir(S.settings.outDir || undefined);
    if (!r || !r.ok) return;
    S.settings.outDir = r.outDir;
    $('in-outdir').value = r.outDir;
    persist();
  };
  $('btn-run').onclick = runAll;
  $('btn-cancel').onclick = async () => {
    S.run.cancel = true;
    toast('run-status', 'cancelling…');
    const r = await window.bridge.cancelRun();
    log('cancel requested' + (r && r.killed ? ' (killed running child)' : ''));
  };
  $('btn-open-out').onclick = async () => {
    const dir = S.run.outDir || (S.settings.outDir || '');
    if (!dir) return;
    const r = await window.bridge.openPath(dir);
    if (!r || !r.ok) log('open folder: ' + ((r && r.error) || 'failed'));
  };
  $('btn-save-csv').onclick = async () => {
    const rows = orderResults();
    const text = P.toCsv(rows);
    const r = await window.bridge.saveCsv('tifa-alignments.csv', text);
    if (!r || !r.ok) { if (r && !r.canceled) log('save csv: ' + r.error); return; }
    log('table saved: ' + r.filePath);
  };
  $('btn-play').onclick = togglePlay;

  $('sel-sort').onchange = (e) => { S.settings.sortMetric = e.target.value; persist(); renderResults(); };
  $('sel-sort-dir').onchange = (e) => { S.settings.sortDir = e.target.value; persist(); renderResults(); };

  window.bridge.onRunChunk(({ stream, text }) => {
    // Vulkan init noise repeats for every file — count it instead of flooding.
    const lines = String(text).split(/\r?\n/).filter(l => l.length);
    const keep = [];
    let muted = 0;
    for (const l of lines) {
      if (/^ggml_vulkan:/i.test(l.trim())) { muted++; continue; }
      keep.push(stream === 'stderr' ? l : '· ' + l);
    }
    if (keep.length) log(keep.join('\n'));
    if (muted) S.run.vulkanLines = (S.run.vulkanLines || 0) + muted;
  });
}

function restoreUiFromSettings() {
  const s = S.settings;
  $('sel-phones-mode').value = s.phonesMode;
  $('in-phones-tier').value = s.phonesTier;
  $('in-phones-tier-2').value = s.phonesTier;
  $('in-inline-phones').value = s.inlinePhones || '';
  $('in-language').value = s.language;
  $('sel-skip').value = s.skipHandling;
  $('in-skip-penalty').value = s.skipPenalty;
  $('in-outdir').value = s.outDir || '';
  $('sel-backend').value = s.backend;
  $('in-export-json').checked = s.exportJson !== false;
  $('sel-sort').value = s.sortMetric;
  $('sel-sort-dir').value = s.sortDir;
  refreshPhonesUI();
}

window.addEventListener('DOMContentLoaded', async () => {
  try {
  const resp = await window.bridge.getConfig();
  // getConfig() answers { ok, config, userDir, version } — unwrap it.
  const cfg = (resp && resp.config) || {};
  const appVersion = (resp && resp.version) || '';
  applyConfig(cfg);
  restoreUiFromSettings();

  bindSetting('sel-phones-mode', 'phonesMode');
  bindSetting('sel-backend', 'backend');
  bindSetting('sel-skip', 'skipHandling');
  bindSetting('in-language', 'language', 'input');
  bindSetting('in-skip-penalty', 'skipPenalty', 'input');
  bindSetting('in-outdir', 'outDir', 'input');
  bindSetting('in-export-json', 'exportJson');
  bindSetting('in-phones-tier', 'phonesTier', 'input');
  bindSetting('in-inline-phones', 'inlinePhones', 'input');
  $('in-phones-tier-2').addEventListener('input', (e) => {
    S.settings.phonesTier = e.target.value;
    $('in-phones-tier').value = e.target.value;
    persist();
  });
  bindSetting('sel-sort', 'sortMetric');
  bindSetting('sel-sort-dir', 'sortDir');

  bindInputs();
  bindModel();
  bindActions();
  bindPreviewInteractions();
  renderInputs();
  renderResults();
  drawPreview();
  window.addEventListener('resize', () => drawPreview());

  await detectCli(true);
  if (cfg && cfg.modelPath) await loadModel(cfg.modelPath, true);
  else setDot('modelDot', '');
  if (cfg && cfg.textgridDir) {
    const r = await window.bridge.exists(cfg.textgridDir);
    if (r && r.exists) $('tgdir-status').textContent = cfg.textgridDir;
  }
  if (cfg && cfg.csvPath) {
    const r = await window.bridge.exists(cfg.csvPath);
    if (r && r.exists) {
      S.settings.csvPath = cfg.csvPath;
      if (!(await loadCsvIndex(cfg.csvPath))) $('csv-status').textContent = P.baseName(cfg.csvPath) + ' (configured)';
    }
  }
  refreshRunReady();
  const readyMsg = 'ready · ui ' + appVersion + ' · ' + JSON.stringify({
    cli: S.cli.path ? P.baseName(S.cli.path) : null,
    model: S.model.name || null,
    runEnabled: !$('btn-run').disabled,
  });
  log(readyMsg);
  console.log('[tifa] ' + readyMsg);
  } catch (err) {
    // Startup must degrade, not die: report and leave the UI usable.
    const msg = (err && err.message) || String(err);
    setDot('cliDot', 'bad');
    toast('cli-status', 'startup error: ' + msg);
    try { log('startup error: ' + msg); } catch { /* log pane gone */ }
    console.error('[tifa] startup error:', err);
  }
});

window.addEventListener('unhandledrejection', (e) => {
  const msg = (e && e.reason && e.reason.message) || String((e && e.reason) || e);
  log('unhandled error: ' + msg);
  console.error('[tifa] unhandled rejection:', e && e.reason);
});
