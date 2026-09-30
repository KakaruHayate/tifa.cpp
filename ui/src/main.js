'use strict';
// ---------------------------------------------------------------------------
// TIFA Aligner Studio — Electron main process.
//
// The renderer owns no Node APIs: everything (dialogs, fs, spawning the ggml
// CLI) goes through the IPC surface registered below and exposed by
// src/preload.js.
// ---------------------------------------------------------------------------
const {
  app, BrowserWindow, ipcMain, dialog, shell, Menu
} = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawn, spawnSync } = require('child_process');
const AdmZip = require('adm-zip');          // reserved for a future .zip model bundle
const P = require('./lib/parse.js');

void AdmZip;

// ---------------------------------------------------------------------------
// Paths / persisted config
// ---------------------------------------------------------------------------
const USER_DIR = app.getPath('userData');
const CONFIG_PATH = path.join(USER_DIR, 'config.json');
const APP_ROOT = path.join(__dirname, '..');           // <repo>/tifa.cpp/ui

function readConfig() {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const o = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
      if (o && typeof o === 'object') return o;
    }
  } catch { /* corrupt config -> start fresh */ }
  return {};
}

function writeConfig(patch) {
  const next = Object.assign(readConfig(), patch || {});
  try {
    fs.mkdirSync(USER_DIR, { recursive: true });
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(next, null, 2), 'utf8');
  } catch (err) {
    return { ok: false, error: err.message, config: next };
  }
  return { ok: true, config: next };
}

// ---------------------------------------------------------------------------
// CLI discovery: the ggml executable that ships next to this repo
// ---------------------------------------------------------------------------
const CLI_NAMES = process.platform === 'win32'
  ? ['tifa_ggml_cli.exe', 'tifa_ggml_cli']
  : ['tifa_ggml_cli'];

function cliCandidates() {
  const cfg = readConfig();
  const out = [];
  if (cfg.cliPath) out.push(cfg.cliPath);
  if (process.env.TIFA_GGML_CLI) out.push(process.env.TIFA_GGML_CLI);
  const dirs = [
    path.join(APP_ROOT, '..', 'build', 'bin'),            // tifa.cpp/build/bin
    path.join(APP_ROOT, '..', '..', 'build', 'bin'),
    path.join(APP_ROOT, 'bin'),
    path.join(APP_ROOT, '..', 'bin'),
    process.cwd(),
    path.dirname(app.getPath('exe')),
  ];
  for (const d of dirs) for (const n of CLI_NAMES) out.push(path.resolve(path.join(d, n)));
  return out;
}

function findCli() {
  for (const c of cliCandidates()) {
    try { if (c && fs.existsSync(c) && fs.statSync(c).isFile()) return c; } catch { /* skip */ }
  }
  return null;
}

function cliVersion(cliPath) {
  try {
    const r = spawnSync(cliPath, ['--version'], { encoding: 'utf8', timeout: 20000, windowsHide: true });
    const txt = ((r.stdout || '') + (r.stderr || '')).trim();
    if (r.error) return { ok: false, error: r.error.message };
    return { ok: r.status === 0 || txt.length > 0, text: txt.split(/\r?\n/).filter(Boolean).pop() || txt };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

function runCliCapture(cliPath, args, timeoutMs) {
  return new Promise((resolve) => {
    let out = ''; let err = '';
    let child;
    try {
      child = spawn(cliPath, args, { windowsHide: true, env: process.env });
    } catch (e) {
      resolve({ ok: false, code: -1, stdout: '', stderr: String(e.message) });
      return;
    }
    const timer = setTimeout(() => { try { child.kill(); } catch { /* ignore */ } }, timeoutMs || 120000);
    child.stdout.on('data', d => { out += d.toString(); });
    child.stderr.on('data', d => { err += d.toString(); });
    child.on('error', (e) => { clearTimeout(timer); resolve({ ok: false, code: -1, stdout: out, stderr: String(e.message) }); });
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ ok: code === 0, code, stdout: out, stderr: err });
    });
  });
}

// ---------------------------------------------------------------------------
// Input scanning (recursive for folders)
// ---------------------------------------------------------------------------
const MAX_SCAN = 20000;

function scanAudio(dirOrFile, acc) {
  const out = acc || [];
  let stat;
  try { stat = fs.statSync(dirOrFile); } catch { return out; }
  if (stat.isFile()) {
    if (P.isAudio(dirOrFile)) out.push(dirOrFile);
    return out;
  }
  let entries;
  try { entries = fs.readdirSync(dirOrFile, { withFileTypes: true }); } catch { return out; }
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const e of entries) {
    if (out.length >= MAX_SCAN) return out;
    const p = path.join(dirOrFile, e.name);
    if (e.isDirectory()) { if (!e.name.startsWith('.')) scanAudio(p, out); }
    else if (e.isFile() && P.isAudio(e.name)) out.push(p);
  }
  return out;
}

// Build the per-file input records. Directories are listed once each so the
// "auto sidecar" decision can be made with the shared pure helper.
function buildInputs(paths) {
  const audios = [];
  for (const p of paths || []) scanAudio(p, audios);
  const seen = new Set();
  const list = [];
  const dirCache = new Map();
  for (const a of audios) {
    const key = a.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const dir = path.dirname(a);
    if (!dirCache.has(dir)) {
      let names = [];
      try { names = fs.readdirSync(dir); } catch { names = []; }
      dirCache.set(dir, names);
    }
    const stem = P.stemOf(a);
    const side = P.resolveSidecar(dirCache.get(dir), stem, { phonesTier: 'phones' });
    list.push({
      path: a,
      dir,
      name: P.baseName(a),
      stem,
      ext: P.extOf(a),
      sidecarKind: side.kind,                       // textgrid | text | none
      sidecarPath: side.file ? path.join(dir, side.file) : null,
      sizeBytes: (() => { try { return fs.statSync(a).size; } catch { return null; } })(),
    });
  }
  return { list, truncated: audios.length >= MAX_SCAN };
}

// ---------------------------------------------------------------------------
// Child-process job control (one CLI process per file)
// ---------------------------------------------------------------------------
let currentChild = null;
let cancelRequested = false;

function killChild(child) {
  if (!child || child.killed) return;
  try {
    if (process.platform === 'win32' && child.pid) {
      spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true });
    } else {
      child.kill('SIGKILL');
    }
  } catch {
    try { child.kill(); } catch { /* ignore */ }
  }
}

// ---------------------------------------------------------------------------
// File-access guard: the renderer may only read audio/TextGrid/JSON under the
// roots the user actually picked (input folders, output folder, model, csv).
// ---------------------------------------------------------------------------
const allowedRoots = new Set();

function allowRoot(p) {
  if (!p) return;
  try { allowedRoots.add(path.resolve(p)); } catch { /* ignore */ }
}

function isAllowed(p) {
  let r;
  try { r = path.resolve(p); } catch { return false; }
  for (const root of allowedRoots) {
    if (r === root) return true;
    if (r.startsWith(root + path.sep)) return true;
  }
  // model files are explicitly chosen one by one
  const cfg = readConfig();
  if (cfg.modelPath && path.resolve(cfg.modelPath) === r) return true;
  if (cfg.csvPath && path.resolve(cfg.csvPath) === r) return true;
  if (cfg.textgridDir && r.startsWith(path.resolve(cfg.textgridDir))) return true;
  return false;
}

function readTextGuarded(p) {
  if (!p || !isAllowed(p)) return null;
  try { return fs.readFileSync(p, 'utf8'); } catch { return null; }
}

function readBytesGuarded(p) {
  if (!p || !isAllowed(p)) return null;
  try { return fs.readFileSync(p); } catch { return null; }
}

// ---------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------
function registerIpc() {
  const handle = (channel, fn) => {
    ipcMain.handle(channel, async (event, ...args) => {
      try { return await fn(event, ...args); }
      catch (err) { return { ok: false, error: (err && err.message) || String(err) }; }
    });
  };

  // --- config -------------------------------------------------------------
  handle('cfg:get', () => ({ ok: true, config: readConfig(), userDir: USER_DIR, version: app.getVersion() }));
  handle('cfg:set', (e, patch) => writeConfig(patch));

  // --- CLI ----------------------------------------------------------------
  handle('cli:detect', () => {
    const cli = findCli();
    if (!cli) return { ok: false, error: 'tifa_ggml_cli not found', candidates: cliCandidates().slice(0, 12) };
    const v = cliVersion(cli);
    return { ok: true, cliPath: cli, version: v.text || null, versionOk: !!v.ok };
  });
  handle('cli:pick', async () => {
    const filt = process.platform === 'win32'
      ? [{ name: 'tifa_ggml_cli', extensions: ['exe'] }]
      : [{ name: 'executable', extensions: ['*'] }];
    const r = await dialog.showOpenDialog({ title: 'Select tifa_ggml_cli', properties: ['openFile'], filters: filt });
    if (r.canceled || !r.filePaths.length) return { ok: false, canceled: true };
    const cliPath = r.filePaths[0];
    if (process.platform !== 'win32') { try { fs.chmodSync(cliPath, 0o755); } catch { /* ignore */ } }
    writeConfig({ cliPath });
    const v = cliVersion(cliPath);
    return { ok: true, cliPath, version: v.text || null, versionOk: !!v.ok, error: v.error };
  });

  // --- model --------------------------------------------------------------
  handle('model:pick', async () => {
    const r = await dialog.showOpenDialog({
      title: 'Select TIFA .gguf model',
      properties: ['openFile'],
      filters: [{ name: 'GGUF model', extensions: ['gguf'] }, { name: 'All files', extensions: ['*'] }],
      defaultPath: readConfig().modelPath ? path.dirname(readConfig().modelPath) : undefined,
    });
    if (r.canceled || !r.filePaths.length) return { ok: false, canceled: true };
    const modelPath = r.filePaths[0];
    writeConfig({ modelPath });
    return { ok: true, modelPath };
  });
  handle('model:set', (e, modelPath) => {
    if (!modelPath || !fs.existsSync(modelPath)) return { ok: false, error: 'file not found: ' + modelPath };
    writeConfig({ modelPath });
    return { ok: true, modelPath };
  });
  handle('model:info', (e, modelPath) => {
    if (!modelPath) return { ok: false, error: 'no model selected' };
    let size = null;
    try { size = fs.statSync(modelPath).size; } catch { /* ignore */ }
    return { ok: true, modelPath, name: P.baseName(modelPath), size };
  });
  handle('model:inspect', async (e, modelPath) => {
    const cli = findCli();
    if (!cli) return { ok: false, error: 'tifa_ggml_cli not found' };
    if (!modelPath || !fs.existsSync(modelPath)) return { ok: false, error: 'model not found: ' + modelPath };
    const r = await runCliCapture(cli, ['inspect', modelPath], 180000);
    const parsed = P.parseInspect(r.stdout || r.stderr);
    if (!r.ok && !parsed.fields.length) {
      return { ok: false, error: (r.stderr || r.stdout || 'inspect failed').trim().slice(-1500) };
    }
    return { ok: true, ...parsed, stderr: r.stderr };
  });

  // --- inputs -------------------------------------------------------------
  handle('inputs:pick-files', async () => {
    const r = await dialog.showOpenDialog({
      title: 'Add audio files',
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'Audio', extensions: P.AUDIO_EXTS }, { name: 'All files', extensions: ['*'] }],
    });
    if (r.canceled || !r.filePaths.length) return { ok: false, canceled: true };
    for (const f of r.filePaths) allowRoot(path.dirname(f));
    return { ok: true, ...buildInputs(r.filePaths) };
  });
  handle('inputs:pick-folder', async () => {
    const r = await dialog.showOpenDialog({ title: 'Add a folder (recursive)', properties: ['openDirectory'] });
    if (r.canceled || !r.filePaths.length) return { ok: false, canceled: true };
    for (const d of r.filePaths) allowRoot(d);
    return { ok: true, ...buildInputs(r.filePaths) };
  });
  handle('inputs:scan', (e, paths) => {
    const list = Array.isArray(paths) ? paths : [paths];
    for (const p of list) {
      try { allowRoot(fs.statSync(p).isDirectory() ? p : path.dirname(p)); } catch { /* ignore */ }
    }
    return { ok: true, ...buildInputs(list) };
  });

  // --- phone sources ------------------------------------------------------
  handle('phones:pick-csv', async () => {
    const r = await dialog.showOpenDialog({
      title: 'Select DiffSinger transcriptions.csv',
      properties: ['openFile'],
      filters: [{ name: 'CSV', extensions: ['csv'] }, { name: 'All files', extensions: ['*'] }],
    });
    if (r.canceled || !r.filePaths.length) return { ok: false, canceled: true };
    const csvPath = r.filePaths[0];
    allowRoot(path.dirname(csvPath));
    writeConfig({ csvPath });
    let text = '';
    try { text = fs.readFileSync(csvPath, 'utf8'); }
    catch (err) { return { ok: false, error: err.message }; }
    const idx = P.csvIndex(text, 'name');
    return { ok: true, csvPath, header: idx.header, keys: idx.keys, count: idx.count };
  });
  handle('phones:pick-textgrid-dir', async () => {
    const r = await dialog.showOpenDialog({ title: 'Select a TextGrid folder', properties: ['openDirectory'] });
    if (r.canceled || !r.filePaths.length) return { ok: false, canceled: true };
    const dir = r.filePaths[0];
    allowRoot(dir);
    writeConfig({ textgridDir: dir });
    let count = 0;
    try { count = fs.readdirSync(dir).filter(n => /\.textgrid$/i.test(n)).length; } catch { /* ignore */ }
    return { ok: true, textgridDir: dir, count };
  });

  // --- output -------------------------------------------------------------
  handle('output:pick-dir', async (e, current) => {
    const r = await dialog.showOpenDialog({
      title: 'Select output directory',
      properties: ['openDirectory', 'createDirectory'],
      defaultPath: current || undefined,
    });
    if (r.canceled || !r.filePaths.length) return { ok: false, canceled: true };
    allowRoot(r.filePaths[0]);
    return { ok: true, outDir: r.filePaths[0] };
  });
  handle('output:ensure-dir', (e, dir) => {
    if (!dir) return { ok: false, error: 'no output directory' };
    try {
      // The directory is derived from an input the user picked (or chosen in a
      // dialog), so creating it is intentional and bounded.
      if (!isAllowed(dir)) allowRoot(path.dirname(path.dirname(dir)));
      fs.mkdirSync(dir, { recursive: true });
      allowRoot(dir);
      return { ok: true, outDir: dir };
    } catch (err) { return { ok: false, error: err.message }; }
  });

  handle('shell:open', (e, target) => {
    if (!target) return { ok: false, error: 'no path' };
    return shell.openPath(target).then(err => (err ? { ok: false, error: err } : { ok: true, target }));
  });

  // --- guarded reads for the preview pane ---------------------------------
  handle('fs:read-text', (e, p) => {
    const t = readTextGuarded(p);
    return t == null ? { ok: false, error: 'not readable (or outside allowed roots)' } : { ok: true, text: t };
  });
  handle('fs:read-bytes', (e, p) => {
    const b = readBytesGuarded(p);
    return b == null ? { ok: false, error: 'not readable (or outside allowed roots)' } : { ok: true, bytes: b };
  });
  handle('fs:exists', (e, p) => {
    try {
      const st = fs.statSync(p);
      return { ok: true, exists: true, isDir: st.isDirectory(), mtimeMs: st.mtimeMs, size: st.size };
    } catch { return { ok: true, exists: false, isDir: false, mtimeMs: null, size: null }; }
  });

  handle('csv:save', async (e, defaultName, text) => {
    const r = await dialog.showSaveDialog({
      title: 'Save table as CSV',
      defaultPath: defaultName || 'tifa-alignments.csv',
      filters: [{ name: 'CSV', extensions: ['csv'] }],
    });
    if (r.canceled || !r.filePath) return { ok: false, canceled: true };
    try { fs.writeFileSync(r.filePath, String(text == null ? '' : text), 'utf8'); }
    catch (err) { return { ok: false, error: err.message }; }
    return { ok: true, filePath: r.filePath };
  });

  // --- run ----------------------------------------------------------------
  handle('run:cancel', () => {
    cancelRequested = true;
    if (currentChild) { killChild(currentChild); return { ok: true, killed: true }; }
    return { ok: true, killed: false };
  });
  handle('run:reset-cancel', () => { cancelRequested = false; return { ok: true }; });

  handle('run:align', (event, req) => {
    const { cliPath, args, item } = req || {};
    if (!cliPath) return { ok: false, error: 'no CLI path' };
    if (!fs.existsSync(cliPath)) return { ok: false, error: 'CLI not found: ' + cliPath };
    if (currentChild) return { ok: false, error: 'another alignment is already running' };

    const outDir = req.outDir;
    if (outDir) { try { allowRoot(path.dirname(path.dirname(outDir))); fs.mkdirSync(outDir, { recursive: true }); allowRoot(outDir); } catch { /* ignore */ } }

    const env = Object.assign({}, process.env);
    if (req.backend && req.backend !== 'auto') env.TIFA_GGML_BACKEND = req.backend;
    if (req.threads) env.TIFA_GGML_THREADS = String(req.threads);

    return new Promise((resolve) => {
      const started = Date.now();
      let child;
      try {
        child = spawn(cliPath, args, { windowsHide: true, env, cwd: path.dirname(cliPath) });
      } catch (err) {
        resolve({ ok: false, error: err.message, identifier: item && item.stem });
        return;
      }
      currentChild = child;
      const send = (stream, chunk) => {
        if (!event.sender.isDestroyed()) {
          event.sender.send('run:chunk', { id: item && item.stem, stream, text: chunk });
        }
      };
      let stdout = ''; let stderr = '';
      child.stdout.on('data', d => { const s = d.toString(); stdout += s; send('stdout', s); });
      child.stderr.on('data', d => { const s = d.toString(); stderr += s; send('stderr', s); });
      child.on('error', (err) => {
        if (currentChild === child) currentChild = null;
        resolve({ ok: false, error: err.message, stdout, stderr, identifier: item && item.stem, durationMs: Date.now() - started });
      });
      child.on('close', (code) => {
        if (currentChild === child) currentChild = null;
        resolve({
          ok: code === 0,
          code,
          stdout,
          stderr,
          identifier: item && item.stem,
          cancelled: cancelRequested,
          durationMs: Date.now() - started,
        });
      });
    });
  });

  handle('sys:info', () => ({
    ok: true, platform: process.platform, arch: process.arch,
    electron: process.versions.electron, node: process.versions.node,
    cpus: os.cpus().length, userData: USER_DIR,
  }));
}

// ---------------------------------------------------------------------------
// Window / app lifecycle
// ---------------------------------------------------------------------------
function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 880,
    minWidth: 980,
    minHeight: 620,
    backgroundColor: '#fdfdfd',
    title: 'TIFA Aligner Studio',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: 'deny' }; });
  win.webContents.on('did-fail-load', (e, code, desc, url) => {
    console.error(`[tifa] renderer failed to load (${code} ${desc}) ${url}`);
  });
  win.webContents.on('render-process-gone', (e, details) => {
    console.error('[tifa] renderer process gone:', details && details.reason);
  });
  // Dev aid: TIFA_UI_LOG=1 mirrors the renderer console to the terminal so the
  // app can be smoke-checked without a visible window.
  if (process.env.TIFA_UI_LOG === '1') {
    win.webContents.on('console-message', (e, level, message) => {
      console.log('[renderer] ' + message);
    });
  }
  return win;
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  registerIpc();
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
}).catch((err) => {
  // Startup must never take the app down silently (e.g. no model configured).
  console.error('[tifa] startup error:', err && err.message ? err.message : err);
});

app.on('window-all-closed', () => {
  if (currentChild) killChild(currentChild);
  if (process.platform !== 'darwin') app.quit();
});
app.on('before-quit', () => { if (currentChild) killChild(currentChild); });
