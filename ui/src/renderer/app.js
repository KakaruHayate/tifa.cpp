'use strict';
// ---------------------------------------------------------------------------
// TIFA Label — 数据集标注工具渲染层。
//
// 流程（与 docs/dataset-workflow.md 一致）：
//   ① 第一遍对齐（TIFA）
//   ② 呼吸检测 AP/SP 并合并（BreathLab，可选）
//   ③ 第二遍对齐（2PASS，可选）
// 每个文件顺序跑完启用的阶段，结果写入输出目录。
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
// 不能把局部变量命名为 bridge：contextBridge 暴露的是不可配置全局属性，
// 顶层同名 const 会直接抛 SyntaxError。
const api = window.bridge;
const P = window.TifaParse;

// ---- 全局状态 --------------------------------------------------------------

const S = {
  cliPath: null,
  modelPath: null,
  breathModel: null,
  inputs: [],            // { path, dir, name, stem, ext, sidecarKind, sidecarPath, status, ms }
  running: false,
  canceled: false,
  outDir: '',
  _unsubChunk: null,
};

// ---- 小工具 ----------------------------------------------------------------

function log(msg) {
  const el = $('log');
  const t = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  el.textContent += `[${t}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

let toastTimer = null;
function setDot(elId, state) {
  const el = $(elId);
  el.className = 'dot' + (state ? ' ' + state : '');
}

function setStatus(textId, dotId, text, state) {
  $(textId).textContent = text;
  setDot(dotId, state);
}

function fmtDur(sec) {
  if (sec == null || !isFinite(sec)) return '—';
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return m > 0 ? `${m}:${s.toFixed(1).padStart(4, '0')}` : `${s.toFixed(1)} s`;
}

// ---- 配置持久化 ------------------------------------------------------------

function settings() {
  return {
    modelPath: S.modelPath,
    breathModel: S.breathModel,
    outDir: $('in-outdir').value.trim(),
    language: $('in-language').value.trim(),
    oovHandling: $('sel-oov').value,
    uniformText: $('in-text').value.trim(),
    phonesTier: $('in-phones-tier').value.trim() || 'phones',
    csvPath: $('csv-status').dataset.path || '',
    textgridDir: $('tgdir-status').dataset.path || '',
    inlinePhones: $('in-inline-phones').value.trim(),
    skipHandling: $('sel-skip').value,
    skipPenalty: $('in-skip-penalty').value,
    exportJson: $('in-export-json').checked,
    backend: $('sel-backend').value,
    quiet: $('in-quiet').checked,
    minInsertMs: Number($('in-min-insert').value) || 50,
    sourceMode: document.querySelector('input[name="src-mode"]:checked').value,
    phonesMode: $('sel-phones-mode').value,
    useBreath: $('chk-breath').checked,
    usePass2: $('chk-pass2').checked,
  };
}

function persist() {
  api.setConfig({
    outDir: $('in-outdir').value.trim(),
    language: $('in-language').value.trim(),
    oov: $('sel-oov').value,
    uniformText: $('in-text').value,
    phonesTier: $('in-phones-tier').value,
    inlinePhones: $('in-inline-phones').value,
    skip: $('sel-skip').value,
    skipPenalty: $('in-skip-penalty').value,
    exportJson: $('in-export-json').checked,
    backend: $('sel-backend').value,
    quiet: $('in-quiet').checked,
    minInsertMs: $('in-min-insert').value,
    sourceMode: document.querySelector('input[name="src-mode"]:checked').value,
    phonesMode: $('sel-phones-mode').value,
    useBreath: $('chk-breath').checked,
    usePass2: $('chk-pass2').checked,
  });
}

function applyConfig(cfg) {
  const set = (id, v) => { if (v != null && v !== '') $(id).value = v; };
  set('in-outdir', cfg.outDir);
  set('in-language', cfg.language);
  set('sel-oov', cfg.oov);
  set('in-text', cfg.uniformText);
  set('in-phones-tier', cfg.phonesTier);
  set('in-inline-phones', cfg.inlinePhones);
  set('sel-skip', cfg.skip);
  set('in-skip-penalty', cfg.skipPenalty);
  set('sel-backend', cfg.backend);
  set('in-min-insert', cfg.minInsertMs);
  if (typeof cfg.exportJson === 'boolean') $('in-export-json').checked = cfg.exportJson;
  if (typeof cfg.quiet === 'boolean') $('in-quiet').checked = cfg.quiet;
  if (typeof cfg.useBreath === 'boolean') $('chk-breath').checked = cfg.useBreath;
  if (typeof cfg.usePass2 === 'boolean') $('chk-pass2').checked = cfg.usePass2;
  if (cfg.sourceMode) {
    const r = document.querySelector(`input[name="src-mode"][value="${cfg.sourceMode}"]`);
    if (r) r.checked = true;
  }
  if (cfg.phonesMode) $('sel-phones-mode').value = cfg.phonesMode;
}

// ---- 环境检测 --------------------------------------------------------------

async function detectCli(quiet) {
  const r = await api.detectCli();
  if (r.ok) {
    S.cliPath = r.cliPath;
    setStatus('cli-status', 'cliDot', `引擎就绪 ${r.version || ''}`.trim(), 'ok');
    $('btn-inspect').disabled = !S.modelPath;
  } else {
    S.cliPath = null;
    setStatus('cli-status', 'cliDot', '未找到 tifa_ggml_cli，请手动定位', 'bad');
    if (!quiet) log('未找到 tifa_ggml_cli：' + (r.error || ''));
  }
  refreshRunReady();
}

async function loadModel(modelPath, quiet) {
  if (!modelPath) return false;
  S.modelPath = modelPath;
  const r = await api.modelInfo(modelPath);
  if (r.ok) {
    const f = r.fields || [];
    const get = (k) => { const x = f.find(v => String(v.key).toLowerCase() === k); return x ? x.value : null; };
    const arch = get('general.architecture') || get('architecture') || '?';
    const num = (f.find(v => /params|numel/i.test(v.key)) || {}).value;
    const bits = [arch];
    if (num) bits.push(`${(Number(num) / 1e6).toFixed(1)}M 参数`);
    setStatus('model-status', 'modelDot', '对齐模型 ' + bits.join(' · ') + '：' + P.baseName(modelPath), 'ok');
    $('btn-inspect').disabled = !S.cliPath;
    refreshRunReady();
    return true;
  }
  setStatus('model-status', 'modelDot', '模型不可用：' + (r.error || ''), 'bad');
  if (!quiet) log('模型信息读取失败：' + (r.error || ''));
  refreshRunReady();
  return false;
}

async function loadBreathModel(modelPath, quiet) {
  if (!modelPath) return false;
  S.breathModel = modelPath;
  const r = await api.modelInfo(modelPath);
  if (r.ok) {
    const f = r.fields || [];
    const arch = (f.find(v => /general\.architecture/.test(v.key)) || {}).value || '?';
    setStatus('breath-status', 'breathDot', `呼吸模型 ${arch}：` + P.baseName(modelPath), 'ok');
    refreshRunReady();
    return true;
  }
  setStatus('breath-status', 'breathDot', '呼吸模型不可用：' + (r.error || ''), 'bad');
  if (!quiet) log('呼吸模型信息读取失败：' + (r.error || ''));
  refreshRunReady();
  return false;
}

// ---- 附加内容（可选，按需下载） --------------------------------------------
// 目前只有日语汉字词典（MeCab/UniDic）：假名歌词开箱即用，汉字需要它。

async function refreshExtras() {
  const r = await api.extrasStatus();
  if (!r.ok) {
    setStatus('unidic-status', 'unidicDot', '日语词典：状态未知', 'bad');
    return;
  }
  if (r.installing) return; // 进度回调在更新这一行
  if (r.unidicInstalled) {
    setStatus('unidic-status', 'unidicDot', '日语汉字词典已安装', 'ok');
    $('btn-unidic').classList.add('hidden');
  } else {
    setStatus('unidic-status', 'unidicDot',
      r.modelDir ? '日语汉字词典未安装（假名可用，汉字需要）' : '日语汉字词典未安装：请先选择对齐模型',
      '');
    $('btn-unidic').classList.toggle('hidden', !r.modelDir);
  }
}

async function installUnidic() {
  const btn = $('btn-unidic');
  btn.disabled = true;
  setStatus('unidic-status', 'unidicDot', '日语词典：下载中 0%', 'running');
  const r = await api.installUnidic();
  btn.disabled = false;
  if (r.ok) {
    setStatus('unidic-status', 'unidicDot', '日语汉字词典已安装', 'ok');
    btn.classList.add('hidden');
    log('日语汉字词典安装完成：' + (r.dir || ''));
  } else {
    setStatus('unidic-status', 'unidicDot', '日语词典安装失败', 'bad');
    log('日语词典安装失败：' + (r.error || ''));
    await refreshExtras();
  }
}

async function runInspect() {
  if (!S.cliPath || !S.modelPath) return;
  const r = await api.inspectModel(S.modelPath);
  const box = $('inspect-box');
  box.classList.remove('hidden');
  box.open = true;
  const dl = $('inspect-fields');
  dl.innerHTML = '';
  if (r.ok && r.fields) {
    for (const f of r.fields) {
      const dt = document.createElement('dt');
      dt.textContent = f.key;
      const dd = document.createElement('dd');
      dd.textContent = Array.isArray(f.value) ? JSON.stringify(f.value) : String(f.value);
      dl.append(dt, dd);
    }
    $('inspect-raw').textContent = r.raw || '';
  } else {
    $('inspect-raw').textContent = '读取失败：' + (r.error || '');
  }
}

// ---- 音频列表 --------------------------------------------------------------

function mergeInputs(recs) {
  const have = new Set(S.inputs.map(i => i.path.toLowerCase()));
  for (const rec of recs || []) {
    if (have.has(rec.path.toLowerCase())) continue;
    have.add(rec.path.toLowerCase());
    S.inputs.push({ ...rec, status: '待处理', ms: null, frames: null, phones: null, note: '' });
  }
  renderInputs();
}

function sourceLabel(item) {
  const s = settings();
  if (s.sourceMode === 'text') {
    if (s.uniformText) return '统一文本';
    if (item.sidecarKind === 'text') return P.baseName(item.sidecarPath);
    return '缺文本';
  }
  switch (s.phonesMode) {
    case 'auto':
      return item.sidecarKind === 'textgrid' ? P.baseName(item.sidecarPath) : '缺 TextGrid';
    case 'csv': return s.csvPath ? P.baseName(s.csvPath) : '未选择 CSV';
    case 'textgrid-dir': return s.textgridDir ? P.baseName(s.textgridDir) + '/' : '未选择目录';
    case 'inline': return s.inlinePhones ? '内联音素' : '未填写音素';
    default: return '—';
  }
}

function renderInputs() {
  const tbody = $('input-list').querySelector('tbody');
  tbody.innerHTML = '';
  for (const item of S.inputs) {
    const tr = document.createElement('tr');
    const tdName = document.createElement('td');
    tdName.textContent = item.name;
    tdName.title = item.path;
    const tdDur = document.createElement('td');
    tdDur.textContent = item.seconds != null ? fmtDur(item.seconds) : (item.sizeBytes ? (item.sizeBytes / 1048576).toFixed(1) + ' MB' : '—');
    const tdSrc = document.createElement('td');
    tdSrc.className = 'muted small';
    tdSrc.textContent = sourceLabel(item);
    const tdStatus = document.createElement('td');
    const bits = [item.status];
    if (item.ms != null) bits[0] += `（${(item.ms / 1000).toFixed(1)}s）`;
    if (item.agreement != null) bits.push('agreement ' + Number(item.agreement).toFixed(3));
    if (item.phones != null) bits.push(item.phones + ' 音素');
    if (item.note) bits.push(item.note);
    tdStatus.textContent = bits.join(' · ');
    if (item.status === '完成') tdStatus.className = 'ok-text';
    else if (item.status === '失败') tdStatus.className = 'bad-text';
    tr.append(tdName, tdDur, tdSrc, tdStatus);
    tbody.appendChild(tr);
  }
  $('input-count').textContent = `${S.inputs.length} 个文件`;
  refreshRunReady();
}

async function addPaths(paths) {
  if (!paths || !paths.length) return;
  const r = await api.scanPaths(paths);
  if (!r.ok) { log('扫描失败：' + (r.error || '')); return; }
  mergeInputs(r.list || []);
  if (r.truncated) {
    const more = $('input-more');
    more.classList.remove('hidden');
    more.textContent = '列表过长，仅扫描了前若干文件。';
  }
}

async function addFileObjects(files) {
  const paths = [];
  for (const f of files || []) {
    const p = api.pathForFile(f);
    if (p) paths.push(p);
  }
  await addPaths(paths);
}

// ---- 来源模式 UI -----------------------------------------------------------

function refreshSourceUI() {
  const s = settings();
  $('pane-text').classList.toggle('hidden', s.sourceMode !== 'text');
  $('pane-phones').classList.toggle('hidden', s.sourceMode !== 'phones');
  $('pane-csv').classList.toggle('hidden', s.phonesMode !== 'csv');
  $('pane-textgrid-dir').classList.toggle('hidden', s.phonesMode !== 'textgrid-dir');
  $('pane-inline').classList.toggle('hidden', s.phonesMode !== 'inline');
  $('pane-breath').classList.toggle('hidden', !s.useBreath);
  renderInputs();
}

// ---- 运行 ------------------------------------------------------------------

function refreshRunReady() {
  const s = settings();
  let why = '';
  if (S.running) why = '运行中…';
  else if (!S.cliPath) why = '未找到引擎';
  else if (!S.modelPath) why = '未选择对齐模型';
  else if (!S.inputs.length) why = '未导入音频';
  else if (s.sourceMode === 'phones') {
    if (s.phonesMode === 'csv' && !s.csvPath) why = '未选择 transcriptions.csv';
    if (s.phonesMode === 'textgrid-dir' && !s.textgridDir) why = '未选择 TextGrid 目录';
    if (s.phonesMode === 'inline' && !s.inlinePhones) why = '未填写音素序列';
  }
  if (!why && s.useBreath && !S.breathModel) why = '呼吸流程需要呼吸模型';
  if (!why && s.usePass2 && !s.useBreath) why = '2PASS 需要先启用呼吸检测';
  $('btn-run').disabled = !!why;
  $('run-status').textContent = why;
}

function setProgress(done, total) {
  const pct = total > 0 ? (done / total) * 100 : 0;
  $('progress-fill').style.width = pct + '%';
}

// The CLI reports per-file failures on stderr; surface the last line.
function lastStderr(res) {
  const lines = String((res && res.stderr) || '').split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  return lines.length ? lines[lines.length - 1] : ((res && res.error) || '');
}

function perFileKind(item, s) {
  if (s.sourceMode === 'text') return { kind: 'text' };
  switch (s.phonesMode) {
    case 'auto':
      if (item.sidecarKind === 'textgrid') return { kind: 'textgrid', tgPath: item.sidecarPath };
      return null;
    case 'csv': return { kind: 'csv', key: item.stem };
    case 'textgrid-dir': return { kind: 'textgrid-dir' };
    case 'inline': return { kind: 'inline' };
    default: return null;
  }
}

function runCli(cliPath, args, item, outDir, backend) {
  return new Promise((resolve) => {
    if (S._unsubChunk) S._unsubChunk();
    S._unsubChunk = api.onRunChunk(({ text }) => { log(text.replace(/\n+$/, '')); });
    const t0 = performance.now();
    api.runAlign({ cliPath, args, item: { stem: item.stem }, outDir, backend })
      .then((res) => {
        if (S._unsubChunk) { S._unsubChunk(); S._unsubChunk = null; }
        resolve({ ...res, wallMs: performance.now() - t0 });
      })
      .catch((err) => resolve({ ok: false, error: String(err) }));
  });
}

async function runAll() {
  const s = settings();
  S.running = true;
  S.canceled = false;
  refreshRunReady();
  $('btn-cancel').disabled = false;
  $('btn-run').disabled = true;
  $('log').textContent = '';
  $('log-box').open = true;

  const outBase = s.outDir || '';
  await api.resetCancel();
  log(`开始：${S.inputs.length} 个文件，流程 = 对齐${s.useBreath ? ' → 呼吸检测+合并' : ''}${s.usePass2 ? ' → 2PASS' : ''}`);
  if (outBase) { try { await api.ensureDir(outBase); } catch { /* CLI creates it */ } }

  let done = 0;
  for (const item of S.inputs) {
    if (S.canceled) { item.status = '已取消'; renderInputs(); break; }
    const outDir = outBase || P.joinPath(item.dir, 'out');
    item.outDir = outDir;   // 当前运行的目标目录优先（上次运行的缓存不能残留）
    const kind = perFileKind(item, s);
    if (!kind) {
      item.status = '跳过';
      item.note = s.sourceMode === 'text' ? '没有 <名称>.txt/.lab' : '没有同名 TextGrid';
      renderInputs();
      log(`跳过 ${item.name}：${item.note}`);
      done++; setProgress(done, S.inputs.length);
      continue;
    }

    const alignSettings = { ...s, outDir };
    try {
      // ---- ① 第一遍对齐 ----
      item.status = '对齐中';
      renderInputs();
      const args1 = P.buildAlignArgs({ ...item, ...kind }, alignSettings);
      if (s.quiet) args1.push('-q');
      log(`① ${item.name}：${args1.join(' ')}`);
      const res = await runCli(S.cliPath, args1, item, outDir, s.backend);
      if (!res.ok) throw new Error(lastStderr(res) || '对齐失败');
      const sum1 = P.parseAlignSummary(res.stdout);
      if (sum1) { item.frames = sum1.frames; item.phones = sum1.phones; item.agreement = sum1.agreement; }

      // ---- ② 呼吸检测并合并 ----
      if (s.useBreath) {
        item.status = '呼吸检测中';
        renderInputs();
        const args2 = ['breathe', item.path, '-m', S.breathModel, '--merge', outDir,
                       '-o', outDir, '--min-insert-ms', String(s.minInsertMs)];
        // Always state the format: the CLI now defaults to JSON on, so omitting
        // the flag would silently ignore an unchecked box.
        args2.push('--output-formats', s.exportJson ? 'textgrid,json' : 'textgrid');
        if (s.backend && s.backend !== 'auto') args2.push('--backend', s.backend);
        if (s.quiet) args2.push('-q');
        log(`② ${item.name}：${args2.join(' ')}`);
        const res2 = await runCli(S.cliPath, args2, item, outDir, s.backend);
        if (!res2.ok) throw new Error(lastStderr(res2) || '呼吸检测失败');
      }

      // ---- ③ 第二遍对齐 ----
      if (s.usePass2) {
        item.status = '2PASS 中';
        renderInputs();
        const args3 = ['align', item.path, '-m', S.modelPath, '--textgrid', outDir,
                       '-o', outDir, '--phones-tier', s.phonesTier];
        if (s.language) args3.push('-l', s.language);
        if (s.skipHandling) args3.push('--skip-handling', s.skipHandling);
        if (s.exportJson !== undefined) args3.push('--output-formats', s.exportJson ? 'textgrid,json' : 'textgrid');
        if (s.backend && s.backend !== 'auto') args3.push('--backend', s.backend);
        if (s.quiet) args3.push('-q');
        log(`③ ${item.name}：${args3.join(' ')}`);
        const res3 = await runCli(S.cliPath, args3, item, outDir, s.backend);
        if (!res3.ok) throw new Error(lastStderr(res3) || '2PASS 对齐失败');
        const sum3 = P.parseAlignSummary(res3.stdout);
        if (sum3) { item.frames = sum3.frames; item.phones = sum3.phones; item.agreement = sum3.agreement; }
      }

      item.ms = res.wallMs;
      item.status = '完成';
      S.outDir = outDir;
      $('btn-open-out').disabled = false;
    } catch (err) {
      item.status = '失败';
      item.note = String(err.message || err);
      log(`✗ ${item.name}：${item.note}`);
    }
    done++;
    setProgress(done, S.inputs.length);
    renderInputs();
  }

  S.running = false;
  $('btn-cancel').disabled = true;
  const okCount = S.inputs.filter(i => i.status === '完成').length;
  const failCount = S.inputs.filter(i => i.status === '失败').length;
  const skipCount = S.inputs.filter(i => i.status === '跳过').length;
  $('run-status').textContent = `完成 ${okCount} · 失败 ${failCount} · 跳过 ${skipCount}`;
  log(`全部结束：完成 ${okCount}，失败 ${failCount}，跳过 ${skipCount}`);
  refreshRunReady();
}

// ---- 事件绑定 --------------------------------------------------------------

function bind() {
  $('btn-cli').onclick = async () => {
    const r = await api.pickCli();
    if (r.ok) { await detectCli(); }
  };
  $('btn-inspect').onclick = runInspect;
  $('btn-model').onclick = async () => {
    const r = await api.pickModel();
    if (r.ok) await loadModel(r.modelPath);
  };
  $('btn-breath-model').onclick = async () => {
    const r = await api.pickBreathModel();
    if (r.ok) await loadBreathModel(r.breathModel);
  };
  $('btn-unidic').onclick = installUnidic;
  api.onExtrasProgress(({ received, total }) => {
    const pct = total ? Math.round((received / total) * 100) : null;
    const mb = Math.round(received / 1048576);
    setStatus('unidic-status', 'unidicDot',
      '日语词典：下载中 ' + (pct != null ? pct + '%' : mb + ' MB'), 'running');
  });

  // 输入
  $('btn-add-files').onclick = async () => {
    const r = await api.pickFiles();
    if (r.ok) {
      if (r.list) {
        mergeInputs(r.list);
        if (r.truncated) {
          $('input-more').classList.remove('hidden');
          $('input-more').textContent = '列表过长，仅扫描了前若干文件。';
        }
      } else if (r.paths) {
        await addPaths(r.paths);
      }
    }
  };
  $('btn-add-folder').onclick = async () => {
    const r = await api.pickFolder();
    if (r.ok) {
      if (r.list) {
        mergeInputs(r.list);
        if (r.truncated) {
          $('input-more').classList.remove('hidden');
          $('input-more').textContent = '列表过长，仅扫描了前若干文件。';
        }
      } else if (r.paths) {
        await addPaths(r.paths);
      }
    }
  };
  $('btn-clear-inputs').onclick = () => { S.inputs = []; renderInputs(); };

  const dz = $('dropZone');
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('over'); };
  dz.ondragleave = () => dz.classList.remove('over');
  dz.ondrop = async (e) => {
    e.preventDefault();
    dz.classList.remove('over');
    await addFileObjects(e.dataTransfer.files);
  };
  $('file-audio').onchange = async (e) => { await addFileObjects(e.target.files); e.target.value = ''; };

  // 来源模式
  for (const r of document.querySelectorAll('input[name="src-mode"]')) {
    r.onchange = () => { refreshSourceUI(); persist(); };
  }
  $('sel-phones-mode').onchange = () => { refreshSourceUI(); persist(); };
  $('btn-pick-csv').onclick = async () => {
    const r = await api.pickCsv();
    if (r.ok) {
      const el = $('csv-status');
      el.dataset.path = r.csvPath;
      el.textContent = `${P.baseName(r.csvPath)}（${r.count} 行）`;
      persist(); renderInputs();
    }
  };
  $('btn-pick-tgdir').onclick = async () => {
    const r = await api.pickTextgridDir();
    if (r.ok) {
      const el = $('tgdir-status');
      el.dataset.path = r.dir;
      el.textContent = r.dir;
      persist(); renderInputs();
    }
  };

  // 流程
  $('chk-breath').onchange = () => { refreshSourceUI(); persist(); };
  $('chk-pass2').onchange = () => { persist(); refreshRunReady(); };
  $('btn-pick-outdir').onclick = async () => {
    const r = await api.pickOutDir($('in-outdir').value.trim());
    if (r.ok) { $('in-outdir').value = r.dir; persist(); }
  };

  // 运行
  $('btn-run').onclick = runAll;
  $('btn-cancel').onclick = async () => {
    S.canceled = true;
    await api.cancelRun();
    log('已请求取消…');
  };
  $('btn-open-out').onclick = () => { if (S.outDir) api.openPath(S.outDir); };

  // 持久化
  for (const id of ['in-outdir', 'in-language', 'sel-oov', 'in-text', 'in-phones-tier',
                    'in-inline-phones', 'sel-skip', 'in-skip-penalty', 'in-export-json',
                    'sel-backend', 'in-quiet', 'in-min-insert']) {
    $(id).addEventListener('change', () => { persist(); renderInputs(); });
  }
}

// ---- 自动化钩子 ------------------------------------------------------------
// 把状态与入口显式挂到 window，供 e2e/自动化驱动使用：Electron 的
// executeJavaScript 看不到脚本级的 const 绑定，只有全局对象的属性。
window.TifaLabel = {
  S, settings, persist,
  detectCli, loadModel, loadBreathModel,
  addPaths, renderInputs, refreshSourceUI, refreshRunReady,
  runAll, lastStderr,
};

// ---- 启动 ------------------------------------------------------------------

(async function main() {
  bind();
  const cfg = await api.getConfig();
  applyConfig(cfg.config || {});
  refreshSourceUI();
  await detectCli(true);
  const conf = cfg.config || {};
  const haveModel = conf.modelPath ? await loadModel(conf.modelPath, true) : false;
  const haveBreath = conf.breathModel ? await loadBreathModel(conf.breathModel, true) : false;

  // 发布包布局：模型就在程序旁边的 models/ 里；没配置过（或配置已失效）时
  // 直接按默认相对路径导入，用户不需要手动选择。
  if (!haveModel || !haveBreath) {
    const auto = await api.modelAutodetect();
    if (auto.ok) {
      if (!haveModel && auto.aligner) {
        await loadModel(auto.aligner, true);
        await api.setModel(auto.aligner);
        log('已按默认路径导入对齐模型：' + auto.aligner);
      }
      if (!haveBreath && auto.breath) {
        await loadBreathModel(auto.breath, true);
        await api.setBreathModel(auto.breath);
        log('已按默认路径导入呼吸模型：' + auto.breath);
      }
    } else if (!haveModel) {
      log('未在默认路径找到模型（' + auto.error + '），请手动选择。');
    }
  }
  await refreshExtras();
  if (cfg.config && cfg.config.csvPath) {
    const el = $('csv-status');
    el.dataset.path = cfg.config.csvPath;
    el.textContent = P.baseName(cfg.config.csvPath);
  }
  if (cfg.config && cfg.config.textgridDir) {
    const el = $('tgdir-status');
    el.dataset.path = cfg.config.textgridDir;
    el.textContent = cfg.config.textgridDir;
  }
  refreshRunReady();
})();
