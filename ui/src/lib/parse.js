'use strict';
// ---------------------------------------------------------------------------
// Pure helpers shared by the Electron main process, the renderer and the
// headless smoke test (scripts/smoke-test.js).
//
// No Node or DOM APIs are used here on purpose: the module is loaded either
// with require() (main process, node test) or with a plain <script> tag in the
// renderer, where it publishes itself as globalThis.TifaParse.
// ---------------------------------------------------------------------------
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TifaParse = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  // -------------------------------------------------------------------------
  // Path helpers (POSIX or Windows separators, no fs)
  // -------------------------------------------------------------------------
  function baseName(p) {
    const s = String(p == null ? '' : p).replace(/\\/g, '/');
    const i = s.lastIndexOf('/');
    return i >= 0 ? s.slice(i + 1) : s;
  }

  function dirName(p) {
    const s = String(p == null ? '' : p).replace(/\\/g, '/');
    const i = s.lastIndexOf('/');
    if (i < 0) return '';
    if (i === 0) return '/';
    return s.slice(0, i);
  }

  function stemOf(p) {
    const b = baseName(p);
    const i = b.lastIndexOf('.');
    return i > 0 ? b.slice(0, i) : b;
  }

  function extOf(p) {
    const b = baseName(p);
    const i = b.lastIndexOf('.');
    return i > 0 ? b.slice(i + 1).toLowerCase() : '';
  }

  function joinPath(dir, name) {
    const d = String(dir || '').replace(/\\/g, '/').replace(/\/+$/, '');
    return d ? d + '/' + name : name;
  }

  const AUDIO_EXTS = ['wav', 'flac', 'mp3'];

  function isAudio(p) {
    return AUDIO_EXTS.indexOf(extOf(p)) >= 0;
  }

  // -------------------------------------------------------------------------
  // RFC4180-ish CSV
  // -------------------------------------------------------------------------
  // Splits into rows of raw string cells; doubled "" inside a quoted field is
  // an escaped quote, quoted fields may contain commas and newlines.
  function splitCsvRows(text) {
    const rows = [];
    let row = [];
    let cell = '';
    let quoted = false;
    let i = 0;
    const s = String(text == null ? '' : text).replace(/^﻿/, '');
    const pushCell = () => { row.push(cell); cell = ''; };
    const pushRow = () => { pushCell(); rows.push(row); row = []; };
    while (i < s.length) {
      const c = s[i];
      if (quoted) {
        if (c === '"') {
          if (s[i + 1] === '"') { cell += '"'; i += 2; continue; }
          quoted = false; i++; continue;
        }
        cell += c; i++; continue;
      }
      if (c === '"') { quoted = true; i++; continue; }
      if (c === ',') { pushCell(); i++; continue; }
      if (c === '\r') { i++; continue; }
      if (c === '\n') { pushRow(); i++; continue; }
      cell += c; i++;
    }
    // trailing partial row (only if the file did not end with a newline)
    if (cell.length || row.length) pushRow();
    while (rows.length && rows[rows.length - 1].every(c => c === '')) rows.pop();
    return rows;
  }

  function parseCsv(text) {
    const rows = splitCsvRows(text);
    if (!rows.length) return { header: [], rows: [] };
    const header = rows[0].map(h => h.trim());
    const out = [];
    for (let r = 1; r < rows.length; r++) {
      const cells = rows[r];
      if (cells.length === 1 && cells[0] === '') continue;
      const obj = {};
      for (let c = 0; c < header.length; c++) obj[header[c]] = cells[c] !== undefined ? cells[c] : '';
      obj.__cells = cells;
      out.push(obj);
    }
    return { header, rows: out };
  }

  // Index a DiffSinger transcriptions.csv by its identifier column.
  // Returns { header, keys: { <id>: { phones: n, index: i } }, count }.
  function csvIndex(text, keyCol) {
    const col = keyCol || 'name';
    const { header, rows } = parseCsv(text);
    const keys = {};
    if (!header.length) return { header, keys, count: 0, keyCol: col };
    const colIdx = header.indexOf(col);
    const phIdx = header.indexOf('ph_seq');
    for (let i = 0; i < rows.length; i++) {
      const id = (colIdx >= 0 ? rows[i][col] : rows[i].__cells[0]) || '';
      const idTrim = String(id).trim();
      if (!idTrim) continue;
      const ph = phIdx >= 0 ? String(rows[i].ph_seq || '').trim() : '';
      keys[idTrim] = { phones: ph ? ph.split(/\s+/).length : 0, index: i };
    }
    return { header, keys, count: rows.length, keyCol: col };
  }

  // -------------------------------------------------------------------------
  // TextGrid (Praat) — long and short ("ooTextFile") variants
  // -------------------------------------------------------------------------
  function unquote(s) {
    let t = String(s == null ? '' : s).trim();
    if (t.length >= 2 && t[0] === '"' && t[t.length - 1] === '"') t = t.slice(1, -1);
    return t.replace(/""/g, '"');
  }

  // Splits an item body into interval objects. Handles named fields
  // (xmin = 0.1 / xmax = ... / text = "...") and the positional short form.
  function parseIntervalBody(body) {
    const out = [];
    if (!body) return out;
    const blocks = body.split(/^[ \t]*intervals[ \t]*\[\d+\][ \t]*:[ \t]*$/m);
    if (blocks.length > 1) {
      for (let i = 1; i < blocks.length; i++) {
        const b = blocks[i];
        const mn = b.match(/^[ \t]*xmin[ \t]*=[ \t]*(-?[\d.eE+-]+)/m);
        const mx = b.match(/^[ \t]*xmax[ \t]*=[ \t]*(-?[\d.eE+-]+)/m);
        const tx = b.match(/^[ \t]*text[ \t]*=[ \t]*(.*)$/m);
        out.push({
          xmin: mn ? Number(mn[1]) : null,
          xmax: mx ? Number(mx[1]) : null,
          text: tx ? unquote(tx[1]) : '',
        });
      }
      return out;
    }
    // short form: bare values in groups of three
    const vals = [];
    for (const raw of body.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line) continue;
      if (/^xmin[ \t]*=/.test(line) || /^xmax[ \t]*=/.test(line) || /^text[ \t]*=/.test(line) ||
          /^intervals[ \t]*:/.test(line)) continue;
      vals.push(line);
    }
    for (let i = 0; i + 2 < vals.length; i += 3) {
      const a = Number(vals[i]), b = Number(vals[i + 1]);
      if (!isFinite(a) || !isFinite(b)) break;
      out.push({ xmin: a, xmax: b, text: unquote(vals[i + 2]) });
    }
    return out;
  }

  function parseTextGrid(text) {
    const s = String(text == null ? '' : text);
    const xminM = s.match(/^[ \t]*xmin[ \t]*=[ \t]*(-?[\d.eE+-]+)/m);
    const xmaxM = s.match(/^[ \t]*xmax[ \t]*=[ \t]*(-?[\d.eE+-]+)/m);
    const grid = {
      xmin: xminM ? Number(xminM[1]) : 0,
      xmax: xmaxM ? Number(xmaxM[1]) : null,
      tiers: [],
    };
    // Every tier body starts with its class line — in both the long form
    // ("item [1]:" then class) and the short form ("item []:" then class).
    const parts = s.split(/^[ \t]*class[ \t]*=[ \t]*"(IntervalTier|TextTier|PointTier)"[ \t]*$/m);
    for (let i = 1; i < parts.length; i += 2) {
      const cls = parts[i];
      const body = parts[i + 1] || '';
      const nameM = body.match(/^[ \t]*name[ \t]*=[ \t]*(.*)$/m);
      const tmn = body.match(/^[ \t]*xmin[ \t]*=[ \t]*(-?[\d.eE+-]+)/m);
      const tmx = body.match(/^[ \t]*xmax[ \t]*=[ \t]*(-?[\d.eE+-]+)/m);
      const ivIdx = body.search(/^[ \t]*intervals[ \t]*[:[]/m);
      const ivBody = ivIdx >= 0 ? body.slice(ivIdx) : '';
      const tier = {
        class: cls,
        name: nameM ? unquote(nameM[1]) : '',
        xmin: tmn ? Number(tmn[1]) : grid.xmin,
        xmax: tmx ? Number(tmx[1]) : grid.xmax,
        intervals: parseIntervalBody(ivBody),
      };
      if (tier.class === 'PointTier') {
        tier.intervals = tier.intervals.map(p => ({ xmin: p.xmin, xmax: p.xmin, text: p.text, point: true }));
      }
      grid.tiers.push(tier);
    }
    if (grid.xmax == null && grid.tiers.length) grid.xmax = grid.tiers[grid.tiers.length - 1].xmax;
    return grid;
  }

  function tierByName(grid, name) {
    if (!grid || !grid.tiers) return null;
    const want = String(name || '').toLowerCase();
    return grid.tiers.find(t => String(t.name).toLowerCase() === want) || null;
  }

  // -------------------------------------------------------------------------
  // CLI output: <name>.diagnosis.json
  // -------------------------------------------------------------------------
  const DIAG_KEYS = ['agreement', 'confidence', 'determinacy', 'monotonicity'];

  function numOrNull(v) {
    const n = typeof v === 'number' ? v : Number(v);
    return isFinite(n) ? n : null;
  }

  function parseDiagnosis(input) {
    let o = input;
    if (typeof input === 'string') {
      try { o = JSON.parse(input); } catch { return null; }
    }
    if (!o || typeof o !== 'object') return null;
    const out = {
      identifier: typeof o.identifier === 'string' ? o.identifier : null,
      num_frames: numOrNull(o.num_frames),
      num_tokens: numOrNull(o.num_tokens),
      agreement: numOrNull(o.agreement),
      confidence: numOrNull(o.confidence),
      determinacy: numOrNull(o.determinacy),
      monotonicity: numOrNull(o.monotonicity),
      raw: o,
    };
    return out;
  }

  // -------------------------------------------------------------------------
  // CLI `inspect <model.gguf>` output -> [{key, value}]
  // -------------------------------------------------------------------------
  function parseInspect(stdout) {
    const fields = [];
    for (const raw of String(stdout == null ? '' : stdout).split(/\r?\n/)) {
      const line = raw.replace(/\s+$/, '');
      if (!line.trim()) continue;
      if (/^ggml_/i.test(line.trim())) continue;        // backend chatter
      const m = line.match(/^([A-Za-z][A-Za-z0-9 _/()+.-]*?)\s*:\s*(.*)$/);
      if (!m) continue;
      const key = m[1].trim();
      if (/^(error|usage)$/i.test(key)) continue;
      fields.push({ key, value: m[2].trim() });
    }
    return { fields, raw: String(stdout == null ? '' : stdout) };
  }

  // -------------------------------------------------------------------------
  // Phone-source resolution for the "Auto (sidecar)" mode
  // -------------------------------------------------------------------------
  // `names` is the list of file names in the audio's directory. Pure so the
  // decision table can be unit-tested.
  function resolveSidecar(names, stem, opts) {
    const o = opts || {};
    const tier = o.phonesTier || 'phones';
    const lookup = new Map();
    for (const n of names || []) lookup.set(String(n).toLowerCase(), n);
    const tg = lookup.get(String(stem + '.TextGrid').toLowerCase());
    if (tg) return { kind: 'textgrid', file: tg, tier };
    for (const ext of ['.txt', '.lab']) {
      const f = lookup.get((stem + ext).toLowerCase());
      if (f) return { kind: 'text', file: f, tier };
    }
    return { kind: 'none', file: null, tier };
  }

  // -------------------------------------------------------------------------
  // CLI argument construction for one file
  // -------------------------------------------------------------------------
  // item  : { path, kind: 'textgrid'|'text'|'textgrid-dir'|'csv'|'inline'|'none',
  //           tgPath?, key? }
  // s     : settings object from the renderer
  function buildAlignArgs(item, s) {
    const a = ['align', item.path, '-m', s.modelPath];
    const outDir = item.outDir || s.outDir;
    if (outDir) a.push('-o', outDir);
    if (s.language) a.push('-l', s.language);

    switch (item.kind) {
      case 'textgrid':
        a.push('--textgrid', item.tgPath);
        a.push('--phones-tier', s.phonesTier || 'phones');
        break;
      case 'textgrid-dir':
        a.push('--textgrid', s.textgridDir);
        a.push('--phones-tier', s.phonesTier || 'phones');
        break;
      case 'csv':
        a.push('--transcriptions-csv', s.csvPath);
        if (s.csvKeyColumn && s.csvKeyColumn !== 'name') a.push('--key', item.key);
        break;
      case 'inline':
        a.push('--phones', s.inlinePhones);
        break;
      case 'text':
        // Text mode: the CLI derives the phones itself (there are no phone
        // flags to pass).  Until that lands the CLI errors out and we surface
        // its stderr verbatim.
        break;
      default:
        break;
    }

    if (s.skipHandling) a.push('--skip-handling', s.skipHandling);
    if (s.skipPenalty !== '' && s.skipPenalty != null && isFinite(Number(s.skipPenalty))) {
      a.push('--skip-penalty', String(s.skipPenalty));
    }
    a.push('--output-formats', s.exportJson ? 'textgrid,json' : 'textgrid');
    if (s.backend && s.backend !== 'auto') a.push('--backend', s.backend);
    if (s.maxFrames) a.push('--max-frames', String(s.maxFrames));
    return a;
  }

  // -------------------------------------------------------------------------
  // Result table: formatting, sorting, CSV export
  // -------------------------------------------------------------------------
  const RESULT_COLUMNS = [
    { key: 'name', label: 'file', metric: false },
    { key: 'frames', label: 'frames', metric: false },
    { key: 'phones', label: 'phones', metric: false },
    { key: 'agreement', label: 'agreement', metric: true },
    { key: 'confidence', label: 'confidence', metric: true },
    { key: 'determinacy', label: 'determinacy', metric: true },
    { key: 'monotonicity', label: 'monotonicity', metric: true },
  ];

  const METRICS = DIAG_KEYS;

  function fmtMetric(v, digits) {
    if (v == null || !isFinite(v)) return '—';
    return Number(v).toFixed(digits == null ? 3 : digits);
  }

  function fmtInt(v) {
    if (v == null || !isFinite(v)) return '—';
    return String(Math.round(v));
  }

  // Worst-first = ascending for every diagnostic metric (higher is better).
  // Rows without the metric always sink to the bottom.
  function sortRows(rows, metric, dir) {
    const list = (rows || []).slice();
    if (!metric || metric === 'name') {
      list.sort((a, b) => String(a.name).localeCompare(String(b.name), undefined, { numeric: true }));
      if (dir === 'desc') list.reverse();
      return list;
    }
    const sign = dir === 'desc' ? -1 : 1;
    list.sort((a, b) => {
      const av = a[metric], bv = b[metric];
      const an = (av == null || !isFinite(av)) ? null : Number(av);
      const bn = (bv == null || !isFinite(bv)) ? null : Number(bv);
      if (an == null && bn == null) return String(a.name).localeCompare(String(b.name));
      if (an == null) return 1;
      if (bn == null) return -1;
      if (an !== bn) return (an - bn) * sign;
      return String(a.name).localeCompare(String(b.name), undefined, { numeric: true });
    });
    return list;
  }

  function csvCell(v) {
    const s = v == null ? '' : String(v);
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function toCsv(rows) {
    const head = ['name', 'path', 'frames', 'phones', 'agreement', 'confidence',
                  'determinacy', 'monotonicity', 'textgrid', 'diagnosis_json',
                  'status', 'error'];
    const lines = [head.join(',')];
    for (const r of rows || []) {
      lines.push([
        csvCell(r.name), csvCell(r.path),
        r.frames == null ? '' : Math.round(r.frames),
        r.phones == null ? '' : Math.round(r.phones),
        r.agreement == null ? '' : r.agreement,
        r.confidence == null ? '' : r.confidence,
        r.determinacy == null ? '' : r.determinacy,
        r.monotonicity == null ? '' : r.monotonicity,
        csvCell(r.tgPath || ''), csvCell(r.jsonPath || ''),
        csvCell(r.status || ''), csvCell(r.error || ''),
      ].join(','));
    }
    return lines.join('\n') + '\n';
  }

  // -------------------------------------------------------------------------
  // Misc
  // -------------------------------------------------------------------------
  function fmtSeconds(t) {
    if (t == null || !isFinite(t)) return '—';
    return Number(t).toFixed(3) + ' s';
  }

  function fmtTime(t) {
    if (t == null || !isFinite(t)) return '—';
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return m + ':' + s.toFixed(2).padStart(5, '0');
  }

  return {
    AUDIO_EXTS,
    RESULT_COLUMNS,
    METRICS,
    DIAG_KEYS,
    baseName, dirName, stemOf, extOf, joinPath, isAudio,
    parseCsv, csvIndex, splitCsvRows,
    parseTextGrid, parseIntervalBody, tierByName, unquote,
    parseDiagnosis, parseInspect, resolveSidecar, buildAlignArgs,
    sortRows, toCsv, fmtMetric, fmtInt, fmtSeconds, fmtTime,
  };
});
