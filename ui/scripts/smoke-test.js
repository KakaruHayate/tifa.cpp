'use strict';
// ---------------------------------------------------------------------------
// Headless smoke test for the pure logic in src/lib/parse.js.
//
//   node scripts/smoke-test.js            # hermetic: inline fixtures only
//   node scripts/smoke-test.js --cli      # + a live end-to-end run, which
//                                         #   needs TIFA_GGML_CLI, TIFA_GGML_MODEL
//                                         #   and TIFA_SMOKE_WAV
//
// Exits non-zero when a check fails, so it is usable as a CI gate.
// ---------------------------------------------------------------------------
const fs = require('fs');
const os = require('os');
const path = require('path');
const P = require('../src/lib/parse.js');

let pass = 0, fail = 0;
function check(name, cond, detail) {
  if (cond) { pass++; console.log('  ok   ' + name); }
  else { fail++; console.log('  FAIL ' + name + (detail === undefined ? '' : '  -> ' + JSON.stringify(detail))); }
}
function eq(name, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  check(name, ok, ok ? undefined : { got, want });
}
function approx(name, got, want, tol) {
  check(name + ' ~ ' + want, got != null && Math.abs(got - want) <= (tol == null ? 1e-6 : tol), got);
}

// ---------------------------------------------------------------------------
// Fixtures (inline — no external files, so the test is hermetic)
// ---------------------------------------------------------------------------
const TG = [
  'File type = "ooTextFile"',
  'Object class = "TextGrid"',
  '',
  'xmin = 0.0',
  'xmax = 1.5',
  'tiers? <exists>',
  'size = 1',
  'item []:',
  '\titem [1]:',
  '\t\tclass = "IntervalTier"',
  '\t\tname = "phones"',
  '\t\txmin = 0.0',
  '\t\txmax = 1.5',
  '\t\tintervals: size = 3',
  '\t\t\tintervals [1]:',
  '\t\t\t\txmin = 0',
  '\t\t\t\txmax = 0.5',
  '\t\t\t\ttext = "AP"',
  '\t\t\tintervals [2]:',
  '\t\t\t\txmin = 0.5',
  '\t\t\t\txmax = 1.0',
  '\t\t\t\ttext = "a"',
  '\t\t\tintervals [3]:',
  '\t\t\t\txmin = 1.0',
  '\t\t\t\txmax = 1.5',
  '\t\t\t\ttext = "b"',
  ''
].join('\n');

const CSV = [
  'name,ph_seq,ph_dur,ph_num',
  'sample_0_0,AP a b,0.5 0.25 0.25,2 1',
  'sample_1_0,AP b,0.7 0.3,1 1',
  ''
].join('\n');

const DIAG = JSON.stringify({
  identifier: 'sample_0_0', num_frames: 150, num_tokens: 3,
  agreement: 0.999, confidence: 0.81, determinacy: 0.9, monotonicity: 0.95
});

const INSPECT = [
  'ggml_vulkan: found 1 device',
  'architecture : tifa-fa',
  'name/version : TIFA-1.0-ST / 1',
  'vocab size   : 256 (symbols: 220)',
  'features     : 48000 Hz, hop 480, fft 2048, win 2048, 80 mels, 0-8000 Hz',
  'timestep     : 0.0100 s',
  'backbone     : dim 256, 8 layers, 8 heads x 64, joint/glu, qk_norm 1, rope 1',
  ''
].join('\n');

// --- 1. TextGrid ------------------------------------------------------------
console.log('[1] parseTextGrid');
const gt = P.parseTextGrid(TG);
eq('xmax', gt.xmax, 1.5);
eq('tier count', gt.tiers.length, 1);
eq('tier name', gt.tiers[0].name, 'phones');
eq('interval count', gt.tiers[0].intervals.length, 3);
eq('interval[0]', [gt.tiers[0].intervals[0].xmin, gt.tiers[0].intervals[0].xmax,
                   gt.tiers[0].intervals[0].text], [0, 0.5, 'AP']);
eq('tierByName', P.tierByName(gt, 'PHONES') === gt.tiers[0], true);
check('numeric bounds everywhere',
  gt.tiers[0].intervals.every(iv => typeof iv.xmin === 'number' && typeof iv.xmax === 'number'));

// --- 2. DiffSinger CSV ------------------------------------------------------
console.log('[2] csvIndex / parseCsv');
const idx = P.csvIndex(CSV, 'name');
eq('header', idx.header.slice(0, 4), ['name', 'ph_seq', 'ph_dur', 'ph_num']);
eq('row count', idx.count, 2);
eq('phones of sample_0_0', idx.keys['sample_0_0'].phones, 3);
const { rows } = P.parseCsv(CSV);
eq('ph_seq split', rows[0].ph_seq.split(/\s+/), ['AP', 'a', 'b']);
approx('ph_dur[0]', parseFloat(rows[0].ph_dur.split(/\s+/)[0]), 0.5, 1e-9);

// --- 3. argv construction ---------------------------------------------------
console.log('[3] buildAlignArgs');
const settings = {
  modelPath: 'model.gguf', outDir: 'out', language: 'zh', phonesTier: 'phones',
  csvPath: 'transcriptions.csv', textgridDir: 'tgs'
};
const aCsv = P.buildAlignArgs({ path: 'a.wav', kind: 'csv' }, settings);
eq('csv mode', aCsv.slice(0, 6), ['align', 'a.wav', '-m', 'model.gguf', '-o', 'out']);
check('csv flag', aCsv.includes('--transcriptions-csv') && aCsv.includes('transcriptions.csv'), aCsv);
check('language flag', aCsv.join(' ').includes('-l zh'), aCsv);
const aTg = P.buildAlignArgs({ path: 'a.wav', kind: 'textgrid', tgPath: 'a.TextGrid' }, settings);
check('textgrid flag', aTg.includes('--textgrid') && aTg.includes('a.TextGrid'), aTg);
check('phones tier flag', aTg.includes('--phones-tier') && aTg.includes('phones'), aTg);
const aText = P.buildAlignArgs({ path: 'a.wav', kind: 'text' }, settings);
check('text mode passes no phone source',
  !aText.includes('--transcriptions-csv') && !aText.includes('--textgrid'), aText);

// --- 4. diagnosis JSON ------------------------------------------------------
console.log('[4] parseDiagnosis');
const diag = P.parseDiagnosis(DIAG);
check('parses', !!diag, diag);
eq('agreement', diag.agreement, 0.999);
eq('fields', [diag.confidence, diag.determinacy, diag.monotonicity].map(v => v != null), [true, true, true]);
eq('bad json -> null', P.parseDiagnosis('{nope'), null);

// --- 5. inspect output ------------------------------------------------------
console.log('[5] parseInspect');
const ins = P.parseInspect(INSPECT);
const text = JSON.stringify(ins.fields);
check('keeps architecture', text.includes('tifa-fa'), ins);
check('drops backend chatter from the fields', !text.includes('ggml_vulkan'), ins.fields);

// --- 6. sidecar resolution --------------------------------------------------
console.log('[6] resolveSidecar');
const side = P.resolveSidecar(['a.wav', 'a.TextGrid', 'a.txt'], 'a', {});
eq('prefers TextGrid', side && side.kind, 'textgrid');
const side2 = P.resolveSidecar(['a.wav', 'a.txt'], 'a', {});
eq('falls back to text', side2 && side2.kind, 'text');
eq('no sidecar -> none', P.resolveSidecar(['a.wav'], 'a', {}).kind, 'none');

// --- 7. table sorting + CSV export -----------------------------------------
console.log('[7] sortRows / toCsv');
const table = [
  { name: 'a', agreement: 0.9, confidence: 0.5 },
  { name: 'b', agreement: 0.3, confidence: 0.9 },
  { name: 'c', agreement: 0.6, confidence: 0.1 }
];
const worst = P.sortRows(table, 'agreement', 'asc').map(r => r.name);
eq('worst-first by agreement', worst, ['b', 'c', 'a']);
const byName = P.sortRows(table, 'name', 'asc').map(r => r.name);
eq('by name', byName, ['a', 'b', 'c']);
const csvOut = P.toCsv(table, ['name', 'agreement']);
check('csv header', csvOut.split(/\r?\n/)[0].includes('name'), csvOut);
check('csv row count', csvOut.trim().split(/\r?\n/).length === 4, csvOut);

// --- 8. path helpers --------------------------------------------------------
console.log('[8] path helpers');
eq('baseName win', P.baseName('C:\\x\\y\\a.wav'), 'a.wav');
eq('stemOf', P.stemOf('some/dir/a.wav'), 'a');
eq('isAudio', [P.isAudio('x.wav'), P.isAudio('x.txt')], [true, false]);

// --- 9. optional live end-to-end -------------------------------------------
if (process.argv.includes('--cli')) {
  console.log('\n[9] live CLI end-to-end');
  const CLI = process.env.TIFA_GGML_CLI || '';
  const MODEL = process.env.TIFA_GGML_MODEL || '';
  const WAV = process.env.TIFA_SMOKE_WAV || '';
  if (!CLI || !MODEL || !WAV) {
    console.log('  skip: set TIFA_GGML_CLI, TIFA_GGML_MODEL and TIFA_SMOKE_WAV to run');
  } else {
    const { spawnSync } = require('child_process');
    const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tifa-smoke-'));
    const r = spawnSync(CLI, ['align', WAV, '-m', MODEL, '--text', 'AP a',
                              '-o', outDir, '--output-formats', 'textgrid,json', '-q'],
                        { encoding: 'utf8', timeout: 300000 });
    check('cli exits 0', r.status === 0, r.stderr);
    const produced = fs.existsSync(outDir) ? fs.readdirSync(outDir) : [];
    check('writes a TextGrid', produced.some(f => f.endsWith('.TextGrid')), produced);
    check('writes a diagnosis JSON', produced.some(f => f.endsWith('.diagnosis.json')), produced);
  }
}

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
