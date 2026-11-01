# tifa · 使用说明 / Usage Guide

本文件随发布包分发。中文在前，English follows.

---

# 中文

## 包内有什么

```
tifa(.exe)                推理引擎（CLI）
ggml*                     引擎依赖（.dll / .so / .dylib）
models/tifa.gguf          对齐模型（full 包为 F16，q4 包为 Q4_0）
USAGE.md                  本文件
LICENSE / THIRD_PARTY_NOTICES.md
```

**这是一个推理引擎，不是完整流水线。** 它不会把文本转成音素：你需要自己提供
音素序列。原因与迁移方法见仓库 `docs/scope-change.md`。

## 快速开始

```bash
# 1) 先确认模型能加载
./tifa inspect models/tifa.gguf

# 2) 内联音素
./tifa align song.wav -m models/tifa.gguf --phones "AP zh e n" -l zh -o out

# 3) 音素文件（空白分隔）
./tifa align song.wav -m models/tifa.gguf --phones-file song.txt -l zh -o out

# 4) 已有 Praat TextGrid（读取 phones 层）
./tifa align song.wav -m models/tifa.gguf --textgrid song.TextGrid -l zh -o out

# 5) DiffSinger transcriptions.csv（读取 ph_seq 列）
./tifa align wavs/ -m models/tifa.gguf --transcriptions-csv transcriptions.csv -l zh -o out
```

> **`-l` 通常不能省。** 模型是多语言的，音素按语言限定（`zh/zh`、`ja/a`…），
> 不带 `-l` 时 `zh`、`a` 这类音素查不到，会报
> `phone 'zh' is not in the model vocabulary`。取值：`zh` 普通话、`ja` 日语、
> `yue` 粤语、`en` 英语。音素本身已经带语言前缀时可省略。

## 输出

每个输入文件产出：

- `<名称>.TextGrid`：三层 `texts` / `words` / `phones`。没有音素覆盖的时段写成
  空标签区间，保证每层首尾相接（vLabeler 等要求 `continuous` 的工具可直接用）。
- `<名称>.diagnosis.json`：`agreement`、`confidence`、`determinacy`、
  `monotonicity` 等自检指标，可按文件排序，只人工校对最差的一小部分。

不想要 JSON 时传 `--output-formats textgrid`。`--fill-gaps LABEL`
（默认 `SP`，传空字符串则留空）控制未被覆盖时段的标签。

## 2PASS

先跑一遍 `align`，用任意工具编辑生成的 `phones` 层（例如由独立的呼吸检测器
写入 `AP`/`SP`），再带 `--textgrid` 重跑：

```bash
./tifa align song.wav -m models/tifa.gguf -l zh -o pass1
# 编辑 pass1/song.TextGrid 的 phones 层
./tifa align song.wav -m models/tifa.gguf --textgrid pass1 -l zh -o pass2
```

第一遍的 `texts`/`words` 分组会按新的音素边界继承下来。

## 两种精度怎么选

| 包 | 模型 | 适用 |
|---|---|---|
| `-full` | F16（全精度） | 追求最高对齐质量；体积大 |
| `-q4` | Q4_0（最小） | 体积约为 full 的一半，边界差异在实测容差内（见仓库 `docs/quant-matrix.md`） |

## 常见问题

- **首次运行 GPU 后端很慢**：Vulkan 需要编译管线并写磁盘缓存，之后每次启动
  都很快（本机实测首跑 22 s、复跑 0.23 s）。
- **没有独立显卡 / 想强制 CPU**：`--backend cpu`。
- **音素查不到**：先确认 `-l` 与标注语言一致；音素必须来自模型词表
  （global 符号如 `AP`/`SP` 除外）。用 `tifa inspect models/tifa.gguf` 查看
  词表规模与 global/stop 符号。
- **Linux 上缺少运行库**：引擎是普通 CLI，若提示找不到 `libggml.so`，确认
  压缩包内的 `libggml*.so` 与 `tifa` 在同一目录。
- **macOS 提示「无法验证开发者」**：包未签名，右键 →「打开」即可；
  或 `xattr -d com.apple.quarantine tifa`。

---

# English

## What's in the package

```
tifa(.exe)                inference engine (CLI)
ggml*                     engine dependencies (.dll / .so / .dylib)
models/tifa.gguf          aligner weights (F16 in `full`, Q4_0 in `q4`)
USAGE.md                  this file
LICENSE / THIRD_PARTY_NOTICES.md
```

**This is an inference engine, not a full pipeline.**  It does not turn text
into phonemes; you supply the phone sequence.  See `docs/scope-change.md` in
the repository for the rationale and the migration guide.

## Quick start

```bash
# 1) check that the model loads
./tifa inspect models/tifa.gguf

# 2) inline phones
./tifa align song.wav -m models/tifa.gguf --phones "AP zh e n" -l zh -o out

# 3) a whitespace-separated phone file
./tifa align song.wav -m models/tifa.gguf --phones-file song.txt -l zh -o out

# 4) an existing Praat TextGrid (reads the phones tier)
./tifa align song.wav -m models/tifa.gguf --textgrid song.TextGrid -l zh -o out

# 5) a DiffSinger transcriptions.csv (reads the ph_seq column)
./tifa align wavs/ -m models/tifa.gguf --transcriptions-csv transcriptions.csv -l zh -o out
```

> **`-l` is normally required.** The model is multilingual and its symbols are
> language-qualified (`zh/zh`, `ja/a`, ...), so without `-l` a phone such as
> `zh` or `a` cannot be resolved and the run fails with
> `phone 'zh' is not in the model vocabulary`.  Values: `zh` Mandarin,
> `ja` Japanese, `yue` Cantonese, `en` English.  Omit it only when every phone
> is already language-prefixed.

## Output

Per input file:

- `<name>.TextGrid` — three tiers `texts` / `words` / `phones`.  Stretches no
  phone covers become empty-label intervals so every tier is continuous (what
  `continuous` TextGrid labelers such as vLabeler require).
- `<name>.diagnosis.json` — `agreement`, `confidence`, `determinacy`,
  `monotonicity`; sort by it and proof-read only the worst files.

Pass `--output-formats textgrid` to skip the JSON.  `--fill-gaps LABEL`
(default `SP`; empty leaves gaps unlabelled) controls the label of uncovered
stretches.

## 2PASS

Run `align` once, edit the generated `phones` tier with any tool (for example
insert `AP`/`SP` from a separate breath detector), then re-run with
`--textgrid`:

```bash
./tifa align song.wav -m models/tifa.gguf -l zh -o pass1
# edit the phones tier of pass1/song.TextGrid
./tifa align song.wav -m models/tifa.gguf --textgrid pass1 -l zh -o pass2
```

The first pass's `texts`/`words` grouping is inherited onto the new boundaries.

## Which precision to pick

| bundle | model | notes |
|---|---|---|
| `-full` | F16 (full precision) | best alignment quality; larger download |
| `-q4` | Q4_0 (smallest) | about half the size; boundary drift within the measured tolerances (`docs/quant-matrix.md`) |

## Troubleshooting

- **First GPU run is slow**: Vulkan compiles its pipelines and writes a disk
  cache; later starts are fast (22 s first run vs 0.23 s afterwards on our
  machine).
- **No discrete GPU / force CPU**: `--backend cpu`.
- **A phone is not in the vocabulary**: check that `-l` matches the annotation
  language; phones must come from the model vocabulary (global symbols such as
  `AP`/`SP` excepted).  `tifa inspect models/tifa.gguf` prints the vocabulary
  size and the global/stop sets.
- **Missing libraries on Linux**: the engine is a plain CLI; if it cannot find
  `libggml.so`, make sure the bundled `libggml*.so` files sit next to `tifa`.
- **macOS says the developer cannot be verified**: the build is unsigned;
  right-click > Open, or `xattr -d com.apple.quarantine tifa`.
