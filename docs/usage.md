# TIFA Label · 使用说明 / Usage Guide

本文件随发布包分发。中文在前，English follows.

---

# 中文

## 包内有什么

**图形工具包（tifa-label-windows-x64.zip）**

```
TIFA Label.exe            图形界面（数据集标注工具）
tifa_ggml_cli.exe         对齐引擎（GUI 自动使用）
ggml*.dll                 引擎依赖
models/
  tifa-1.0-st-*.gguf      对齐模型（full 或 q4）
  breath-v5-24k-f16.gguf  呼吸检测模型（FBL；可选阶段用）
  dictionaries/           发音词典（中文/粤语/日语/英语）
  cpp_pinyin/             汉语拼音引擎词典
  assets/LstmG2p-Eng.gguf 英文 OOV 推理模型
USAGE.md                  本文件
```

**命令行包（tifa-cli-<平台>-full/-q4.tar.gz）**：同上的引擎 + DLL/动态库 +
`models/`，没有图形界面。

## 快速开始（图形界面）

1. 解压后直接运行 `TIFA Label.exe`。
2. **不需要任何设置**：引擎、对齐模型、呼吸模型都会从程序旁边的默认相对
   路径自动导入（状态灯变绿即就绪）。只有把模型放到别处时才需要手动选择。
3. 三步得到数据集标注：
   1. **导入音频**：拖入文件或文件夹（递归扫描）。
   2. **标注来源**：文本转录（自动 G2P，支持 PFML）或已有音素标注
      （同名 TextGrid / DiffSinger `transcriptions.csv` / TextGrid 文件夹 /
      内联音素序列）。**已有 `.lab`/音素标注的工作流不会触发 G2P。**
   3. **数据集流程**：第一遍对齐 →（可选）呼吸检测 AP/SP 并合并 →
      （可选）2PASS 重对齐。点“开始标注”。
4. 输出为 `<名称>.TextGrid`（可同时导出诊断 JSON），落在输出目录。

## 快速开始（命令行）

```bash
# ① 第一遍对齐：文本 + 音频（自动 G2P）
./tifa_ggml_cli align song.wav -m models/tifa-1.0-st-q4_0.gguf --text-file song.txt -l zh -o out

#    或：已有音素标注（不经过 G2P）
./tifa_ggml_cli align song.wav -m models/tifa-1.0-st-q4_0.gguf \
    --transcriptions-csv transcriptions.csv -l zh -o out

# ② 呼吸检测 AP/SP 并合并进 phones 层
./tifa_ggml_cli breathe song.wav -m models/breath-v5-24k-f16.gguf --merge out -o out

# ③ 第二遍对齐（2PASS）
./tifa_ggml_cli align song.wav -m models/tifa-1.0-st-q4_0.gguf --textgrid out -l zh -o out2
```

> **`-l` 不能省。** 模型是多语言的，音素表按语言限定（`zh/zh`、`ja/a`…），
> 不带 `-l` 时 `zh`、`a` 这类音素查不到，会报
> `phone 'zh' is not in the model vocabulary`。取值：`zh` 普通话、`ja` 日语、
> `yue` 粤语、`en` 英语（英文走 `--text`/`--text-file` 时用 `-l en`）。
> PFML 输入自带 `<scope language>`，不需要 `-l`。

完整流程说明见仓库 `docs/dataset-workflow.md`。

## 两种精度怎么选

| 包 | 模型 | 适用 |
|---|---|---|
| `-full` | F16（全精度） | 追求最高对齐质量；体积大 |
| `-q4` | Q4_0（最小） | 体积约为 full 的一半，边界差异在实测容差内（见仓库 `docs/quant-matrix.md`） |

## 呼吸模型（FBL）

发布包已附带转换好的 `models/breath-v5-24k-f16.gguf`。如果包内没有（源下载
失败时会省略），可以自行转换——发布页同时提供源文件
`breathLab_models_dml.zip`：

```bash
unzip breathLab_models_dml.zip -d models_dml
python scripts/convert_breath_to_gguf.py --model-dir models_dml/models_dml --name v5_24k -o breath.gguf
```

## 常见问题

- **首次运行 GPU 后端很慢**：Vulkan 需要编译管线并写磁盘缓存，之后每次启动
  都很快（本机实测首跑 22 s、复跑 0.23 s）。
- **没有独立显卡 / 想强制 CPU**：`--backend cpu`（GUI 在高级选项里）。
- **词典找不到**：词典以文件形式放在模型目录旁（`models/dictionaries/`）。
  换用自己的模型目录时，把 `dictionaries/`、`cpp_pinyin/`、`assets/` 一起放过去。
- **PFML**：文本里可以写 `<word phonemes="zh ong">重</word>`（直接给音素，
  不经词典）或 `<scope language="ja">東京</scope>`（指定语言区间）。

---

# English

## What's in the package

**GUI bundle (`tifa-label-windows-x64.zip`)**

```
TIFA Label.exe            the dataset annotation GUI
tifa_ggml_cli.exe         the aligner engine (used by the GUI automatically)
ggml*.dll                 engine dependencies
models/
  tifa-1.0-st-*.gguf      aligner weights (full or q4)
  breath-v5-24k-f16.gguf  FBL breath/AP detector (optional stages)
  dictionaries/           pronunciation dictionaries (zh/yue/ja/en)
  cpp_pinyin/             Mandarin pinyin engine tables
  assets/LstmG2p-Eng.gguf English OOV inference model
USAGE.md                  this file
```

**CLI bundle (`tifa-cli-<platform>-full/-q4.tar.gz`)**: the same engine +
shared libraries + `models/`, no GUI.

## Quick start (GUI)

1. Unzip and run `TIFA Label.exe`.
2. **No configuration needed**: the engine, aligner model and breath model are
   picked up automatically from the default relative paths beside the app
   (their status dots turn green).  Manual selection is only for models kept
   elsewhere.
3. Three steps to dataset annotations:
   1. **Import audio** — files or folders (scanned recursively).
   2. **Annotation source** — text/G2P (PFML accepted) or existing phones
      (same-name TextGrid / DiffSinger `transcriptions.csv` / TextGrid folder /
      an inline list).  The classic `.lab` workflow never invokes G2P.
   3. **Pipeline** — first-pass align → (optional) FBL breath AP/SP merged
      into the phones tier → (optional) 2PASS re-align.  Press start.
4. Output per file: `<name>.TextGrid` (plus an optional diagnosis JSON).

## Quick start (CLI)

```bash
# (1) first-pass align: text + audio (G2P)
./tifa_ggml_cli align song.wav -m models/tifa-1.0-st-q4_0.gguf --text-file song.txt -l zh -o out

#     or: existing phone annotations (no G2P)
./tifa_ggml_cli align song.wav -m models/tifa-1.0-st-q4_0.gguf \
    --transcriptions-csv transcriptions.csv -l zh -o out

# (2) breath/AP-SP detection merged into the phones tier
./tifa_ggml_cli breathe song.wav -m models/breath-v5-24k-f16.gguf --merge out -o out

# (3) second-pass align (2PASS)
./tifa_ggml_cli align song.wav -m models/tifa-1.0-st-q4_0.gguf --textgrid out -l zh -o out2
```

> **`-l` is not optional.** The model is multilingual and its symbols are
> language-qualified (`zh/zh`, `ja/a`, ...), so without `-l` a phone such as
> `zh` or `a` cannot be resolved and the run fails with
> `phone 'zh' is not in the model vocabulary`. Values: `zh` Mandarin,
> `ja` Japanese, `yue` Cantonese, `en` English (use `-l en` with
> `--text`/`--text-file`). PFML input carries its own `<scope language>` and
> needs no `-l`.

The pipeline rationale lives in `docs/dataset-workflow.md`.

## Which precision to pick

| bundle | model | notes |
|---|---|---|
| `-full` | F16 (full precision) | best alignment quality; larger download |
| `-q4` | Q4_0 (smallest) | about half the size; boundary drift within the measured tolerances (`docs/quant-matrix.md`) |

## Breath model (FBL)

The bundles ship `models/breath-v5-24k-f16.gguf`.  When it is absent (the
source download failed during packaging), convert it yourself from the
`breathLab_models_dml.zip` published beside this archive:

```bash
unzip breathLab_models_dml.zip -d models_dml
python scripts/convert_breath_to_gguf.py --model-dir models_dml/models_dml --name v5_24k -o breath.gguf
```

## Troubleshooting

- **First GPU run is slow**: Vulkan compiles its pipelines and writes a disk
  cache; later starts are fast (22 s first run vs 0.23 s afterwards on our
  machine).
- **No discrete GPU / force CPU**: `--backend cpu` (GUI: advanced options).
- **Dictionaries not found**: they are files beside the model directory
  (`models/dictionaries/`).  When using your own model directory, copy
  `dictionaries/`, `cpp_pinyin/` and `assets/` along with it.
- **PFML**: transcripts may embed final phonemes directly
  (`<word phonemes="zh ong">重</word>`) or language scopes
  (`<scope language="ja">東京</scope>`).
