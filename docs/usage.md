# TIFA Label · 使用说明 / Usage Guide

本文件随发布包分发。中文在前，English follows.

---

# 中文

## 包内有什么

**图形工具包（`tifa-label-<平台>.zip` / `.tar.gz`，三个平台各一份）**

```
TIFA Label(.exe / .app)   图形界面（数据集标注工具）
tifa_ggml_cli(.exe)       对齐引擎（GUI 自动使用）
ggml*                     引擎依赖（.dll / .so / .dylib）
models/
  tifa-1.0-st-q4_0.gguf   对齐模型（Q4_0）
  breath-fbl-q4_0.gguf    呼吸/AP 检测模型（FoxBreatheLabeler，**默认**）
  breath-v5-24k-f16.gguf  呼吸检测模型（BreathLab，备选）
  dictionaries/           发音词典（中文/粤语/日语/英语）
  cpp_pinyin/             汉语拼音引擎词典
  assets/LstmG2p-Eng.gguf 英文 OOV 推理模型
USAGE.md                  本文件
```

**命令行包（`tifa-cli-<平台>-full/-q4.tar.gz`）**：同上的引擎 + 动态库 +
`models/`，没有图形界面；`full` 用 F16 全精度模型，`q4` 用 Q4_0 最小模型。

> 每个包都自带模型、词典与运行库，**下载一个就能用，不需要自己拼装**。

## 快速开始（图形界面）

1. 解压后直接运行：Windows 双击 `TIFA Label.exe`；Linux 运行 `./TIFA Label`；
   macOS 打开 `TIFA Label.app`（未签名，首次需右键 →「打开」）。
2. **不需要任何设置**：引擎、对齐模型、呼吸模型都会从程序旁边的默认相对
   路径自动导入（状态灯变绿即就绪）。只有把模型放到别处时才需要手动选择。
3. 三步得到数据集标注：
   1. **导入音频**：拖入文件或文件夹（递归扫描）。
   2. **标注来源**：文本转录（自动 G2P，支持 PFML）或已有音素标注
      （同名 TextGrid / DiffSinger `transcriptions.csv` / TextGrid 文件夹 /
      内联音素序列）。**已有 `.lab`/音素标注的工作流不会触发 G2P。**
   3. **数据集流程**：第一遍对齐 →（可选）呼吸检测 AP/SP 并合并 →
      （可选）2PASS 重对齐。点“开始标注”。
4. 输出为 `<名称>.TextGrid` 和 `<名称>.diagnosis.json`（诊断信息，默认导出），
   落在输出目录。

> **每一层都是首尾相接的。** 没有任何音素/词覆盖的时段（解码器的 gap 状态）
> 会写成 `text = ""` 的空区间——这等价于上游 TIFA 经由 Python `textgrid` 包的
> `_fillInTheGaps` 写出来的结果。Praat 两种都能读，但 vLabeler 的 TextGrid
> 标注器声明了 `continuous`，有空隙的 TextGrid 会直接建不了工程。

> **诊断 JSON 默认导出**，这是新流程里的重要工具：里面的 `agreement`、
> `confidence`、`determinacy`、`monotonicity` 可以按文件排序，**只需要人工校对
> 最差的那 10%**，其余直接过。
> 不想要的话在高级选项里取消勾选，或命令行传 `--output-formats textgrid`。

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

## 呼吸模型（AP 检测）

默认用 **FoxBreatheLabeler（FBL）**，包里是 `models/breath-fbl-q4_0.gguf`
（精度与包一致：full 包是 `-f16`）。它直接对原始波形分帧（44.1 kHz，50 fps），
输出 AP 概率，`breathe --merge` 把 AP 折进 phones 层供 2PASS 使用。

**BreathLab**（作者 [Xiantaidu](https://github.com/Xiantaidu)）作为备选一并附带
（`models/breath-v5-24k-f16.gguf`），它多一个 SP 头、100 fps；两者都是
`-m` 指哪个用哪个，格式相同。

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
  不经词典）或 `<scope language="ja">東京</scope>`（指定语言区间）。按 PFML 1.0
  解析：支持完整 `<reading>/<path>/<group>` 树、注释 `<!-- -->` 与 CDATA；
  **未知元素或未知属性会报错**（写错属性名不会被静默忽略）。语言有三种状态：
  不写 = 继承外层 scope、`language=""` = 清空、`language-kind="any"` = 任意
  （与"无语言"不同）。
- **Linux 打不开图形界面**：Electron 依赖一批系统库，最小安装的发行版需要先装：
  `sudo apt install libnss3 libatk-bridge2.0-0 libgtk-3-0 libgbm1 libasound2`
  （其他发行版对应包名类似）。命令行包不受影响。
- **macOS 提示「无法验证开发者」**：包未签名，右键 →「打开」即可；
  或 `xattr -d com.apple.quarantine "TIFA Label.app"`。

---

# English

## What's in the package

**GUI bundle (`tifa-label-<platform>.zip` / `.tar.gz`, one per platform)**

```
TIFA Label(.exe / .app)   the dataset annotation GUI
tifa_ggml_cli(.exe)       the aligner engine (used by the GUI automatically)
ggml*                     engine dependencies (.dll / .so / .dylib)
models/
  tifa-1.0-st-q4_0.gguf   aligner weights (Q4_0)
  breath-fbl-q4_0.gguf    breath/AP detector (FoxBreatheLabeler, the **default**)
  breath-v5-24k-f16.gguf  BreathLab breath/AP detector (the alternative)
  dictionaries/           pronunciation dictionaries (zh/yue/ja/en)
  cpp_pinyin/             Mandarin pinyin engine tables
  assets/LstmG2p-Eng.gguf English OOV inference model
USAGE.md                  this file
```

**CLI bundle (`tifa-cli-<platform>-full/-q4.tar.gz`)**: the same engine +
shared libraries + `models/`, no GUI; `full` carries the F16 weights, `q4` the
Q4_0 ones.

> Every archive is self-contained — **one download is all it takes, nothing has
> to be assembled by hand**.

## Quick start (GUI)

1. Unpack and run it: Windows `TIFA Label.exe`, Linux `./TIFA Label`,
   macOS `TIFA Label.app` (unsigned: right-click > Open the first time).
2. **No configuration needed**: the engine, aligner model and breath model are
   picked up automatically from the default relative paths beside the app
   (their status dots turn green).  Manual selection is only for models kept
   elsewhere.
3. Three steps to dataset annotations:
   1. **Import audio** — files or folders (scanned recursively).
   2. **Annotation source** — text/G2P (PFML accepted) or existing phones
      (same-name TextGrid / DiffSinger `transcriptions.csv` / TextGrid folder /
      an inline list).  The classic `.lab` workflow never invokes G2P.
   3. **Pipeline** — first-pass align → (optional) BreathLab breath AP/SP merged
      into the phones tier → (optional) 2PASS re-align.  Press start.
4. Output per file: `<name>.TextGrid` and `<name>.diagnosis.json` (default).

> **Every tier is written continuously.**  A stretch that no phone or word
> claims — the decoder's gap states — becomes an interval with an empty label,
> which is what upstream TIFA gets from the Python `textgrid` package's
> `_fillInTheGaps`.  Praat reads both, but vLabeler's TextGrid labeler is
> declared `continuous` and refuses a TextGrid that has holes.

> **The diagnosis JSON is exported by default** — it is what makes the workflow
> scale: sort the files by its `agreement`, `confidence`, `determinacy` and
> `monotonicity` and **only proof-read the worst ~10%**.  Untick it in the advanced options, or
> pass `--output-formats textgrid`, to turn it off.

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

## Breath model (AP detection)

The default is **FoxBreatheLabeler (FBL)**, shipped as
`models/breath-fbl-q4_0.gguf` (precision follows the bundle: `-f16` in the full
one).  It frames the raw waveform itself (44.1 kHz, 50 fps) and emits AP
probability; `breathe --merge` folds AP into the phones tier for the 2PASS
stage.

**BreathLab** (by [Xiantaidu](https://github.com/Xiantaidu)) ships alongside as
the alternative (`models/breath-v5-24k-f16.gguf`) -- it adds an SP head and runs
at 100 fps.  Both are selected the same way: point `-m` at the one you want.

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
  (`<scope language="ja">東京</scope>`).  Parsing follows PFML 1.0: full
  `<reading>`/`<path>`/`<group>` trees, comments `<!-- -->` and CDATA are
  accepted, **unknown elements and unknown attributes are errors** (a typo'd
  attribute is not silently ignored).  A language has three states: omitted
  inherits the enclosing scope, `language=""` clears it, and
  `language-kind="any"` means any language (distinct from "no language").
- **The GUI will not start on Linux**: Electron needs a few system libraries;
  on a minimal install add
  `sudo apt install libnss3 libatk-bridge2.0-0 libgtk-3-0 libgbm1 libasound2`
  (similar packages elsewhere).  The CLI bundles are unaffected.
- **macOS says the developer cannot be verified**: the build is unsigned;
  right-click > Open, or `xattr -d com.apple.quarantine "TIFA Label.app"`.
