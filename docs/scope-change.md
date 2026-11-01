# 项目范围变更：从“数据集工具链”回到“TIFA 推理引擎” / Scope change: back to the TIFA inference engine

> 本文件解释 0.2.0 为什么是一次**收缩**，而不是一次功能升级。
> This document explains why 0.2.0 is a **scope reduction**, not a feature upgrade.

---

## 中文

### 一句话

TIFA 模型本身已经稳定；不稳定的是围绕它的 G2P、呼吸检测、词典和桌面工具。
从 0.2.0 起，本项目只做一件事：**给定音频和已知音素序列，用 TIFA 模型预测音素边界。**

### 为什么要收缩

`v0.1.x` 试图把“歌声数据集标注”的整条流水线都装进这个仓库：

- **G2P 的变种太多。** 中文要拼音词典（还要处理多音字、儿化、分词），粤语要
  粤拼词典，日语要假名转换、汉字还要 MeCab + UniDic 形态分析，英语要
  CMUdict 加 LSTM 生词推理；上游还定义了 PFML 这种把音素/语言直接写进文本的
  标记语言。每一种语言都有独立的词典格式、许可证、体积和上游更新节奏。
- **“全方面适配”的边际成本越来越高。** 每加一种词典或一种输入变体，就要同时
  改解析器、候选网格、打分、测试、发布包、CI 缓存和文档；而这些内容与 TIFA
  网络本身没有任何关系。
- **附属模型也在分叉。** 呼吸检测同时存在 BreathLab 和 FoxBreatheLabeler，
  输出帧率、标签集和精度都不同；桌面端又是一个独立的 Electron 工程。
- **结果**：仓库的大部分体积、构建失败面和 issue 都落在附属功能上，而真正被
  复用、也真正稳定的部分——TIFA 的前向、Viterbi 对齐和 TextGrid 输出——反而
  被淹没。

TIFA 的接口其实非常干净：**波形 + 音素序列 → 每个音素的时间区间**。把
“文本 → 音素”留在上游/下游，引擎只负责它最擅长的一段，维护成本会低一个
数量级，也更容易被其它工具嵌入。

> 旧的 README 里还留过一条警告：字音识别与原版差异较大、尚未解决。那类差异
> 正是附属 G2P 层的问题；0.2.0 不再内置 G2P，字音转换交给外部工具，
> 这类“与原版一致”的承诺也就不再由本仓库承担。

### 具体改了什么

| 项目 | `v0.1.x`（已存档） | `v0.2.0`（当前） |
|---|---|---|
| G2P / PFML | `src/g2p/`、`include/tifa_ggml/g2p.h`、`tifa_ggml_g2p` 库 | **删除** |
| 呼吸 / AP-SP 检测 | `src/breath/`、`breathe` 命令、FBL/BreathLab 模型 | **删除** |
| 桌面工具 | `ui/`（Electron，TIFA Label） | **删除** |
| 发音词典 / MeCab | `dicts/`、`cmake/Mecab.cmake`、UniDic 发布资产 | **删除** |
| CLI | `tifa_ggml_cli align/breathe/inspect`，支持 `--text` 自动 G2P | `tifa align/inspect`，**只接受音素** |
| GGUF | 内嵌完整 G2P 管线配置 | 只保留词表与 global/stop 符号 |
| 发布物 | GUI 包 + CLI 包 + 词典/呼吸模型 | 只有 `tifa-cli-<平台>-{full,q4}.tar.gz` |
| 版本 | 0.1.6 | 0.2.0（破坏性范围变更） |

保留的仍然是完整的推理能力：log-mel 前端、JEBF backbone、余弦相似度、
Viterbi 解码、诊断指标（agreement/confidence/determinacy/monotonicity）、
TextGrid 三层输出、JSON 诊断、音素/TextGrid/DiffSinger CSV 输入、
CPU/Vulkan/Metal/CUDA 后端，以及 2PASS 时从第一遍 TextGrid 继承
`texts`/`words` 分层的能力。

### 旧版本在哪里

- Git 分支：`archive/main-v0.1.6`（`v0.1.6` 的完整代码、G2P、breath、UI、
  词典和所有脚本）。
- Release：`v0.1.6` 及更早的 GUI/CLI 包保持原样，不删除。

```bash
git fetch origin archive/main-v0.1.6
git switch --track origin/archive/main-v0.1.6
```

### 迁移指南

如果你的流程原来是 `tifa_ggml_cli align song.wav -l zh`（靠 `.txt`/`.lab`
自动 G2P），现在需要先自己得到音素序列，再交给引擎。推荐：

1. 使用上游的 [openvpi/g2pflow](https://github.com/openvpi/g2pflow)，或你自己的
   G2P/标注工具，把文本转成音素；
2. 通过以下任一方式把音素交给 `tifa`：
   - `--phones "AP zh e n"`；
   - `--phones-file phones.txt`（空白分隔）；
   - `--transcriptions-csv transcriptions.csv`（DiffSinger 的 `ph_seq` 列）；
   - `--textgrid song.TextGrid`（读取 `phones` 层，可用于 2PASS）。
3. 其余参数（`-l`、`--skip-*`、`--output-formats`、`--backend`）保持不变。

旧的 `breathe --merge` + 2PASS 流程不再提供；需要呼吸标注时，请用独立的
呼吸检测工具先把 `AP`/`SP` 写进 `phones` 层，再用 `tifa align --textgrid`。

### 我们承诺什么

- 引擎的输入契约稳定：**音频 + 音素 + 语言**。
- 模型格式稳定：`tifa-fa` GGUF；0.1.x 的 GGUF 仍然可以加载（多出来的
  `g2p` 元数据会被忽略）。
- 不再为任何特定 G2P 词典、分词器、呼吸模型或 GUI 框架的变种做适配；
  这些应当由独立项目维护。

---

## English

### In one sentence

The TIFA model is stable; what kept changing was everything around it — G2P,
breath detection, dictionaries and the desktop app.  Starting with 0.2.0 this
project does exactly one thing: **given audio and a known phone sequence, run
the TIFA model and predict phone boundaries.**

### Why the reduction

`v0.1.x` tried to own the whole singing-dataset annotation pipeline:

- **G2P has too many variants.** Mandarin needs a pinyin dictionary (plus
  polyphones, erhua and word segmentation), Cantonese needs jyutping, Japanese
  needs kana conversion and — for kanji — MeCab + UniDic morphological
  analysis, English needs CMUdict plus an LSTM for OOV words; upstream also
  defines PFML, a markup language that embeds phonemes and language scopes in
  the transcript.  Every language brings its own dictionary format, license,
  download size and upstream release cadence.
- **“Support every variant” has runaway marginal cost.** Each new dictionary
  or input flavour touches the parser, the candidate grid, scoring, tests, the
  release bundles, CI caches and the docs — none of which has anything to do
  with the TIFA network.
- **The auxiliary models forked too.** Breath detection exists as both
  BreathLab and FoxBreatheLabeler, with different frame rates, label sets and
  accuracy; the desktop tool is a separate Electron project.
- **Net effect:** most of the repository, its build failures and its issues
  were about the auxiliaries, while the part that is actually reused and
  actually stable — the TIFA forward pass, Viterbi alignment and TextGrid
  output — was buried.

TIFA's interface is clean: **waveform + phone sequence → one time span per
phone.**  Leaving “text → phones” to upstream/downstream tools makes the
engine an order of magnitude cheaper to maintain and much easier to embed.

> The old README carried a warning that word/pronunciation recognition differed
> noticeably from upstream and was unresolved.  That difference belonged to the
> bundled G2P layer; 0.2.0 no longer ships G2P, so that compatibility promise is
> no longer this repository's to make.

### What changed

| Area | `v0.1.x` (archived) | `v0.2.0` (current) |
|---|---|---|
| G2P / PFML | `src/g2p/`, `include/tifa_ggml/g2p.h`, `tifa_ggml_g2p` library | **removed** |
| Breath / AP-SP | `src/breath/`, `breathe` command, FBL/BreathLab models | **removed** |
| Desktop tool | `ui/` (Electron, TIFA Label) | **removed** |
| Dictionaries / MeCab | `dicts/`, `cmake/Mecab.cmake`, UniDic release asset | **removed** |
| CLI | `tifa_ggml_cli align/breathe/inspect`, `--text` auto-G2P | `tifa align/inspect`, **phones only** |
| GGUF | embeds the full G2P pipeline config | vocabulary + global/stop symbols only |
| Releases | GUI + CLI + dictionaries/breath models | only `tifa-cli-<platform>-{full,q4}.tar.gz` |
| Version | 0.1.6 | 0.2.0 (breaking scope change) |

Everything needed for inference remains: the log-mel front-end, the JEBF
backbone, cosine similarity, Viterbi decoding, diagnosis metrics
(agreement/confidence/determinacy/monotonicity), three-tier TextGrid output,
diagnosis JSON, phone/TextGrid/DiffSinger-CSV inputs, CPU/Vulkan/Metal/CUDA
backends, and the 2PASS ability to inherit `texts`/`words` tiers from a
first-pass TextGrid.

### Where the old version lives

- Git branch `archive/main-v0.1.6` — the complete `v0.1.6` tree (G2P, breath,
  UI, dictionaries, all scripts).
- Release `v0.1.6` and older keep their GUI/CLI assets; nothing was deleted.

```bash
git fetch origin archive/main-v0.1.6
git switch --track origin/archive/main-v0.1.6
```

### Migration

If your pipeline used `tifa_ggml_cli align song.wav -l zh` with a sidecar
`.txt`/`.lab` (auto-G2P), produce the phone sequence yourself first, then feed
it to the engine:

1. Use upstream [openvpi/g2pflow](https://github.com/openvpi/g2pflow), or your
   own G2P/annotation tool, to turn text into phones.
2. Pass them to `tifa` in any of these forms:
   - `--phones "AP zh e n"`;
   - `--phones-file phones.txt` (whitespace-separated);
   - `--transcriptions-csv transcriptions.csv` (DiffSinger `ph_seq` column);
   - `--textgrid song.TextGrid` (reads the `phones` tier; also used for 2PASS).
3. The other flags (`-l`, `--skip-*`, `--output-formats`, `--backend`) are
   unchanged.

The old `breathe --merge` + 2PASS pipeline is gone.  If you need breath
annotation, run a separate breath detector first and write `AP`/`SP` into the
`phones` tier, then call `tifa align --textgrid`.

### What we commit to

- A stable engine contract: **audio + phones + language**.
- A stable model format: `tifa-fa` GGUF; 0.1.x GGUFs still load (their extra
  `g2p` metadata is ignored).
- No more adaptation to particular G2P dictionaries, tokenizers, breath models
  or GUI frameworks; those belong in their own projects.
