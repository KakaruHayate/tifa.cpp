# tifa.cpp —— TIFA（Token-Imputing Forced Aligner）的 ggml 推理引擎

[openvpi/TIFA](https://github.com/openvpi/TIFA) 的 C++ 原生推理实现。TIFA 是
歌声数据集制作中使用的多语言强制对齐器；本引擎支持 CPU / Vulkan / Metal /
CUDA，运行时不依赖 Python。

**这是一个推理引擎，不是一条完整流水线。** 输入是音频和**已知的音素序列**，
输出是音素级时间边界。0.2.0 起，文本转音素（G2P）、PFML、呼吸检测、发音词典
和 Electron 标注工具都已移除——原因与迁移方法见
[docs/scope-change.md](docs/scope-change.md)。

> English: [README.md](README.md) · 范围变更说明：[docs/scope-change.md](docs/scope-change.md) · 发布包使用说明（中英双语）：[docs/usage.md](docs/usage.md)

## 下载

预编译包见 [Releases](https://github.com/KakaruHayate/tifa.cpp/releases)：

| 包 | 内容 |
|---|---|
| `tifa-cli-<平台>-full.tar.gz` | `tifa` 引擎 + F16 全精度模型 |
| `tifa-cli-<平台>-q4.tar.gz` | `tifa` 引擎 + Q4_0 最小模型 |

`<平台>` = `windows-x64` / `linux-x64` / `macos-arm64`。每个包都自带模型与
运行库，**下载一个就能用**；用法见包内 `USAGE.md`（中英双语）。自行构建见下。

> macOS 包未签名，首次打开需右键 →「打开」。

## 功能

给定一段录音和它的音素序列，TIFA 预测音素级时间边界，写出 Praat 三层
TextGrid（`texts` / `words` / `phones`），并可选输出自检指标，用于筛查需要
人工复核的对齐。

每一层都是首尾相接的：没有任何音素/词覆盖的时段（解码器的 gap 状态）会写成
`text = ""` 的空区间。这与上游 TIFA 经由 Python `textgrid` 包的
`_fillInTheGaps` 写出的结果一致，也是 vLabeler 的 `continuous` TextGrid
标注器所要求的。

```
音频 ──► log-mel（48 kHz，hop 480，80 bins）
     ──► JEBF 主干（8 层，dim 256，联合注意力，RoPE，qk-norm）
     ──► 帧/音素特征 + token logits
音素 ──► 余弦相似度 [T, N] ──► Viterbi 解码 ──► 时间区间 ──► TextGrid
```

## 特性

- **与参考实现逐层对齐** —— PyTorch 参考实现验证：spans 完全一致，浮点差异
  ≤ 1e-3。
- **快** —— RTX 2070（Vulkan，F16）20 段基准上每文件 0.15 s，约 42× 实时；
  CPU F16 约 11.5× 实时。CgMLP 深度卷积走专用 `GGML_OP_CONV_2D_DW` kernel
  （非 im2col），与 game.cpp 同一条直接路径。
- **契约稳定** —— 输入只有音频 + 音素 + 语言；没有词典、分词器或文本预处理。
- **兼容已有标注** —— 支持内联音素、`.txt`、Praat TextGrid、DiffSinger
  `transcriptions.csv`；`--textgrid` 还能在 2PASS 重对齐时继承第一遍的
  `texts`/`words` 分层。
- **量化矩阵** —— F32/F16/Q8_0/Q4_0 逐张量配方及实测边界误差
  （`scripts/quant_matrix.py`）。

## 构建

```bash
# Windows（MSVC）：按需修改 .bat 里的 VS 路径
scripts\configure.bat
scripts\build.bat
# Linux/macOS
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTIFA_GGML_VULKAN=ON
cmake --build build -j
```

选项：`TIFA_GGML_VULKAN`、`TIFA_GGML_CUDA`、`TIFA_GGML_METAL`、
`TIFA_GGML_BUILD_CLI`、`TIFA_GGML_BUILD_TESTS`。可执行文件为
`build/bin/tifa`（Windows 上 `tifa.exe`）。

## 模型

从 [TIFA releases](https://github.com/openvpi/TIFA/releases) 下载
`TIFA-1.0-ST.zip`，然后转换 checkpoint：

```bash
python scripts/convert_tifa_to_gguf.py \
    --model-dir models/TIFA-1.0-ST -o models/tifa-1.0-st-f16.gguf --dtype f16
```

GGUF 内嵌词表与 global/stop 符号集；不再内嵌 G2P 管线配置（引擎不需要）。

## 命令行

`tifa` 必须提供音素来源，不会从文本自动生成音素。

```bash
# 已知音素序列（DiffSinger 数据集）
tifa align wavs/ -m models/tifa-1.0-st-f16.gguf \
    --transcriptions-csv transcriptions.csv -l zh -o out --output-formats textgrid,json

# 从 TextGrid / 内联 / .txt 文件读取音素
tifa align song.wav -m model.gguf --textgrid song.TextGrid -l zh
tifa align song.wav -m model.gguf --phones "AP zh e n a"
tifa align song.wav -m model.gguf --phones-file phones.txt -l zh

tifa inspect model.gguf
```

`--skip-handling discard|omit|preserve` 控制零宽（被跳过）音素的处理；
`--skip-penalty` 是跳过的原始余弦代价（0.5 与参考实现一致）；另有
`--backend cpu|vulkan|cuda|metal|auto`、`--output-formats textgrid,json`、
`--fill-gaps`、`--max-frames`。除非所有音素都已带语言前缀，否则必须用 `-l`
指定语言：模型是多语言的，符号按 `zh/zh`、`ja/a` 这类形式存储。

## 2PASS

引擎仍支持数据集的第二遍对齐：先跑一次 `align`，用外部工具编辑生成的
`phones` 层（例如由独立的呼吸检测器插入 `AP`/`SP`），再带 `--textgrid`
重跑。第一遍的 `texts`/`words` 分组会按新的音素边界继承下来。

```bash
tifa align song.wav -m models/tifa.gguf -l zh -o pass1
# 用任意工具编辑 pass1/song.TextGrid 的 phones 层
tifa align song.wav -m models/tifa.gguf --textgrid pass1 -l zh -o pass2
```

## 目录

```
src/                 引擎：backend / gguf_io / tensor_utils / mel / ops_* / model_tifa
src/cli/             tifa（align / inspect）
include/tifa_ggml/   公开 C++ API（PIMPL，头文件不依赖 ggml）
scripts/             checkpoint 转换、参考 dump、golden 对比、量化矩阵
docs/scope-change.md 为什么 0.2.0 移除 G2P / breath / UI
AGENT.md             改动计算图前必读的踩坑记录
```

## 许可证

MPL-2.0（本移植）。TIFA 模型权重遵循其上游许可证。
