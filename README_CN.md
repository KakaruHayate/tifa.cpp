# tifa.cpp — TIFA（Token-Imputing Forced Aligner）的 ggml 原生实现

[openvpi/TIFA](https://github.com/openvpi/TIFA) 的 C++ 原生推理——用于歌声数据集
制作的多语言强制对齐器。支持 CPU / Vulkan / Metal / CUDA，运行时不需要 Python；
附带桌面工具 **TIFA Label**（歌声数据集标注工具：导入音频 → 导出 TextGrid）。

> English: [README.md](README.md) · 数据集总流程：[docs/dataset-workflow.md](docs/dataset-workflow.md) · 发布包使用说明（双语）：[docs/usage.md](docs/usage.md)

## 下载

预编译包见 [Releases](https://github.com/KakaruHayate/tifa.cpp/releases)：

| 包 | 内容 |
|---|---|
| `tifa-label-<平台>.zip` / `.tar.gz` | 图形标注工具（Electron + 引擎 + 模型 + 词典），解压即用 |
| `tifa-cli-<平台>-full.tar.gz` | 命令行版，F16 全精度模型 |
| `tifa-cli-<平台>-q4.tar.gz` | 命令行版，Q4_0 最小模型 |

`<平台>` = `windows-x64` / `linux-x64` / `macos-arm64`，三个平台都有图形工具与
命令行包。**每个包都自带模型、词典与运行库，下载一个就能用，不需要自己拼装**；
用法见包内 `USAGE.md`（中英双语）。想自己构建见下方「构建」。

唯一的例外是日语汉字：其 MeCab/UniDic 词典（压缩后约 49 MB）是单独的
`unidic-lite-dicdir.zip` release 资产——GUI 里一键安装，或自己解压到
`models/unidic/`。不装它假名歌词照常可用。

> macOS 的包未签名，首次打开需右键 →「打开」。

## 功能

给定一段录音和它的文本（或已知的音素序列），TIFA 预测音素级时间边界，写出
Praat 三层 TextGrid（`texts` / `words` / `phones`），并可选输出自检指标
（agreement / monotonicity 等）用于筛查需要人工复核的对齐。

```
音频 ──► log-mel（48 kHz，hop 480，80 bins）
     ──► JEBF 主干（8 层，dim 256，联合注意力，RoPE，qk-norm）
     ──► 帧/音素特征 + token logits
文本 ──► G2P ──► 候选网格 ──►（可选读音打分）
     ──► 余弦相似度 [T, N] ──► Viterbi 解码 ──► 时长区间 ──► TextGrid
```

## 亮点

- **与参考实现逐层一致** —— 边界完全一致，浮点差异 ≤ 1e-3。
- **快** —— 20 段基准下 RTX 2070（Vulkan，F16）每文件 0.15 s，约 42× 实时；
  CPU F16 约 11.5× 实时。CgMLP 的深度可分离卷积走专用的
  `GGML_OP_CONV_2D_DW` 核（无 im2col），与 game.cpp 相同的 direct 路径。
- **多语言 G2P** —— 中文（拼音/汉字）、粤语（粤拼）、日语（假名开箱即用；
  汉字经 MeCab + UniDic 形态分析——`unidic-lite-dicdir` release 资产，GUI
  一键安装）、英文（词典 + LSTM OOV 推理）；完整保留模型训练时使用的候选网格。
- **支持 PFML** —— 上游 openvpi 要求的 Pronunciation Flow Markup Language：
  文本里可以直接固定音素与语言区间（`<word phonemes="zh ong">重</word>`、
  `<scope language="ja">東京</scope>`）。
- **完整数据集流程** —— TIFA 对齐 → BreathLab 呼吸检测（AP/SP）合并 → 2PASS
  重对齐，三步均在 CLI 与图形工具中实现（见 dataset-workflow.md）。
- **量化矩阵** —— F32/F16/Q8_0/Q4_0 逐张量配方与边界误差实测
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

构建选项：`TIFA_GGML_VULKAN`、`TIFA_GGML_CUDA`、`TIFA_GGML_METAL`、
`TIFA_GGML_BUILD_CLI`、`TIFA_GGML_BUILD_TESTS`、`TIFA_GGML_BUILD_MODEL`
（关掉则只构建 G2P/PFML 库，见下）。CI 覆盖 8 个平台×后端组合
（Linux/Windows/macOS × CPU/Vulkan/CUDA/Metal）；Release 每个平台产出 GUI 包
（自包含）+ 全精度（F16）与最小（Q4_0）两档 CLI 包。

## G2P / PFML 库（供下游工具使用）

G2P 与 PFML 可以脱离 ggml 单独构建、单独链接——做标注编辑器这类"对齐器之前"
的工具时不需要模型、后端或 GGUF：

```cmake
set(TIFA_GGML_BUILD_MODEL OFF)          # 只要 tifa_ggml_g2p，不拉 ggml
add_subdirectory(tifa.cpp)
target_link_libraries(your_editor PRIVATE tifa_ggml::g2p)
```

接口在 [`include/tifa_ggml/g2p.h`](include/tifa_ggml/g2p.h)：文本/PFML 转词
（`convert` / `convert_pfml`）、读回 PFML（`to_pfml`，带 round-trip 保证）、
纯语法校验（`validate_pfml`，不需要模型）、候选读音（`candidates`）与音素
校验（`resolve_phoneme`）。英文 OOV 的 LSTM 转换器依赖 ggml，在该配置下会被
编译掉，英文生词回退到词典。

## 模型

从 [TIFA releases](https://github.com/openvpi/TIFA/releases) 下载
`TIFA-1.0-ST.zip` 并转换：

```bash
python scripts/convert_tifa_to_gguf.py \
    --model-dir models/TIFA-1.0-ST -o models/tifa.gguf --dtype f16

# 英文 G2P 的 OOV 推理模型（可选，有英文时才需要）
python scripts/convert_lstm_g2p_to_gguf.py \
    --model-dir models/TIFA-1.0-ST/assets/LstmG2p-Eng \
    -o models/assets/LstmG2p-Eng.gguf

# 呼吸检测模型（BreathLab，作者 Xiantaidu，ONNX → GGUF）
python scripts/convert_breath_to_gguf.py \
    --model-dir breathlab/models_dml --name v5_24k -o models/breath-v5-24k.gguf
```

GGUF 内嵌词表与 G2P 管线配置；发音词典以文件形式放在模型目录旁：

```
models/
  tifa.gguf
  dictionaries/   ds-zh-pinyin-lite.txt / jyutping_dict.txt / japanese_dict_full.txt / ds_cmudict-07b.txt
  cpp_pinyin/     汉语拼音引擎词典（mandarin / cantonese）
  assets/         LstmG2p-Eng.gguf（英文 OOV）
  unidic/         （可选）MeCab/UniDic 词典——日语汉字歌词需要；从 release 下载
                  unidic-lite-dicdir.zip 解压到这里，或在 GUI"附加内容"里一键安装
```

## 命令行

```bash
# 已知音素序列（DiffSinger 数据集）
tifa_ggml_cli align wavs/ -m models/tifa.gguf \
    --transcriptions-csv transcriptions.csv -l zh -o out --output-formats textgrid,json

# TextGrid / 内联音素
tifa_ggml_cli align song.wav -m models/tifa.gguf --textgrid song.TextGrid -l zh
tifa_ggml_cli align song.wav -m models/tifa.gguf --phones "AP zh e n a"

# 文本 + 音频（自动 G2P；音频旁的 <名称>.txt / .lab 会自动读取；支持 PFML）
tifa_ggml_cli align song.wav -m models/tifa.gguf -l zh

# 呼吸检测 AP/SP；--merge 折进第一遍的 phones 层（2PASS 第二步）
tifa_ggml_cli breathe song.wav -m models/breath-v5-24k.gguf --merge out -o out

# 第二遍对齐（2PASS 第三步）
tifa_ggml_cli align song.wav -m models/tifa.gguf --textgrid out -o out2

tifa_ggml_cli inspect models/tifa.gguf
```

- `--skip-handling discard|omit|preserve`：零宽（被跳过）音素的处理方式。
- `--skip-penalty`：跳过一个音素的原始余弦代价（0.5 与参考实现一致）。
- `--backend cpu|vulkan|cuda|metal|auto`、`--output-formats textgrid,json`。
- `--oov-handling raise|discard|force`：G2P 生词的处理方式。

## 桌面工具（TIFA Label）

`ui/` 是一个 Electron 应用——**TIFA Label 歌声数据集标注工具**（批量流程）：

1. **导入音频**：拖入文件或文件夹（递归扫描）。
2. **标注来源**：文本转录（自动 G2P，支持 PFML）或已有音素标注
   （自动查找同名 TextGrid / DiffSinger `transcriptions.csv` / TextGrid 文件夹 /
   内联音素序列）——旧工作流的 `.lab + wav` 直接走第二条，不触发 G2P。
3. **数据集流程**：第一遍对齐 →（可选）BreathLab 呼吸检测并合并 →（可选）2PASS。

输出 TextGrid **与诊断 JSON（默认导出，用于按质量排序、只校对最差的 10%）**，
逐文件状态与进度、流式日志、可取消。
界面为中文。发布包里引擎与模型都在程序旁的默认路径上，**开箱即用、无需配置**
（只有把模型放在别处时才需要手动选择）。

## 目录结构

```
src/                 引擎：backend / gguf_io / tensor_utils / mel / ops_* / model_tifa
src/g2p/             拼音引擎、转换器（含英文 LSTM、PFML）、候选网格、选择
src/breath/          BreathLab 呼吸检测（BreathLab ONNX 移植为 ggml 图）
src/cli/             tifa_ggml_cli（align / breathe / inspect）
include/tifa_ggml/   公开 C++ API（PIMPL，头文件不暴露 ggml）
scripts/             转换器、参考导出、golden 对比、量化矩阵、评测、基准
ui/                  Electron 桌面工具（TIFA Label）
cmake/patches/       应用到 ggml 的补丁（含 LSTM 算子、Vulkan pipeline cache）
docs/                基准、量化矩阵、数据集总流程（中文）
AGENT.md             改图之前必读的硬性约束
```

## 许可

MPL-2.0（本移植）。TIFA 的模型权重与词典遵循上游各自的许可。
