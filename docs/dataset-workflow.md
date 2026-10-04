# 数据集制作流程（TIFA → 呼吸检测 → 2PASS）

> 本文档描述用 tifa.cpp 制作歌声数据集标注的标准流程。核心三步由 TIFA 作者
> 给出：**先用 TIFA 做一遍强制对齐，再把呼吸（AP/SP）补进标注，最后用 TIFA
> 做第二遍（2PASS）对齐**，得到干净的乐句切分与音素边界。
> 三个步骤在本仓库的 CLI 里均已实现并通过端到端验证。
>
> 中间那步在参考流程里用的是 FBL；本仓库实现的是 **BreathLab** 模型
> （作者 [Xiantaidu](https://github.com/Xiantaidu)，随发布包附带）——它同样输出
> AP/SP 时间线，我们把它折进 phones 层再跑 2PASS。下文提到「呼吸检测」时，
> 指的都是这一步。

## 总览

```
wav + 文本/音素序列
   │
   │  ① tifa_ggml_cli align        （第一遍：整句对齐）
   ▼
<name>.TextGrid                    phones 层：模型放置的音素边界
   │
   │  ② tifa_ggml_cli breathe --merge  （BreathLab 检测 AP/SP 并折进 phones 层）
   ▼
<name>.TextGrid（覆盖）            phones 层 = 音素 + 检测到的 AP/SP
   │                              breath 层 = AP/SP/V 全程时间线（供检查）
   │  ③ tifa_ggml_cli align --textgrid （第二遍：带呼吸标注重新对齐）
   ▼
最终 <name>.TextGrid              音素边界围绕真实呼吸重新排布
```

为什么需要三步：第一遍对齐只保证"文本对得上音频"；歌手的换气（AP）与停顿
（SP）不在文本里，模型会把它们算进相邻音素，导致边界偏移。BreathLab 把呼吸检测出
来并写回标注，第二遍对齐时模型就能把音素边界排到呼吸两侧，得到干净的乐句
切分。

## ① 第一遍：TIFA 对齐

文本 + 音频（自动 G2P，支持中/英/日/粤混合）：

```bash
tifa_ggml_cli align song.wav -m tifa-1.0-st-q4_0.gguf \
    --text-file song.txt -o out/
```

已有音素序列（旧工作流：DiffSinger `transcriptions.csv` / `.lab` / TextGrid /
纯音素列表）——此路径**不触发 G2P**，直接使用标注中的音素：

```bash
tifa_ggml_cli align song.wav -m tifa-1.0-st-q4_0.gguf \
    --transcriptions-csv transcriptions.csv -o out/
```

`.lab` 里的空格分隔音素序列可以直接用 `--phones-file`（或 `--phones "..."`）
传入。产物 `<name>.TextGrid` 含 `texts` / `words` / `phones` 三层，以及可选
的 `<name>.diagnosis.json`（agreement/monotonicity 等自检指标）。

## ② BreathLab 呼吸检测并合并

```bash
tifa_ggml_cli breathe song.wav -m breath-v5-24k-f16.gguf \
    --merge out/ -o out/
```

- `--merge` 指向第一遍的产物（目录或单个 TextGrid 文件）。合并器把检测到的
  AP/SP 折进 `phones` 层：
  - 与已有标注重叠、且标签相同的呼吸**跳过**（不重复插入）；
  - 落在音素内部的长呼吸把该音素**切开**，呼吸插在中间；
  - 落在间隙（句首/句尾静音、句间停顿）的呼吸**原样插入**；
  - 相邻同标签区间合并（呼吸类标签 100 ms 内、其它标签仅严格相接时）；
  - 空区间（未标注段）不写入输出。
- 输出：`<name>.TextGrid`（phones 层 = 增强后的音素序列 + breath 层时间线），
  供第二遍使用；`--min-insert-ms`（默认 50 ms）控制最短插入时长。
- 不需要合并时直接 `tifa_ggml_cli breathe song.wav -m breath.gguf` 即可得到
  独立的 `<name>.breath.TextGrid` + `<name>.breath.json`。

## ③ 第二遍：带呼吸标注重新对齐

```bash
tifa_ggml_cli align song.wav -m tifa-1.0-st-q4_0.gguf \
    --textgrid out/song.TextGrid -o out2/
```

第二遍读取合并后 TextGrid 的 `phones` 层（含 AP/SP），重新对齐。音素边界
此时会围绕呼吸排布，句读更干净。产物同样是三层 TextGrid，可以直接进入
DiffSinger 训练数据制作。

**第二遍会保住乐句行。** 它只从 `phones` 层拿音素，本身产生不出 `texts`/`words`
的内容，所以实现上会把第一遍 TextGrid 里的这两层读回来，按"每个 token 落在哪个
词里"重新贴到新的边界上——词序不变，边界跟着新对齐走。为此第 ② 步的
`breathe --merge` 也会把 `texts`/`words` 一起写进合并产物（它以前只写
`phones` + `breath`，这正是乐句行丢失的原因）。

**空隙会被补上标签。** 对齐器不会覆盖整段音频，未覆盖的地方（句首句尾、句间
停顿）在 Praat 里必须是连续的 interval，所以会被填上标签：`align` 与
`breathe --merge` 的 `--fill-gaps LABEL` 控制它，**默认 `SP`**；传空字符串则
留成空标签（上游 `textgrid` 包的行为）。`texts`/`words` 层不受影响，仍留空。

> 注意：`SP` 是模型的 stop symbol（终止标记），**不参与时长标注**——它进不了
> 对齐器，只作为输出里的标签存在。所以"补 SP"是序列化层面的补齐，不是把 SP
> 塞进模型。`AP`/`EP`/`GS` 是 global 符号，会真正参与第二遍对齐。

## 与本仓库实现的对应关系

| 步骤 | 实现 | 状态 |
|---|---|---|
| ① 第一遍对齐 | `align`（`src/cli/main.cpp`），G2P 管线 `src/g2p/` | ✅ 已有，40+ 中文数据等价性验证 |
| ② BreathLab AP/SP | `src/breath/`（BreathLab ONNX 移植为 ggml 图），`breathe` 命令 | ✅ 端到端验证，6 段/2 AP 检测样例 |
| ②→③ 合并器 | `merge_breath_into_phones`（`src/cli/phone_input.cpp`） | ✅ 6 个单元测试锁定行为 |
| ③ 2PASS | `align --textgrid` + 合并产物 | ✅ 真实数据集 48 音素句子全流程通过 |

模型权重转换：

```bash
# TIFA（已有脚本）
python scripts/convert_tifa_to_gguf.py --model-dir TIFA-1.0-ST -o tifa.gguf
# 呼吸检测器（本仓库新增）
python scripts/convert_breath_to_gguf.py \
    --model-dir <breathlab>/models_dml --name v5_24k -o breath-v5-24k.gguf
```

## 与 openvpi/g2pflow 的关系

上游 TIFA 已把 G2P 流程拆分为独立库
[g2pflow](https://github.com/openvpi/g2pflow)（Python，Pydantic 配置）。本仓库
的 C++ G2P 管线（`src/g2p/`）与该库**同一架构**：相同的 preprocessors
（`filter-punctuation` / `strip-whitespace` / `lowercase` / `remove-accents`）、
相同的 converters（`chinese-pinyin` / `yue-jyutping` / `japanese-kana` /
`dictionary` / `characters` / `passthrough` / `lstm`）、相同的数据模型
（word → readings → paths → groups）。g2pflow 新增的 **PFML 标记语言**
（`<scope language="zh">…</scope>`、`<word phonemes="zh ong">重庆</word>`）
允许在文本里直接固定音素/词边界——这是后续可以加入数据集工具的一个输入
模式（"直接音素 + 自动 G2P 混合文本"），当前版本尚未实现。

## 参考

- 模型与训练：<https://github.com/openvpi/TIFA>
- 数据集工具（HFA）：<https://github.com/openvpi/dataset-tools>
- 呼吸检测模型 BreathLab（作者 [Xiantaidu](https://github.com/Xiantaidu)）：
  随发布包附带 `breathLab_models_dml.zip`（模型源文件）
- G2P 独立库：<https://github.com/openvpi/g2pflow>
