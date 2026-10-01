# AGENT.md — working notes for AI agents & contributors

Read this before touching the build system, the ggml dependency, or the
network graph.  Everything here was learned the hard way during the port; each
item has cost hours at least once already.

---

## 1. ggml version gate — stay on v0.19.0

`cmake/Dependencies.cmake` pins **ggml `v0.19.0`** (URL + `URL_HASH`) and
applies two in-repo patches (Metal binary-archive PSO cache, Vulkan
disk-backed pipeline cache).  Like the sibling `game.cpp` port:

- v0.20.x turns `GGML_CPU_ALL_VARIANTS` into **dlopen MODULE plugins**
  (`GGML_BACKEND_DL`); DL mode does not link the CPU backend into the ggml
  umbrella target, while `src/backend.cpp` calls the CPU backend API directly
  (`ggml_backend_cpu_init`, `ggml_backend_cpu_set_n_threads`,
  `ggml_threadpool_new`, …).
- `GGML_NATIVE` and `GGML_BACKEND_DL` are mutually exclusive upstream.

Do not bump the tag without refactoring `backend.cpp` to load the CPU backend
through the DL registry, and re-run the parity suite afterwards.

---

## 2. Arena aliasing: **inputs are re-uploaded before every compute**

`ggml_gallocr` reuses memory across the graph.  A graph *input* that is
consumed early (e.g. the RoPE position iota) can share its buffer with a
temporary allocated later — writing the input once at graph-build time is
**not** enough.  Symptom in this repo's history: the attention Q/K were
rotated with garbage positions (`float 1.0f` reinterpreted as `int32`
1065353216), which produced plausible-looking but wrong alignments.

Therefore `Model::Impl::run_forward` writes **mel, tokens, padding masks and
position iotas** immediately before `ggml_backend_graph_compute`, every call.

## 3. Debug taps must be copied out of the graph

`ggml_set_output()` alone does not make an intermediate readable after
compute — its memory may already be reused.  Dump taps with

```cpp
ggml_tensor * dst = ggml_dup_tensor(ctx, tap);
ggml_tensor * cpy = ggml_cpy(ctx, tap, dst);
ggml_set_output(cpy);
ggml_build_forward_expand(graph, cpy);   // must happen *before* gallocr alloc
```

A "self-inconsistent dump" (e.g. `qkv != linear(norm)` computed from the same
dump) is the fingerprint of this class of bug.  See
`JebfTaps`/`TIFA_GGML_DUMP_LAYERS=1` in `src/ops_jebf.*` and `src/model_tifa.cpp`.

## 4. Exact-GELU, not tanh

TIFA calls bare `F.gelu(x)` → PyTorch's default **erf** formulation.
`ggml_gelu` is the tanh approximation and differs by up to ~3e-3 per
activation; over 8 layers that alone moved `similarity` by ~1e-2 and changed
decoded spans.  Use `ggml_gelu_erf` everywhere (`src/ops_ffn.cpp`).

## 5. TIFA's JEBF differs from GAME's — do not "fix" it back

- no attention mask at all: padded positions stay in the key/value set (their
  K/V are the projection bias terms).  Padding is applied as a **multiply**
  (`x *= t_mask`, `token *= n_mask`) before each sub-block and on the attention
  outputs, never as an additive mask;
- plain RoPE per stream (token positions `0..N-1`, frame positions `0..T-1`,
  theta 1e4) — no region RoPE;
- **no 0.5 factor** on the FFN residuals (that is the single-stream EBF
  convention);
- both streams are always computed (the token stream has output heads too).

Because padding enters attention, **batch size is part of the numerics**:
reference goldens must be produced with PyTorch `batch_size=1` (which is what
`tifa_ggml_cli` implements: exact-length, unpadded).

## 6. GGUF dtype policy

`scripts/convert_tifa_to_gguf.py:f32_only()` keeps these F32 regardless of
`--dtype`: every bias, every `*norm*.weight`, every LayerScale `.scale`, and
the depthwise conv weights (`.dw.`, `merge_dw_conv_*`).  A missing case bites
immediately on CPU with `binary_op: unsupported types: dst: f32, src0: f32,
src1: f16` — that is how the `token_q_norm` case was found.  Quantizing the
depthwise kernels/`norm` weights is not worth the support matrix; Q8_0/Q4_K
on the 2-D matmul weights is (see `scripts/quant_matrix.py`).

## 7. Load-time tensor rewrites (`src/tensor_utils.cpp`)

- GLU `ln1` weights/biases are split into `.a`/`.b` halves (two `mul_mat`s, no
  strided view + `cont`); `ops_ffn` prefers the halves when present.
- F16 depthwise-conv weights get persistent F32 copies (the direct
  `CONV_2D_DW` path wants F32, the legacy im2col path wants F16).
- The GAME-style **EBF LayerScale folding is disabled** (`enable_layer_scale_fold
  = false`): TIFA applies LayerScale explicitly in `ops_jebf.cpp`, so folding
  would apply it twice.

## 8. Reproducing the reference (goldens)

```bash
# one-off: needs the PyTorch reference checkout and its dependencies
python scripts/dump_tifa_reference.py \
    --tifa-root ../refs/TIFA --model-dir ../models/TIFA-1.0-ST \
    --audio sample48k.wav --phones "AP zh e n" -l zh --out-dir golden/ref

TIFA_GGML_DUMP_LAYERS=1 TIFA_GGML_BACKEND=cpu build/bin/tifa_ggml_cli.exe \
    align sample48k.wav -m ../models/tifa-1.0-st-f32.gguf \
    --phones "AP zh e n" -l zh -o out --dump-dir golden/cpp

python scripts/compare_golden.py --ref golden/ref --cpp golden/cpp --tolerance 1e-3
```

Feed **48 kHz** audio to both sides to keep resampling out of the comparison.
Expected agreement: `mel` ≤ 5e-4, features/logits ≤ 5e-3 (F32 accumulation
order), `spans` **exactly equal**.

## 9. Environment

- MSVC (VS 18 BuildTools) + Ninja; `scripts/configure.bat` / `scripts/build.bat`
  set up vcvars64 and configure with `TIFA_GGML_VULKAN=ON`.
- `TIFA_GGML_BACKEND=cpu|vulkan|cuda|metal` forces a backend (also via the CLI
  `--backend`), `TIFA_GGML_THREADS=N` sets the CPU thread count.
- Measured on an RTX 2070, one 8.7 s clip: CPU F32 ≈ 5.7 s, CPU F16 ≈ 2.9 s,
  Vulkan F16 ≈ 0.19 s.

## 10. Session-2026-10-01 invariants（新增，勿踩）

- **深度卷积 direct 门控必须设置**：`Model::load` 里按后端名调用
  `ops::set_direct_dwconv(...)`（CPU/Vulkan/CUDA/Metal → true）。忘了它 =
  CgMLP 走 im2col，CPU 慢 4.4×（2.96 s → 0.67 s/8.7 s clip）。新加模型加载
  路径时照抄 `model_tifa.cpp` 的闸门。
- **LSTM 解码器的 `in_h` 必须用独立 buffer**
  （`lstm_g2p.cpp::GraphRun::allocate_inputs`）：同一张图被反复 compute、
  每次重传状态，而 gallocr 曾把它的区间复用给 `out_logits`，第二步读到的
  就是被覆盖的状态。**不要把这条推广到所有图输入** —— TIFA 主模型的输入
  （`in_mel`/`in_tokens`/mask/位置）按 §2 建在状态图的 context 里、每次
  compute 前重传即可（golden 验证过的路径）。判断依据是"同一张图是否会被
  反复 compute 且输入区间可能被后续节点写回"，而不是"输入"这个身份本身。
- **breath 的 taps 需要 `ggml_cpy` 复制**（AGENT.md §3）。
- **mid-graph 调试读回先 cpy**：直接读中间节点是活内存，结果随机
  （tifa 词正确、test 词乱码的那类诡异现象即此）。
- **PFML 是文本入口的一部分**：`build_text_request` 先 `looks_like_pfml`
  再分流；`<word phonemes=...>` 直出音素，不经词典/预处理器（注意：普通
  文本会先 lowercase，直接音素不要放普通文本里）。
- **发布包布局**：`models/{tifa.gguf, dictionaries/, cpp_pinyin/, assets/LstmG2p-Eng.gguf}`。
  GGUF 里 `@dictionaries/...`、`@assets/...` 相对模型目录解析，平铺即失效。
- **发布包必须在打包处"就地跑一次"**：workflow 组装完 bundle 后，
  `cd` 进 bundle 目录执行 `./tifa_ggml_cli --version` 与
  `inspect models/tifa.gguf`，失败即 fail job。教训：2026-10-01 的 GUI 包
  里 `tifa_ggml_cli.exe` 是 MinGW 链接的（`libgcc_s_seh-1.dll`/
  `libstdc++-6.dll`/`libwinpthread-1.dll`/`libgomp-1.dll`），而这些 DLL
  没随包发布 —— 在干净 Windows 上直接加载失败、**零输出退出**。构建树里跑
  得好好的，所以只有"在打包产物里跑"才暴露得出来。
- **Windows 构建不要写 `-G Ninja`**：runner 的 PATH 上有 Strawberry Perl 的
  gcc，Ninja 会优先选它 → MinGW 工具链。留空让 CMake 用默认的 Visual Studio
  生成器（MSVC）。GUI 包与 CLI 包两个 job 必须用同一套工具链。
- **`CMAKE_BUILD_RPATH_USE_ORIGIN ON` 不能删**：发布包是把 `build/bin/*`
  整体拷贝出去的，绝对 RPATH 会让 Linux/macOS 包在用户机上找不到
  `libggml*.so`/`.dylib`。Windows 无此问题（先搜 exe 所在目录）。
- **MSVC 必须用 `/EHs`，不能用 CMake 默认的 `/EHsc`**（CMakeLists 顶部有
  `string(REPLACE "/EHsc" "/EHs" ...)`）：`/EHsc` 的 `c` = "假定 `extern "C"`
  的函数不会抛异常"，而 ggml 整个 API 都是 `extern "C"`。于是编译器把
  `init_backend()` 里包着 `ggml_backend_vk_init` 的 try/catch **整个删掉**，
  Vulkan 初始化抛出的 `vk::SystemError` 直接穿到 `main()` —— 结果是进程直接
  死掉，而不是回退 CPU。触发条件很常见：机器上装了 Vulkan loader（Electron
  包自带 `vulkan-1.dll`）但没有可用 ICD。症状是 `error: vk::createInstance:
  ErrorIncompatibleDriver` 且**没有** "vulkan backend init failed" 那行。
  验证方法：`VK_ICD_FILENAMES=C:/nonexistent.json tifa_ggml_cli inspect <模型>`
  应当回退到 CPU 并成功。
- **UI 层**：Electron 顶层脚本里不要 `const bridge = window.bridge`
  （contextBridge 属性不可配置，重复声明直接 SyntaxError）；渲染层 API 挂在
  `window.TifaLabel` 供 e2e 驱动（脚本级 const 对 executeJavaScript 不可见）。
- **LSTM 算子**：`GGML_OP_LSTM`（`cmake/patches/ggml-lstm-op.md`）CPU-only，
  GPU 后端 decline 后调度器自动放回 CPU；新增后端 kernel 前不要改调度假设。
