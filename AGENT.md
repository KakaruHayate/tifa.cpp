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
