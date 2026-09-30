# TIFA-1.0-ST quantization matrix

Per-tensor dtype study (`scripts/quant_matrix.py`).  Norms, LayerScale,
depthwise-conv kernels and all biases stay F32 in every recipe.


dataset: dataset (first 25 files), backend: vulkan

| recipe | size (MB) | onset MAE (ms) | hit@20ms (%) | ms/file |
|---|---|---|---|---|
| f32 | 158.5 | 5.45 | 93.2 | 4199 |
| f16 | 80.0 | 5.45 | 93.2 | 4272 |
| q8_0-all | 46.3 | 5.47 | 93.2 | 3966 |
| q8_0-attn | 70.2 | 5.45 | 93.2 | 3935 |
| q8_0-ffn | 56.4 | 5.38 | 93.3 | 3836 |
| q4_0-all | 28.3 | 5.39 | 93.3 | 3788 |
| mix-q8attn-q4ffn | 34.0 | 5.49 | 92.6 | 3800 |

## Reading the table

- **F16 is free**: identical accuracy to F32 at half the size — ship it by default.
- **Q8_0 costs nothing measurable**: 46.3 MB (29 % of F32), same boundary error
  (the 0.02 ms delta is noise at this sample count).
- **Q4_0 at 28.3 MB (18 % of F32) still lands within 0.06 ms** of F32 here — a
  good small-download option; re-check on a larger sample before defaulting to it.
- Quantizing only the attention (`q8_0-attn`, 70.2 MB) buys little over F16:
  the FFN + CgMLP weights dominate the parameter count, so the *all* recipes
  are the interesting ones.
- `ms/file` includes process start + model load (one CLI process per file in
  this study), so it is not steady-state latency — see `benchmark-sample.md`.

## Gotcha this study found

`resolve_dtype` originally let a rule override the F32-only roles, which put
F16 RMSNorm weights next to F32 activations: the CPU backend aborts with
`binary_op: unsupported types: dst: f32, src0: f32, src1: f16`, and Vulkan ran
but produced garbage (888 ms boundary MAE instead of 5.4 ms).  F32-only roles
are now checked *before* the rules.
