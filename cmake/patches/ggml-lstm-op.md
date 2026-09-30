# ggml-lstm-op.patch

Fused batched LSTM sweep for ggml v0.19.0 — the first LSTM cell in ggml
core, authored for tifa.cpp's English G2P (LstmG2p) following the fused-RNN
pattern of `GGML_OP_GRU` in [ggml-audio-patch](https://github.com/KakaruHayate/ggml-audio-patch)
(qvac-ops patch 2; the GRU itself is CPU + Vulkan there — this LSTM ships
CPU-only, which is all the G2P path needs: the net is tiny, runs per OOV
word, and the recurrence is inherently sequential).

## What it adds

`GGML_OP_LSTM` + `ggml_lstm()` — the whole recurrent sweep over the L
time-steps as one op, parallel over the batch:

```
struct ggml_tensor * ggml_lstm(ctx, whh, gi_all, bhh, h0, c0, reverse);
```

* `whh` `[H, 4H]` — recurrence weights, gate `g`'s H weights at row `g`
  (the output-major layout the GGUF converter already produces, so no
  transpose at load time).
* `gi_all` `[4H, B, L]` — the *input* projection `W·x + b`, precomputed
  outside the op with plain `mul_mat` + `add` nodes so the heavy matmuls
  stay on whatever backend the graph runs on.
* `bhh` `[4H]` — recurrent bias (added every step, cannot fold into
  `gi_all`).
* `h0` / `c0` `[H, B]` — optional initial state; `NULL` = zeros.  Required
  for the decoder, whose state crosses op invocations, and lets the
  encoder layer-1 pass start from zeros like layer 0.
* `reverse` — sweep time backwards (the bidirectional encoder).
* `dst` `[H, B, L + 2]` — planes `0..L-1` the `h` sequence (time-ordered
  regardless of direction), plane `L` the final `h`, plane `L+1` the final
  `c`.  Final states are needed by the encoder (they average into the
  decoder start state); emitting them as extra planes costs nothing and
  keeps every consumer a plain `ggml_view` — no cont copies.

Gate semantics follow the ONNX LSTM convention the released LstmG2p ONNX
pair is exported with — order `i, o, f, c`:

```
c' = sigmoid(f) * c + sigmoid(i) * tanh(g)
h' = sigmoid(o) * tanh(c')
```

with gate pre-activations `gi + (whh·h + bhh)`.

## Touches

| file | change |
|---|---|
| `include/ggml.h` | `GGML_OP_LSTM` (after `GGML_OP_GLU`), `ggml_lstm()` declaration |
| `src/ggml.c` | op name + symbol tables, `GGML_OP_COUNT` 101 → 102, builder |
| `src/ggml-cpu/ggml-cpu.c` | compute dispatch + `n_tasks = n_threads` (kernel splits the batch) |
| `src/ggml-cpu/ops.cpp` | CPU kernel |
| `src/ggml-cpu/ops.h` | kernel declaration |
| `src/ggml-backend-meta.cpp` | the recurrent sweep must not be split: route to the generic scalar-only split state |

No Vulkan/Metal/CUDA kernel — `ggml_backend_*_supports_op` returns false
for the new op on those backends, so the scheduler places `ggml_lstm` on
the CPU backend automatically; everything around it (matmuls) still runs
on the accelerator.
