#pragma once

// BreathLab breath/AP network in ggml.
//
// The ONNX graph is a small conv-attention net over the 66-row feature matrix:
//
//   front   : Conv2d(1->32,3x3) -> ReLU -> Conv2d(32->32,3x3) -> ReLU -> MaxPool(2,1)
//             x3 stacks over the frequency axis (66 -> 33 -> 16 -> 8 bins),
//             then flatten [C=128, F=8] -> 1024 and project to 192.
//   blocks  : 5 x { residual(GLU-SiLU FFN, x0.5); residual(PAC) } where PAC is
//             attention(QKV + RoPE, 4 heads x 48) concatenated with a CgMLP-style
//             depthwise-conv branch and merged through a depthwise conv.
//   head    : LayerNorm -> Linear(192->4); the 4 channels are AP / SP / onset / offset.
//
// Everything is expressed with ggml primitives; single `ggml_context` +
// `ggml_new_graph_custom` + `ggml_gallocr` per window length, exactly like
// `model_tifa.cpp`, and the input is re-uploaded before every compute (see
// AGENT.md §2).  Not part of the public API.

#include "gguf_io.h"

#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct ggml_gallocr;
struct ggml_backend;

namespace tifa_ggml::internal {

// Graph geometry, read from the GGUF (`breath.model.*`).
struct BreathNetConfig {
    int   dim          = 192;
    int   n_blocks     = 5;
    int   ffn_hidden   = 768;
    int   n_heads      = 4;
    int   head_dim     = 48;
    int   n_out        = 4;
    float rope_theta   = 10000.0f;
    float attn_scale   = 0.0f;      // 1/sqrt(head_dim)
    std::vector<int> front_channels;   // one entry per front conv
    std::vector<int> front_pools;      // conv indices followed by a MaxPool over freq
    int   front_freq_bins = 8;         // frequency bins entering the flatten
    int   input_rows      = 66;
    int   conv_kernel     = 15;        // channel-branch depthwise kernel
    int   merge_kernel    = 31;        // merge depthwise kernel
    int   proj_in         = 1024;
    float norm_eps        = 1e-5f;     // ONNX LayerNormalization epsilon
};

class BreathNet {
public:
    // Empty net; every entry point requires a `load()`ed instance.
    BreathNet();

    static BreathNet load(const std::string & gguf_path);

    ~BreathNet();
    BreathNet(BreathNet &&) noexcept;
    BreathNet & operator=(BreathNet &&) noexcept;
    BreathNet(const BreathNet &) = delete;
    BreathNet & operator=(const BreathNet &) = delete;

    const BreathNetConfig & config() const noexcept;

    // One forward pass over a fixed window of `T` frames.
    //   `rows` is [input_rows, stride] row-major (the layout BreathFeatures
    //   uses); frames [0, T) of that matrix are the window.  `stride` lets a
    //   caller score a sliding window without copying.
    // Returns [n_out, T] (AP / SP / onset / offset probabilities).
    std::vector<float> run(const float * rows, int stride, int T) const;
    // The returned buffer is the raw ggml [n_out, T] tensor: head h of frame t
    // is at index h + n_out * t (ne0 is the innermost axis).

    const char * backend_name() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tifa_ggml::internal
