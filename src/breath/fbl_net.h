#pragma once

// FoxBreatheLabeler (FBL) AP network in ggml.
//
// The published release is a PyTorch checkpoint (see
// scripts/convert_fbl_to_gguf.py), not an ONNX, and the model is a CVNT
// conformer over *raw waveform frames* -- there is no mel/F0 front end, so the
// framing happens here:
//
//   pad(x, (spec_win-hop)//2, (spec_win-hop+1)//2)
//   frame t -> [spec_win] -> inlinear(spec_win -> dim)
//   N x block(dim, kernel, heads x head_dim)
//   final_norm -> outlinear(dim -> 1) -> sigmoid      (AP probability, 50 fps)
//
// block(x):
//   x = ffn1(norm1(x)) * 0.5 + x          ffn: Linear -> SiLU -> Linear
//   x = att(norm2(x)) + x                 multi-head, no RoPE
//   x = conv(norm3(x)) + x                pw1 -> GLU -> depthwise(k) -> BN -> SiLU -> pw2
//   x = ffn2(norm4(x)) * 0.5 + x
//
// Everything is expressed with ggml primitives; one `ggml_context` +
// `ggml_new_graph_custom` + `ggml_gallocr` per frame count, and the input is
// re-uploaded before every compute (AGENT.md §2).  Not part of the public API.

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

// Graph geometry and decoding parameters, read from the GGUF (`fbl.*`).
struct FblConfig {
    int   spec_win    = 1024;      // samples per frame
    int   hop         = 882;       // 20 ms at 44.1 kHz
    int   sample_rate = 44100;
    int   dim         = 512;
    int   n_blocks    = 6;
    int   n_heads     = 8;
    int   head_dim    = 64;
    int   kernel      = 31;        // channel-branch depthwise kernel
    float fps         = 50.0f;
    float threshold   = 0.4f;      // AP probability threshold
    int   min_dur     = 4;         // frames; 0.08 s
    int   max_gap     = 5;         // frames of dip tolerated inside one AP run
    float norm_eps    = 1e-5f;     // PyTorch LayerNorm default
};

class FblNet {
public:
    FblNet();

    static FblNet load(const std::string & gguf_path);

    ~FblNet();
    FblNet(FblNet &&) noexcept;
    FblNet & operator=(FblNet &&) noexcept;
    FblNet(const FblNet &) = delete;
    FblNet & operator=(const FblNet &) = delete;

    const FblConfig & config() const noexcept;

    // One forward pass over `T` frames.
    //   `frames` is [spec_win, T] row-major: frame t is column t, i.e. the
    //   caller has already padded and unfolded the waveform.
    // Returns [T] AP probabilities.
    std::vector<float> run(const float * frames, int T) const;

    const char * backend_name() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tifa_ggml::internal
