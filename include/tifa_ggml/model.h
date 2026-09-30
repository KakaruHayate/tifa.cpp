#pragma once

// High-level Model API.  PIMPL-backed so consumers never include ggml headers.

#include "tifa_ggml/config.h"
#include "tifa_ggml/types.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tifa_ggml {

class Model {
public:
    // Load a TIFA GGUF produced by `scripts/convert_tifa_to_gguf.py`.
    // Initialises the best available backend (Metal → CUDA → Vulkan → CPU).
    // Throws GgufError / BackendError / NotImplemented on failure.
    static Model load(const std::string & gguf_path);

    ~Model();
    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    // ---- inspection ------------------------------------------------------
    const TifaModelConfig & config() const noexcept;

    // Resolve a phone label against the vocabulary (literal first, then
    // "<language>/<symbol>").  Returns -1 when unresolved.
    int resolve_symbol(const std::string & symbol,
                       const std::vector<std::string> & languages) const;

    // ---- inference -------------------------------------------------------

    // Forced alignment of a mono waveform (any sample rate; resampled to the
    // model rate internally) against a known phone sequence.
    //
    // A single Model instance is NOT thread-safe; hold one per thread.
    AlignResult align(const float * waveform, std::size_t n_samples,
                      int sample_rate, const AlignRequest & request);

    // Same, but with the token sequence already resolved (the G2P path):
    // `tokens` are vocabulary ids, `labels` their display labels, `words` and
    // `groups` the 1-based semantic-word / pronunciation-group ids per token.
    AlignResult align_tokens(const float * waveform, std::size_t n_samples,
                             int sample_rate, const AlignTokenRequest & request);

    // Lower-level entry point used by the test suite and the scoring pass:
    // mel → network → cosine similarity [T, N] plus frame/token logits.
    struct Forward {
        int T = 0;                        // valid frames
        int N = 0;                        // valid tokens
        std::vector<float> mel;              // row-major [T, in_dim]
        std::vector<float> similarity;       // row-major [T, N]
        std::vector<float> frame_features;   // row-major [T, out_dim]
        std::vector<float> token_features;   // row-major [N, out_dim]
        std::vector<float> frame_logits;     // row-major [T, V]
        std::vector<float> token_logits;     // row-major [N, V]

        // Per-layer activations (empty unless TIFA_GGML_DUMP_LAYERS=1):
        // entry [i] is the output *after* layer i, entry 0 the input
        // projections, so the vector has num_layers + 1 entries.
        std::vector<std::vector<float>> layer_x;      // each row-major [T, dim]
        std::vector<std::vector<float>> layer_token;  // each row-major [N, dim]

        // Layer-0 sub-block taps (TIFA_GGML_DUMP_LAYERS=1), row-major [·, dim].
        std::vector<float> tap_ffn1_token, tap_ffn1_x;
        std::vector<float> tap_attn_token, tap_attn_x;
        std::vector<float> tap_pjac_token, tap_pjac_x;
        std::vector<float> tap_ffn1_x_norm, tap_ffn1_x_glu;
        std::vector<float> tap_q_token, tap_k_token, tap_v_token;
        std::vector<float> tap_q_x, tap_k_x, tap_v_x;
        std::vector<float> tap_qkv_token, tap_qkv_x;
        std::vector<float> tap_attn_norm_token, tap_attn_norm_x;
        std::vector<float> tap_q_pre, tap_q_normed;
    };
    Forward forward(const float * waveform, std::size_t n_samples, int sample_rate,
                    const std::vector<std::int32_t> & tokens);

    // Internal state (opaque to consumers so the public ABI stays ggml-free).
    struct Impl;

    // Escape hatch for the test suite.  Not part of the stable API.
    Impl & internals() noexcept;

private:
    Model();
    std::unique_ptr<Impl> impl_;
};

}  // namespace tifa_ggml
