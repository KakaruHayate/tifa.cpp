#pragma once

// Internal state of tifa_ggml::Model (PIMPL).  Not part of the public API.

#include "gguf_io.h"
#include "ops_jebf.h"
#include "tensor_utils.h"

#include "tifa_ggml/mel.h"
#include "tifa_ggml/model.h"

#include <memory>
#include <vector>

struct ggml_backend;
struct ggml_context;
struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_backend *  ggml_backend_t;
typedef struct ggml_gallocr *  ggml_gallocr_t;

namespace tifa_ggml {

// Weights of the ForcedAlignmentModel, bound from the GGUF tensor table.
struct TifaWeights {
    ggml_tensor * token_embedding = nullptr;      // (embedding_dim, V)

    ggml_tensor * audio_input_w = nullptr;        // (in_dim, dim)
    ggml_tensor * audio_input_b = nullptr;        // (dim)
    ggml_tensor * text_input_w  = nullptr;        // (embedding_dim, dim)
    ggml_tensor * text_input_b  = nullptr;        // (dim)

    std::vector<internal::ops::JebfBlockWeights> layers;

    ggml_tensor * output_norm_x     = nullptr;    // (dim)
    ggml_tensor * output_norm_token = nullptr;    // (dim)
    ggml_tensor * output_proj_x_w      = nullptr; // (dim, out_dim + V)
    ggml_tensor * output_proj_x_b      = nullptr;
    ggml_tensor * output_proj_token_w  = nullptr;
    ggml_tensor * output_proj_token_b  = nullptr;
};

struct Model::Impl {
    TifaModelConfig cfg;
    std::unique_ptr<internal::GgufFile> gguf;      // metadata handle (kept for diagnostics)
    ggml_backend_t backend = nullptr;
    std::unique_ptr<internal::LoadedWeights> weights;
    std::unique_ptr<MelExtractor> mel;
    TifaWeights w;

    // ---- one persistent graph per (T, N) shape --------------------------
    struct Stage {
        ggml_context * ctx   = nullptr;
        ggml_cgraph *  graph = nullptr;
        ggml_gallocr_t alloc = nullptr;

        ggml_tensor * in_mel    = nullptr;   // (in_dim, T, 1) f32
        ggml_tensor * in_tokens = nullptr;   // (N,) i32
        ggml_tensor * in_n_mask = nullptr;   // (1, N, 1, 1) f32
        ggml_tensor * in_t_mask = nullptr;   // (1, T, 1, 1) f32
        ggml_tensor * token_pos = nullptr;   // (N,) i32 — fixed iota
        ggml_tensor * x_pos     = nullptr;   // (T,) i32 — fixed iota

        ggml_tensor * out_frame_feats  = nullptr;  // (out_dim, T, 1)
        ggml_tensor * out_frame_logits = nullptr;  // (V, T, 1)
        ggml_tensor * out_token_feats  = nullptr;  // (out_dim, N, 1)
        ggml_tensor * out_token_logits = nullptr;  // (V, N, 1)

        // Per-layer debug taps (only populated when TIFA_GGML_DUMP_LAYERS=1).
        std::vector<ggml_tensor *> dbg_x;
        std::vector<ggml_tensor *> dbg_token;
        // Layer-0 sub-block taps (TIFA_GGML_DUMP_LAYERS=1).
        internal::ops::JebfTaps taps0;

        int T = 0;
        int N = 0;

        bool matches(int t, int n) const noexcept { return ctx && T == t && N == n; }
        void reset();
        ~Stage();
    };
    std::unique_ptr<Stage> stage;

    // ---- helpers ---------------------------------------------------------
    // Build (or reuse) the persistent graph for this (T, N) shape.
    Stage & ensure_stage(int T, int N);

    // Load the waveform, mel it, and compute the network forward.
    Model::Forward run_forward(const float * waveform, std::size_t n_samples,
                               int sample_rate,
                               const std::vector<std::int32_t> & tokens);

    // Resolve phone labels to token ids (throws on unknown symbols).
    // Stop symbols (SP/sil/pau) are dropped from the sequence; `labels_out`
    // receives the surviving labels so they stay index-aligned with tokens.
    std::vector<std::int32_t> encode_phones(const std::vector<std::string> & phones,
                                            const std::string & language,
                                            std::vector<std::string> * labels_out = nullptr) const;
};

}  // namespace tifa_ggml
