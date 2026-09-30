#pragma once

// Model configuration populated from a GGUF file's metadata section.
//
// Part of the public API so third-party consumers can introspect a model
// without running inference.

#include <map>
#include <string>
#include <vector>

namespace tifa_ggml {

// Configuration of the JEBFBackbone (modules/backbones/jebf.py) used by
// TIFA's ForcedAlignmentModel.
struct BackboneConfig {
    // Fully-qualified PyTorch class, e.g. "modules.backbones.jebf.JEBFBackbone".
    // Diagnostics only.
    std::string cls;

    int dim        = 256;
    int num_layers = 8;
    int num_heads  = 8;
    int head_dim   = 64;

    // CgMLP kernel sizes (per stream) and depthwise merge kernels.
    int c_kernel_size_token = 7;
    int m_kernel_size_token = 5;
    int c_kernel_size_x     = 31;
    int m_kernel_size_x     = 31;

    std::string attn_type = "joint";   // "joint" | "split"
    std::string ffn_type  = "glu";     // "glu" | "ffn" | "cgmlp" | "eglu"
    bool  qk_norm         = true;
    bool  use_rope        = true;
    bool  use_ls          = true;
    bool  use_out_norm    = true;
    bool  skip_first_ffn  = false;
    bool  skip_out_ffn    = false;
    float theta           = 10000.0f;
};

// Spectrogram front-end configuration (lib/feature/mel.py).
struct FeaturesConfig {
    int   audio_sample_rate = 48000;
    int   hop_size          = 480;
    int   fft_size          = 2048;
    int   win_size          = 2048;
    int   num_bins          = 80;
    float fmin              = 0.0f;
    float fmax              = 8000.0f;
    float clip_val          = 1e-5f;

    // Frame duration in seconds (the TextGrid unit conversion factor).
    float timestep() const noexcept {
        return audio_sample_rate > 0
            ? static_cast<float>(hop_size) / static_cast<float>(audio_sample_rate)
            : 0.0f;
    }
};

// Top-level TIFA model configuration (GGUF metadata `tifa.*`).
struct TifaModelConfig {
    std::string architecture;   // must equal "tifa-fa"
    std::string name;           // human name from the converter
    std::string version;        // schema version ("1", …)
    std::string arch;           // "ForcedAlignmentModel"

    int max_vocab_size = 256;   // token-classifier width (V)
    int in_dim         = 80;    // mel bins
    int embedding_dim  = 256;   // token embedding width
    int out_dim        = 256;   // feature width after the output heads

    BackboneConfig backbone;
    FeaturesConfig features;

    // Inference defaults (mirrors infer.py).
    float       skip_penalty = 0.5f;
    std::string score_unit   = "levenshtein";

    // ---- Vocabulary (vocabulary.json embedded in the GGUF) ----
    std::map<std::string, int> symbol_to_id;
    std::vector<std::string>   id_to_symbol;      // index = id; "<unused>" for gaps
    std::vector<std::string>   global_symbols;    // e.g. AP, EP, GS
    std::vector<std::string>   stop_symbols;      // e.g. SP, sil, pau
    std::vector<std::vector<std::string>> merged_groups;

    // Raw G2P pipeline configuration as JSON text (parsed by the G2P layer).
    std::string g2p_json;
};

}  // namespace tifa_ggml
