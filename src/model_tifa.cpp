// TIFA ForcedAlignmentModel: GGUF binding + ggml graph + inference entry
// points.  Mirrors openvpi/TIFA modules/forced_alignment.py and
// modules/backbones/jebf.py.

#include "model_impl.h"

#include "align_decode.h"
#include "metrics.h"
#include "audio_io.h"
#include "backend.h"
#include "ops_basic.h"
#include "ops_jebf.h"

#include "tifa_ggml/errors.h"
#include "tifa_ggml/model.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace tifa_ggml {

using internal::ops::JebfBlockWeights;
using internal::ops::JebfCgmlpWeights;
using internal::ops::JebfFfnWeights;
using internal::ops::JebfMergeWeights;
using internal::ops::JointAttentionWeights;
using internal::ops::PjacWeights;

namespace {

// ---------------------------------------------------------------------------
// Weight binding
// ---------------------------------------------------------------------------

ggml_tensor * need(const internal::LoadedWeights & w, const std::string & name) {
    return w.get(name);   // throws GgufError with the tensor name
}

ggml_tensor * opt(const internal::LoadedWeights & w, const std::string & name) {
    return w.try_get(name);
}

void bind_ffn(const internal::LoadedWeights & w, const std::string & prefix,
              const std::string & norm_name, JebfFfnWeights & f)
{
    f.w_norm  = need(w, norm_name);
    f.w_ln1   = need(w, prefix + ".ln1.weight");
    f.b_ln1   = need(w, prefix + ".ln1.bias");
    f.w_ln1_a = nullptr;  // DEBUG: disable split halves
    f.b_ln1_a = opt (w, prefix + ".ln1.bias.a");
    f.w_ln1_b = opt (w, prefix + ".ln1.weight.b");
    f.b_ln1_b = opt (w, prefix + ".ln1.bias.b");
    f.w_ln2   = need(w, prefix + ".ln2.weight");
    f.b_ln2   = need(w, prefix + ".ln2.bias");
}

void bind_cgmlp(const internal::LoadedWeights & w, const std::string & prefix,
                JebfCgmlpWeights & c, int kernel)
{
    c.w_pw1  = need(w, prefix + ".pw1.weight");
    c.b_pw1  = need(w, prefix + ".pw1.bias");
    c.w_norm = need(w, prefix + ".norm.weight");
    c.w_dw   = need(w, prefix + ".dw.weight");
    c.b_dw   = opt (w, prefix + ".dw.bias");
    c.w_pw2  = need(w, prefix + ".pw2.weight");
    c.b_pw2  = need(w, prefix + ".pw2.bias");
    c.kernel = kernel;
}

void bind_layer(const internal::LoadedWeights & w, const std::string & p,
                const TifaModelConfig & cfg, JebfBlockWeights & L)
{
    const auto & b = cfg.backbone;

    L.has_ffn1 = !b.skip_first_ffn;
    L.has_ffn2 = !b.skip_out_ffn;

    if (L.has_ffn1) {
        bind_ffn(w, p + ".ffn1_x",     p + ".norm_ffn1_x.weight",     L.ffn1_x);
        bind_ffn(w, p + ".ffn1_token", p + ".norm_ffn1_token.weight", L.ffn1_token);
    }
    if (L.has_ffn2) {
        bind_ffn(w, p + ".ffn2_x",     p + ".norm_ffn2_x.weight",     L.ffn2_x);
        bind_ffn(w, p + ".ffn2_token", p + ".norm_ffn2_token.weight", L.ffn2_token);
    }

    JointAttentionWeights & j = L.pjac.jattn;
    j.w_token_norm   = need(w, p + ".attn.jattn.token_norm.weight");
    j.w_x_norm       = need(w, p + ".attn.jattn.x_norm.weight");
    j.w_token_qkv    = need(w, p + ".attn.jattn.token_qkv.weight");
    j.b_token_qkv    = need(w, p + ".attn.jattn.token_qkv.bias");
    j.w_x_qkv        = need(w, p + ".attn.jattn.x_qkv.weight");
    j.b_x_qkv        = need(w, p + ".attn.jattn.x_qkv.bias");
    j.w_token_q_norm = need(w, p + ".attn.jattn.token_q_norm.weight");
    j.w_token_k_norm = need(w, p + ".attn.jattn.token_k_norm.weight");
    j.w_x_q_norm     = need(w, p + ".attn.jattn.x_q_norm.weight");
    j.w_x_k_norm     = need(w, p + ".attn.jattn.x_k_norm.weight");
    j.w_token_out    = need(w, p + ".attn.jattn.token_out.weight");
    j.b_token_out    = need(w, p + ".attn.jattn.token_out.bias");
    j.w_x_out        = need(w, p + ".attn.jattn.x_out.weight");
    j.b_x_out        = need(w, p + ".attn.jattn.x_out.bias");

    L.pjac.w_c_norm_token = need(w, p + ".attn.c_norm_token.weight");
    L.pjac.w_c_norm_x     = need(w, p + ".attn.c_norm_x.weight");
    bind_cgmlp(w, p + ".attn.c_token", L.pjac.c_token, b.c_kernel_size_token);
    bind_cgmlp(w, p + ".attn.c_x",     L.pjac.c_x,     b.c_kernel_size_x);

    L.pjac.merge_token.w_linear = need(w, p + ".attn.merge_linear_token.weight");
    L.pjac.merge_token.b_linear = need(w, p + ".attn.merge_linear_token.bias");
    L.pjac.merge_token.w_dw     = opt (w, p + ".attn.merge_dw_conv_token.weight");
    L.pjac.merge_token.b_dw     = opt (w, p + ".attn.merge_dw_conv_token.bias");
    L.pjac.merge_token.kernel   = L.pjac.merge_token.w_dw ? b.m_kernel_size_token : 0;

    L.pjac.merge_x.w_linear = need(w, p + ".attn.merge_linear_x.weight");
    L.pjac.merge_x.b_linear = need(w, p + ".attn.merge_linear_x.bias");
    L.pjac.merge_x.w_dw     = opt (w, p + ".attn.merge_dw_conv_x.weight");
    L.pjac.merge_x.b_dw     = opt (w, p + ".attn.merge_dw_conv_x.bias");
    L.pjac.merge_x.kernel   = L.pjac.merge_x.w_dw ? b.m_kernel_size_x : 0;

    if (b.use_ls) {
        if (L.has_ffn1) {
            L.ffn1_x.w_lay_scale     = need(w, p + ".layer_scale_ffn1_x.scale");
            L.ffn1_token.w_lay_scale = need(w, p + ".layer_scale_ffn1_token.scale");
        }
        L.pjac.w_lay_scale_token = need(w, p + ".layer_scale_jpac_token.scale");
        L.pjac.w_lay_scale_x     = need(w, p + ".layer_scale_jpac_x.scale");
        if (L.has_ffn2) {
            L.ffn2_x.w_lay_scale     = need(w, p + ".layer_scale_ffn2_x.scale");
            L.ffn2_token.w_lay_scale = need(w, p + ".layer_scale_ffn2_token.scale");
        }
    }
}

TifaWeights bind_weights(const internal::LoadedWeights & w, const TifaModelConfig & cfg) {
    TifaWeights t;
    t.token_embedding = need(w, "token_embedding.weight");

    t.audio_input_w = need(w, "backbone.audio_input_proj.weight");
    t.audio_input_b = need(w, "backbone.audio_input_proj.bias");
    t.text_input_w  = need(w, "backbone.text_input_proj.weight");
    t.text_input_b  = need(w, "backbone.text_input_proj.bias");

    t.layers.resize(static_cast<std::size_t>(cfg.backbone.num_layers));
    for (int i = 0; i < cfg.backbone.num_layers; ++i) {
        bind_layer(w, "backbone.layers." + std::to_string(i), cfg,
                   t.layers[static_cast<std::size_t>(i)]);
    }

    if (cfg.backbone.use_out_norm) {
        t.output_norm_x     = need(w, "backbone.output_norm_x.weight");
        t.output_norm_token = need(w, "backbone.output_norm_token.weight");
    }
    t.output_proj_x_w     = need(w, "backbone.output_proj_x.weight");
    t.output_proj_x_b     = need(w, "backbone.output_proj_x.bias");
    t.output_proj_token_w = need(w, "backbone.output_proj_token.weight");
    t.output_proj_token_b = need(w, "backbone.output_proj_token.bias");

    return t;
}

}  // namespace

// ---------------------------------------------------------------------------
// Stage lifecycle
// ---------------------------------------------------------------------------

void Model::Impl::Stage::reset() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    graph = nullptr;
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    in_mel = in_tokens = in_n_mask = in_t_mask = nullptr;
    out_frame_feats = out_frame_logits = out_token_feats = out_token_logits = nullptr;
    T = N = 0;
}

Model::Impl::Stage::~Stage() { reset(); }

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;

const TifaModelConfig & Model::config() const noexcept { return impl_->cfg; }
Model::Impl & Model::internals() noexcept { return *impl_; }

Model Model::load(const std::string & gguf_path) {
    Model m;
    auto & impl = *m.impl_;

    impl.gguf = std::make_unique<internal::GgufFile>(internal::GgufFile::open(gguf_path));
    impl.cfg  = internal::load_config(*impl.gguf);

    if (impl.cfg.arch != "ForcedAlignmentModel") {
        throw NotImplemented("TIFA architecture '" + impl.cfg.arch + "' is not supported");
    }
    const auto & b = impl.cfg.backbone;
    if (b.attn_type != "joint") {
        throw NotImplemented("attn_type '" + b.attn_type + "' is not implemented (only 'joint')");
    }
    if (b.ffn_type != "glu") {
        throw NotImplemented("ffn_type '" + b.ffn_type + "' is not implemented (only 'glu')");
    }
    if (!b.qk_norm || !b.use_rope) {
        throw NotImplemented("this port requires qk_norm=true and use_rope=true");
    }

    impl.backend = internal::init_best_backend();

    // Depthwise convs (the CgMLP branch of every PAC block): use the dedicated
    // per-channel GGML_OP_CONV_2D_DW kernel (no im2col) on every backend that
    // implements it — CPU, Vulkan, Metal and CUDA all accept it, and our
    // direct path casts the stored depthwise weight to the F32 the kernel
    // reads.  Other/unknown backends fall back to ggml_conv_1d_dw
    // (im2col + F16).  Same capability gate as game.cpp's model loader;
    // TIFA_GGML_DWCONV=legacy|direct overrides per process.
    {
        std::string bn = internal::backend_name(impl.backend);
        std::transform(bn.begin(), bn.end(), bn.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const bool direct_ok = bn.find("cpu")    != std::string::npos ||
                               bn.find("vulkan") != std::string::npos ||
                               bn.find("cuda")   != std::string::npos ||
                               bn.find("metal")  != std::string::npos;
        internal::ops::set_direct_dwconv(direct_ok);
    }

    impl.weights = std::make_unique<internal::LoadedWeights>(
        internal::LoadedWeights::load_all(*impl.gguf, impl.backend));
    impl.w = bind_weights(*impl.weights, impl.cfg);

    MelConfig mc;
    mc.sample_rate = impl.cfg.features.audio_sample_rate;
    mc.n_fft       = impl.cfg.features.fft_size;
    mc.win_length  = impl.cfg.features.win_size;
    mc.hop_length  = impl.cfg.features.hop_size;
    mc.n_mels      = impl.cfg.features.num_bins;
    mc.fmin        = impl.cfg.features.fmin;
    mc.fmax        = impl.cfg.features.fmax;
    mc.clip_val    = impl.cfg.features.clip_val;
    impl.mel = std::make_unique<MelExtractor>(mc);

    return m;
}

int Model::resolve_symbol(const std::string & symbol,
                          const std::vector<std::string> & languages) const
{
    const auto & cfg = impl_->cfg;
    const bool global = std::find(cfg.global_symbols.begin(), cfg.global_symbols.end(), symbol)
                        != cfg.global_symbols.end();
    auto lookup = [&](const std::string & key) -> int {
        const auto it = cfg.symbol_to_id.find(key);
        return it == cfg.symbol_to_id.end() ? -1 : it->second;
    };
    if (global) return lookup(symbol);
    for (const auto & lang : languages) {
        const int id = lookup(lang + "/" + symbol);
        if (id >= 0) return id;
    }
    return lookup(symbol);   // literal (already qualified) form
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

namespace {

constexpr std::size_t kGraphCtxBytes = 64ull * 1024 * 1024;
constexpr int         kGraphNodes    = 65536;

}  // namespace

Model::Impl::Stage & Model::Impl::ensure_stage(int T, int N) {
    if (stage && stage->matches(T, N)) return *stage;

    stage = std::make_unique<Stage>();

    ggml_init_params ip{};
    ip.mem_size = kGraphCtxBytes;
    ip.no_alloc = true;
    stage->ctx  = ggml_init(ip);
    if (!stage->ctx) throw Error("ggml_init failed (graph context)");
    stage->graph = ggml_new_graph_custom(stage->ctx, kGraphNodes, /*grads=*/false);

    auto * ctx = stage->ctx;
    const auto & cfg = this->cfg;
    const auto & W   = this->w;
    const int V      = cfg.max_vocab_size;

    // ---- inputs ----------------------------------------------------------
    stage->in_mel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cfg.in_dim, T, 1);
    ggml_set_input(stage->in_mel);

    stage->in_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_input(stage->in_tokens);

    stage->in_n_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, N, 1, 1);
    ggml_set_input(stage->in_n_mask);
    stage->in_t_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T, 1, 1);
    ggml_set_input(stage->in_t_mask);

    // Position tensors are fixed iota sequences — written once after the
    // arena is allocated (before that they have no backing buffer).
    ggml_tensor * token_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor * x_pos     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(token_pos);
    ggml_set_input(x_pos);
    stage->token_pos = token_pos;
    stage->x_pos     = x_pos;

    // ---- forward ---------------------------------------------------------
    ggml_tensor * x     = internal::ops::linear(ctx, stage->in_mel, W.audio_input_w, W.audio_input_b);
    ggml_tensor * token = internal::ops::embedding(ctx, W.token_embedding, stage->in_tokens);
    token = internal::ops::linear(ctx, token, W.text_input_w, W.text_input_b);

    const bool dump_layers = std::getenv("TIFA_GGML_DUMP_LAYERS") != nullptr;
    if (dump_layers) {
        stage->dbg_x.push_back(x);
        stage->dbg_token.push_back(token);
    }
    for (std::size_t li = 0; li < W.layers.size(); ++li) {
        const auto & layer = W.layers[li];
        auto out = internal::ops::jebf_block(ctx, token, x, layer,
                                             token_pos, x_pos,
                                             cfg.backbone.num_heads, cfg.backbone.head_dim,
                                             cfg.backbone.theta,
                                             nullptr, nullptr,
                                             (dump_layers && li == 0) ? &stage->taps0 : nullptr);
        token = out.token;
        x     = out.x;
        if (dump_layers) {
            stage->dbg_x.push_back(x);
            stage->dbg_token.push_back(token);
        }
    }

    if (cfg.backbone.use_out_norm) {
        x     = internal::ops::rms_norm(ctx, x,     W.output_norm_x,     1e-6f);
        token = internal::ops::rms_norm(ctx, token, W.output_norm_token, 1e-6f);
    }

    ggml_tensor * frame_out = internal::ops::linear(ctx, x,     W.output_proj_x_w,     W.output_proj_x_b);
    ggml_tensor * token_out = internal::ops::linear(ctx, token, W.output_proj_token_w, W.output_proj_token_b);

    // Split [features (out_dim) | logits (V)] along ne0.
    auto split_head = [&](ggml_tensor * full, int n_tok,
                          ggml_tensor ** feats, ggml_tensor ** logits) {
        const std::size_t esize = ggml_element_size(full);
        *feats  = ggml_cont(ctx, ggml_view_3d(ctx, full, cfg.out_dim, n_tok, 1,
                                              full->nb[1], full->nb[2], 0));
        *logits = ggml_cont(ctx, ggml_view_3d(ctx, full, V, n_tok, 1,
                                              full->nb[1], full->nb[2],
                                              static_cast<std::size_t>(cfg.out_dim) * esize));
        ggml_set_output(*feats);
        ggml_set_output(*logits);
    };
    split_head(frame_out, T, &stage->out_frame_feats, &stage->out_frame_logits);
    split_head(token_out, N, &stage->out_token_feats, &stage->out_token_logits);

    // Sub-block taps must be marked *before* the allocator runs, otherwise
    // their memory is reused by later nodes and the readback is garbage.
    if (dump_layers) {
        for (ggml_tensor * t : {stage->taps0.ffn1_token, stage->taps0.ffn1_x,
                                stage->taps0.attn_token, stage->taps0.attn_x,
                                stage->taps0.pjac_token, stage->taps0.pjac_x,
                                stage->taps0.ffn1_x_norm, stage->taps0.ffn1_x_glu}) {
            if (t) ggml_set_output(t);
        }
    }

    // Debug taps are copied into dedicated output tensors before the graph is
    // built: reading an ordinary intermediate back after compute() returns
    // memory the allocator may already have reused.
    if (dump_layers) {
        auto copy_out = [&](ggml_tensor * t) -> ggml_tensor * {
            if (!t) return nullptr;
            ggml_tensor * dst = ggml_dup_tensor(ctx, t);
            ggml_tensor * cpy = ggml_cpy(ctx, t, dst);
            ggml_set_output(cpy);
            ggml_build_forward_expand(stage->graph, cpy);
            return cpy;
        };
        stage->taps0.ffn1_token = copy_out(stage->taps0.ffn1_token);
        stage->taps0.ffn1_x     = copy_out(stage->taps0.ffn1_x);
        stage->taps0.attn_token = copy_out(stage->taps0.attn_token);
        stage->taps0.attn_x     = copy_out(stage->taps0.attn_x);
        stage->taps0.pjac_token = copy_out(stage->taps0.pjac_token);
        stage->taps0.pjac_x     = copy_out(stage->taps0.pjac_x);
        stage->taps0.ffn1_x_norm = copy_out(stage->taps0.ffn1_x_norm);
        stage->taps0.ffn1_x_glu  = copy_out(stage->taps0.ffn1_x_glu);
        stage->taps0.attn_q_token = copy_out(stage->taps0.attn_q_token);
        stage->taps0.attn_k_token = copy_out(stage->taps0.attn_k_token);
        stage->taps0.attn_v_token = copy_out(stage->taps0.attn_v_token);
        stage->taps0.attn_q_x     = copy_out(stage->taps0.attn_q_x);
        stage->taps0.attn_k_x     = copy_out(stage->taps0.attn_k_x);
        stage->taps0.attn_v_x     = copy_out(stage->taps0.attn_v_x);
        stage->taps0.attn_qkv_token = copy_out(stage->taps0.attn_qkv_token);
        stage->taps0.attn_qkv_x     = copy_out(stage->taps0.attn_qkv_x);
        stage->taps0.attn_norm_token = copy_out(stage->taps0.attn_norm_token);
        stage->taps0.attn_norm_x     = copy_out(stage->taps0.attn_norm_x);
        stage->taps0.attn_q_token_pre    = copy_out(stage->taps0.attn_q_token_pre);
        stage->taps0.attn_q_token_normed = copy_out(stage->taps0.attn_q_token_normed);
        for (std::size_t i = 0; i < stage->dbg_x.size(); ++i) {
            stage->dbg_x[i]     = copy_out(stage->dbg_x[i]);
            stage->dbg_token[i] = copy_out(stage->dbg_token[i]);
        }
    }

    ggml_build_forward_expand(stage->graph, stage->out_frame_feats);
    ggml_build_forward_expand(stage->graph, stage->out_frame_logits);
    ggml_build_forward_expand(stage->graph, stage->out_token_feats);
    ggml_build_forward_expand(stage->graph, stage->out_token_logits);

    stage->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(stage->alloc, stage->graph)) {
        throw Error("ggml_gallocr_alloc_graph failed (T=" + std::to_string(T) +
                    ", N=" + std::to_string(N) + ")");
    }

    // Now that the arena exists, fill the fixed position tensors.
    {
        std::vector<std::int32_t> p(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) p[static_cast<std::size_t>(i)] = i;
        ggml_backend_tensor_set(token_pos, p.data(), 0, p.size() * sizeof(std::int32_t));
        p.resize(static_cast<std::size_t>(T));
        for (int i = 0; i < T; ++i) p[static_cast<std::size_t>(i)] = i;
        ggml_backend_tensor_set(x_pos, p.data(), 0, p.size() * sizeof(std::int32_t));
    }

    stage->T = T;
    stage->N = N;
    return *stage;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

Model::Forward Model::Impl::run_forward(const float * waveform, std::size_t n_samples,
                                        int sample_rate,
                                        const std::vector<std::int32_t> & tokens)
{
    if (!waveform || n_samples == 0) throw InvalidArgument("empty waveform");
    if (tokens.empty()) throw InvalidArgument("empty token sequence");

    const int target_rate = cfg.features.audio_sample_rate;

    // 1. resample when the input rate differs from the model rate.
    std::vector<float> mono;
    if (sample_rate != target_rate) {
        mono.assign(waveform, waveform + n_samples);
        mono = internal::resample_to(mono, sample_rate, target_rate);
    } else {
        mono.assign(waveform, waveform + n_samples);
    }

    // 2. log-mel [T, n_mels].
    const int T = mel->num_frames(mono.size());
    if (T <= 0) throw InvalidArgument("audio too short for one mel frame");
    std::vector<float> melbuf = mel->forward(mono.data(), mono.size());
    const int N = static_cast<int>(tokens.size());

    Stage & st = ensure_stage(T, N);

    // Re-upload the position iotas on every run: they are graph leaves whose
    // backing buffer is owned by the arena, and re-writing is cheap insurance
    // against any arena reuse between runs.
    {
        std::vector<std::int32_t> p(static_cast<std::size_t>(N));
        const bool zero = std::getenv("TIFA_GGML_POS_TEST") != nullptr;
        for (int i = 0; i < N; ++i) p[static_cast<std::size_t>(i)] = zero ? 0 : i;
        ggml_backend_tensor_set(st.token_pos, p.data(), 0, p.size() * sizeof(std::int32_t));
        p.resize(static_cast<std::size_t>(T));
        for (int i = 0; i < T; ++i) p[static_cast<std::size_t>(i)] = zero ? 0 : i;
        ggml_backend_tensor_set(st.x_pos, p.data(), 0, p.size() * sizeof(std::int32_t));
    }

    // 3. upload inputs.  Batch size is 1 and the audio is used at its exact
    //    length, so every frame/token is valid (the padding masks are ones).
    ggml_backend_tensor_set(st.in_mel, melbuf.data(), 0,
                            static_cast<std::size_t>(T) * cfg.features.num_bins * sizeof(float));
    ggml_backend_tensor_set(st.in_tokens, tokens.data(), 0,
                            static_cast<std::size_t>(N) * sizeof(std::int32_t));
    if (st.in_n_mask && st.in_n_mask->buffer && st.in_t_mask && st.in_t_mask->buffer) {
        static thread_local std::vector<float> ones;
        ones.assign(static_cast<std::size_t>(std::max(T, N)), 1.0f);
        ggml_backend_tensor_set(st.in_n_mask, ones.data(), 0,
                                static_cast<std::size_t>(N) * sizeof(float));
        ggml_backend_tensor_set(st.in_t_mask, ones.data(), 0,
                                static_cast<std::size_t>(T) * sizeof(float));
    }

    // 4. compute.
    if (ggml_backend_graph_compute(backend, st.graph) != GGML_STATUS_SUCCESS) {
        throw Error("TIFA graph compute failed");
    }

    // 5. read back.
    Forward f;
    f.T = T;
    f.N = N;
    f.mel = melbuf;
    f.frame_features.resize(static_cast<std::size_t>(T) * cfg.out_dim);
    f.token_features.resize(static_cast<std::size_t>(N) * cfg.out_dim);
    f.frame_logits.resize(static_cast<std::size_t>(T) * cfg.max_vocab_size);
    f.token_logits.resize(static_cast<std::size_t>(N) * cfg.max_vocab_size);

    // ggml tensors are (dim, n, 1) row-major with dim innermost == PyTorch
    // [n, dim] row-major — a straight linear copy.
    ggml_backend_tensor_get(st.out_frame_feats, f.frame_features.data(), 0,
                            f.frame_features.size() * sizeof(float));
    ggml_backend_tensor_get(st.out_token_feats, f.token_features.data(), 0,
                            f.token_features.size() * sizeof(float));
    ggml_backend_tensor_get(st.out_frame_logits, f.frame_logits.data(), 0,
                            f.frame_logits.size() * sizeof(float));
    ggml_backend_tensor_get(st.out_token_logits, f.token_logits.data(), 0,
                            f.token_logits.size() * sizeof(float));

    if (!st.dbg_x.empty()) {
        const std::size_t dim = static_cast<std::size_t>(cfg.backbone.dim);
        f.layer_x.resize(st.dbg_x.size());
        f.layer_token.resize(st.dbg_token.size());
        for (std::size_t i = 0; i < st.dbg_x.size(); ++i) {
            f.layer_x[i].resize(static_cast<std::size_t>(T) * dim);
            f.layer_token[i].resize(static_cast<std::size_t>(N) * dim);
            ggml_backend_tensor_get(st.dbg_x[i], f.layer_x[i].data(), 0,
                                    f.layer_x[i].size() * sizeof(float));
            ggml_backend_tensor_get(st.dbg_token[i], f.layer_token[i].data(), 0,
                                    f.layer_token[i].size() * sizeof(float));
        }
        auto pull = [&](ggml_tensor * t, int n, std::vector<float> & dst, std::size_t cols = 0) {
            if (!t) return;
            if (cols == 0) cols = dim;
            dst.resize(static_cast<std::size_t>(n) * cols);
            ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
        };
        pull(st.taps0.ffn1_token, N, f.tap_ffn1_token);
        pull(st.taps0.ffn1_x,     T, f.tap_ffn1_x);
        pull(st.taps0.attn_token, N, f.tap_attn_token);
        pull(st.taps0.attn_x,     T, f.tap_attn_x);
        pull(st.taps0.pjac_token, N, f.tap_pjac_token);
        pull(st.taps0.pjac_x,     T, f.tap_pjac_x);
        pull(st.taps0.ffn1_x_norm, T, f.tap_ffn1_x_norm);
        pull(st.taps0.ffn1_x_glu,  T, f.tap_ffn1_x_glu);
        const std::size_t hd = static_cast<std::size_t>(cfg.backbone.num_heads) * cfg.backbone.head_dim;
        pull(st.taps0.attn_q_token_pre, N, f.tap_q_pre, hd);
        pull(st.taps0.attn_q_token_normed, N, f.tap_q_normed, hd);
        pull(st.taps0.attn_q_token, N, f.tap_q_token, hd);
        pull(st.taps0.attn_k_token, N, f.tap_k_token, hd);
        pull(st.taps0.attn_v_token, N, f.tap_v_token, hd);
        pull(st.taps0.attn_q_x,     T, f.tap_q_x, hd);
        pull(st.taps0.attn_k_x,     T, f.tap_k_x, hd);
        pull(st.taps0.attn_v_x,     T, f.tap_v_x, hd);
        pull(st.taps0.attn_qkv_token, N, f.tap_qkv_token, hd * 3);
        pull(st.taps0.attn_qkv_x,     T, f.tap_qkv_x, hd * 3);
        pull(st.taps0.attn_norm_token, N, f.tap_attn_norm_token);
        pull(st.taps0.attn_norm_x,     T, f.tap_attn_norm_x);
    }

    // 6. cosine similarity [T, N] (modules/functional.py cross_cosine_similarity).
    f.similarity.assign(static_cast<std::size_t>(T) * N, 0.0f);
    const float eps = 1e-8f;
    for (int t = 0; t < T; ++t) {
        const float * xf = f.frame_features.data() + static_cast<std::size_t>(t) * cfg.out_dim;
        float xn = 0.0f;
        for (int d = 0; d < cfg.out_dim; ++d) xn += xf[d] * xf[d];
        xn = 1.0f / std::sqrt(std::max(xn, eps * eps));
        for (int i = 0; i < N; ++i) {
            const float * tf = f.token_features.data() + static_cast<std::size_t>(i) * cfg.out_dim;
            float dot = 0.0f, tn = 0.0f;
            for (int d = 0; d < cfg.out_dim; ++d) {
                dot += xf[d] * tf[d];
                tn  += tf[d] * tf[d];
            }
            const float inv = 1.0f / std::sqrt(std::max(tn, eps * eps));
            f.similarity[static_cast<std::size_t>(t) * N + i] = dot * xn * inv;
        }
    }
    return f;
}

Model::Forward Model::forward(const float * waveform, std::size_t n_samples, int sample_rate,
                              const std::vector<std::int32_t> & tokens)
{
    return impl_->run_forward(waveform, n_samples, sample_rate, tokens);
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

std::vector<std::int32_t> Model::Impl::encode_phones(
    const std::vector<std::string> & phones, const std::string & language,
    std::vector<std::string> * labels_out) const
{
    if (labels_out) labels_out->clear();
    std::vector<std::string> langs;
    if (!language.empty()) langs.push_back(language);

    std::vector<std::int32_t> tokens;
    tokens.reserve(phones.size());
    for (const auto & ph : phones) {
        const bool global = std::find(cfg.global_symbols.begin(), cfg.global_symbols.end(), ph)
                            != cfg.global_symbols.end();
        const bool stop = std::find(cfg.stop_symbols.begin(), cfg.stop_symbols.end(), ph)
                          != cfg.stop_symbols.end();
        if (stop) continue;   // stop symbols are dropped from the sequence

        int id = -1;
        if (global) {
            const auto it = cfg.symbol_to_id.find(ph);
            if (it != cfg.symbol_to_id.end()) id = it->second;
        } else {
            for (const auto & lang : langs) {
                const auto it = cfg.symbol_to_id.find(lang + "/" + ph);
                if (it != cfg.symbol_to_id.end()) { id = it->second; break; }
            }
            if (id < 0) {
                const auto it = cfg.symbol_to_id.find(ph);   // literal / pre-qualified
                if (it != cfg.symbol_to_id.end()) id = it->second;
            }
        }
        if (id < 0) {
            throw InvalidArgument("phone '" + ph + "' is not in the model vocabulary");
        }
        tokens.push_back(id);
        if (labels_out) labels_out->push_back(ph);
    }
    if (tokens.empty()) throw InvalidArgument("no usable phones after filtering stop symbols");
    return tokens;
}

AlignResult Model::align(const float * waveform, std::size_t n_samples,
                         int sample_rate, const AlignRequest & request)
{
    AlignTokenRequest tokens;
    tokens.tokens        = impl_->encode_phones(request.phones, request.language, &tokens.labels);
    tokens.language      = request.language;
    tokens.skip_penalty  = request.skip_penalty;
    tokens.skip_handling = request.skip_handling;
    tokens.score_unit    = request.score_unit;
    tokens.words.assign(tokens.tokens.size(), 1);
    tokens.groups.resize(tokens.tokens.size());
    for (std::size_t i = 0; i < tokens.groups.size(); ++i) {
        tokens.groups[i] = static_cast<std::int32_t>(i + 1);
    }
    tokens.word_texts.assign(1, request.word_text);
    tokens.group_scripts = tokens.labels;
    return align_tokens(waveform, n_samples, sample_rate, tokens);
}

AlignResult Model::align_tokens(const float * waveform, std::size_t n_samples,
                                int sample_rate, const AlignTokenRequest & request)
{
    auto & impl = *impl_;
    if (request.tokens.empty()) throw InvalidArgument("empty token sequence");

    const std::vector<std::int32_t> & tokens = request.tokens;
    Model::Forward f = impl.run_forward(waveform, n_samples, sample_rate, tokens);

    std::vector<std::int32_t> groups = request.groups;
    if (groups.empty()) {
        groups.resize(static_cast<std::size_t>(f.N));
        for (int i = 0; i < f.N; ++i) groups[static_cast<std::size_t>(i)] = i + 1;
    } else if (groups.size() != static_cast<std::size_t>(f.N)) {
        throw InvalidArgument("group count does not match the token count");
    }

    const auto frame_spans = internal::decode_alignment_flat(
        f.similarity.data(), f.T, f.N, groups.data(), request.skip_penalty);

    AlignResult out;
    out.num_frames = f.T;
    const float timestep = impl.cfg.features.timestep();
    out.labels = request.labels.size() == static_cast<std::size_t>(f.N)
        ? request.labels
        : std::vector<std::string>(static_cast<std::size_t>(f.N), "");
    out.groups = groups;
    out.words = request.words.size() == static_cast<std::size_t>(f.N)
        ? request.words
        : std::vector<std::int32_t>(static_cast<std::size_t>(f.N), 1);

    // Language prefix stripping (matches SaveTextGridCallback).
    if (!request.language.empty()) {
        const std::string prefix = request.language + "/";
        for (auto & lab : out.labels) {
            if (lab.size() > prefix.size() && lab.compare(0, prefix.size(), prefix) == 0) {
                lab = lab.substr(prefix.size());
            }
        }
    }

    // Tier labels (word texts / pronunciation-group scripts).
    out.word_texts    = request.word_texts;
    out.group_scripts = request.group_scripts.size() == out.labels.size()
        ? request.group_scripts
        : out.labels;

    out.spans.resize(static_cast<std::size_t>(f.N));
    for (std::size_t i = 0; i < frame_spans.size(); ++i) {
        out.spans[i].onset  = static_cast<float>(frame_spans[i].first)  * timestep;
        out.spans[i].offset = static_cast<float>(frame_spans[i].second) * timestep;
    }

    // Reference-free diagnostics (confidence / determinacy / monotonicity).
    {
        std::vector<char> t_mask(static_cast<std::size_t>(f.T), 1);
        std::vector<char> n_mask(static_cast<std::size_t>(f.N), 1);
        const auto diag = internal::compute_diagnosis(f.similarity.data(), f.T, f.N,
                                                      frame_spans, t_mask, n_mask);
        out.confidence   = diag.confidence;
        out.determinacy  = diag.determinacy;
        out.monotonicity = diag.monotonicity;
    }

    // agreement: mean softmax probability of the input tokens.
    {
        double acc = 0.0;
        for (int i = 0; i < f.N; ++i) {
            const float * lg = f.token_logits.data() + static_cast<std::size_t>(i) * impl.cfg.max_vocab_size;
            float mx = -std::numeric_limits<float>::infinity();
            for (int v = 0; v < impl.cfg.max_vocab_size; ++v) mx = std::max(mx, lg[v]);
            double sum = 0.0;
            for (int v = 0; v < impl.cfg.max_vocab_size; ++v) sum += std::exp(lg[v] - mx);
            acc += std::exp(lg[tokens[static_cast<std::size_t>(i)]] - mx) / sum;
        }
        out.agreement = static_cast<float>(acc / f.N);
    }

    return out;
}

}  // namespace tifa_ggml
