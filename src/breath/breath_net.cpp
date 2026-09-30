// BreathLab network graph — see breath_net.h for the block structure.

#include "breath_net.h"

#include "backend.h"
#include "ops_basic.h"
#include "ops_rope.h"
#include "tensor_utils.h"

#include "tifa_ggml/errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace tifa_ggml::internal {

namespace {

constexpr std::size_t kGraphCtxBytes = 64ull * 1024 * 1024;
constexpr int         kGraphNodes    = 65536;
constexpr int         kMaxStages     = 4;    // distinct window lengths to keep

// ---------------------------------------------------------------------------
// weight binding
// ---------------------------------------------------------------------------

struct LinearW { ggml_tensor * w = nullptr; ggml_tensor * b = nullptr; };
struct NormW   { ggml_tensor * w = nullptr; ggml_tensor * b = nullptr; };
using ConvW = LinearW;   // conv kernels bind the same (weight, bias) pair

struct BlockW {
    NormW   ffn1_norm, ffn2_norm, ln, ln_out;
    LinearW ffn1_fc1, ffn1_fc2, ffn2_fc1, ffn2_fc2;
    LinearW qkv, attn_out;
    LinearW conv_pw1, conv_pw2;
    LinearW merge_proj;
    ConvW   conv_dw, merge_dw;
};

struct Weights {
    std::vector<ConvW>  front;
    LinearW             proj, head;
    std::vector<BlockW> blocks;
};

ggml_tensor * need(const LoadedWeights & w, const std::string & name) {
    return w.get(name);
}

void bind_linear(const LoadedWeights & w, const std::string & p, LinearW & l) {
    l.w = need(w, p + ".weight");
    l.b = need(w, p + ".bias");
}

void bind_norm(const LoadedWeights & w, const std::string & p, NormW & n) {
    n.w = need(w, p + ".weight");
    n.b = need(w, p + ".bias");
}

Weights bind_weights(const LoadedWeights & w, const BreathNetConfig & cfg) {
    Weights t;
    for (int i = 0; i < static_cast<int>(cfg.front_channels.size()); ++i) {
        ConvW c;
        bind_linear(w, "front.conv." + std::to_string(i), c);
        t.front.push_back(c);
    }
    bind_linear(w, "proj", t.proj);
    bind_linear(w, "head", t.head);

    t.blocks.resize(static_cast<std::size_t>(cfg.n_blocks));
    for (int i = 0; i < cfg.n_blocks; ++i) {
        const std::string p = "blocks." + std::to_string(i);
        BlockW & L = t.blocks[static_cast<std::size_t>(i)];
        bind_norm  (w, p + ".ffn1.norm", L.ffn1_norm);
        bind_norm  (w, p + ".ffn2.norm", L.ffn2_norm);
        bind_norm  (w, p + ".ln",       L.ln);
        bind_norm  (w, p + ".ln_out",   L.ln_out);
        bind_linear(w, p + ".ffn1.fc1", L.ffn1_fc1);
        bind_linear(w, p + ".ffn1.fc2", L.ffn1_fc2);
        bind_linear(w, p + ".ffn2.fc1", L.ffn2_fc1);
        bind_linear(w, p + ".ffn2.fc2", L.ffn2_fc2);
        bind_linear(w, p + ".attn.qkv", L.qkv);
        bind_linear(w, p + ".attn.out", L.attn_out);
        bind_linear(w, p + ".conv.pw1", L.conv_pw1);
        bind_linear(w, p + ".conv.pw2", L.conv_pw2);
        bind_linear(w, p + ".merge.proj", L.merge_proj);
        bind_linear(w, p + ".conv.dw",  L.conv_dw);    // [K, 1, C] kernel + [C] bias
        bind_linear(w, p + ".merge.dw", L.merge_dw);   // [K, 1, C] kernel + [C] bias
    }
    return t;
}

// ---------------------------------------------------------------------------
// graph helpers
// ---------------------------------------------------------------------------

// ggml_add aborts the process on an invalid broadcast; check first so a shape
// bug surfaces as a catchable error naming the two operands.
ggml_tensor * add(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (!ggml_can_repeat(b, a)) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "breath: add shape mismatch: a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]",
            (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2],
            (long long) a->ne[3], (long long) b->ne[0], (long long) b->ne[1],
            (long long) b->ne[2], (long long) b->ne[3]);
        throw Error(msg);
    }
    return ggml_add(ctx, a, b);
}

ggml_tensor * mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                      const char * what) {
    if (a->ne[0] != b->ne[0] || b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "breath: mul_mat(%s) a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]",
            what, (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2],
            (long long) a->ne[3], (long long) b->ne[0], (long long) b->ne[1],
            (long long) b->ne[2], (long long) b->ne[3]);
        throw Error(msg);
    }
    return ggml_mul_mat(ctx, a, b);
}

ggml_tensor * reshape3(ggml_context * ctx, ggml_tensor * a, int64_t n0, int64_t n1,
                       int64_t n2, const char * what) {
    if (!ggml_is_contiguous(a) || ggml_nelements(a) != n0 * n1 * n2) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "breath: reshape3(%s) [%lld,%lld,%lld,%lld]=%lld -> [%lld,%lld,%lld]",
            what, (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2],
            (long long) a->ne[3], (long long) ggml_nelements(a),
            (long long) n0, (long long) n1, (long long) n2);
        throw Error(msg);
    }
    return ggml_reshape_3d(ctx, a, n0, n1, n2);
}

// RoPE in the ONNX's convention: the graph builds `cat(-x[D/2:], x[:D/2])` and
// multiplies it by sin/cos of `pos * theta^(-i/(D/2))`, i.e. the *half-split*
// (Neox) layout.  ggml's default NORMAL mode rotates interleaved pairs
// (i, i+1), which is a different model of the same idea — the two only agree
// when D == 2.
ggml_tensor * rope_neox(ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions,
                        int n_dims, float theta) {
    return ggml_rope_ext(ctx, x, positions, /*freq_factors=*/nullptr,
                         n_dims,
                         /*mode=*/GGML_ROPE_TYPE_NEOX,
                         /*n_ctx_orig=*/0,
                         /*freq_base=*/theta,
                         /*freq_scale=*/1.0f,
                         /*ext_factor=*/0.0f,
                         /*attn_factor=*/1.0f,
                         /*beta_fast=*/0.0f,
                         /*beta_slow=*/0.0f);
}

// Linear with the same operand checks (see mul_mat above).  The ONNX exporter
// emits the pointwise ("pw") branches as Conv1d(kernel=1), whose weight keeps
// the trailing kernel axis — [OC, IC, 1] -> ggml (1, IC, OC) — so collapse it
// back to the (in, out) matrix the mul_mat wants.
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const LinearW & l,
                     const char * what) {
    ggml_tensor * w = l.w;
    if (w->ne[0] == 1 && w->ne[3] == 1 && w->ne[1] > 1 && w->ne[2] > 1) {
        w = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);
    }
    ggml_tensor * y = mul_mat(ctx, w, x, what);
    return l.b ? add(ctx, y, l.b) : y;
}

// ONNX LayerNormalization(axis=-1):  y = (x - mu) / sqrt(var + eps) * w + b,
// normalised along ne0 (the channel axis), which is what ggml_norm does.
ggml_tensor * layernorm(ggml_context * ctx, ggml_tensor * x, const NormW & n, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    return add(ctx, ggml_mul(ctx, y, n.w), n.b);
}

// Per-channel bias broadcast against a [W, H, C, N] tensor (front conv layout).
ggml_tensor * bias_whcn(ggml_context * ctx, ggml_tensor * b, int64_t C) {
    return ggml_reshape_4d(ctx, b, 1, 1, C, 1);
}

// Per-channel bias broadcast against a [T, C, N] tensor (depthwise layout).
ggml_tensor * bias_tcn(ggml_context * ctx, ggml_tensor * b, int64_t C) {
    return ggml_reshape_4d(ctx, b, 1, C, 1, 1);
}

// NOTE on ggml_permute: the argument list is the *inverse* of the natural
// "result axis i comes from source axis p[i]" reading — ggml_permute(x, p0..p3)
// places source axis i at result axis p[i].  The 2-cycles (…,1,0,…)/(…,2,1,…)
// are self-inverse so the distinction only shows up on 3-cycles like the
// conv-output rotation below.

// 2-D convolution (kernel [KW, KH, IC, OC], input [W, H, IC, N]) built from
// im2col + mul_mat with F32 operands.  ggml_conv_2d would force the im2col
// intermediate through F16; the ONNX reference runs in F32 and the front convs
// sit right on the input, so the accumulator is kept in F32 here.
// Returns [OW, OH, OC, N] — the ONNX NHWC layout.
ggml_tensor * conv2d_f32(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * x,
                         int p0, int p1) {
    const int64_t K  = kernel->ne[0] * kernel->ne[1] * kernel->ne[2];
    const int64_t OC = kernel->ne[3];
    ggml_tensor * im = ggml_im2col(ctx, kernel, x, /*s0=*/1, /*s1=*/1,
                                   p0, p1, /*d0=*/1, /*d1=*/1,
                                   /*is_2D=*/true, GGML_TYPE_F32);
    const int64_t OW = im->ne[1], OH = im->ne[2], N = im->ne[3];
    ggml_tensor * w    = ggml_reshape_2d(ctx, kernel, K, OC);
    ggml_tensor * cols = ggml_reshape_2d(ctx, im, K, OW * OH * N);
    ggml_tensor * y    = mul_mat(ctx, w, cols, "conv2d");
    y = ggml_reshape_4d(ctx, y, OC, OW, OH, N);
    // (OC, OW, OH, N) -> (OW, OH, OC, N)
    return ggml_cont(ctx, ggml_permute(ctx, y, 2, 0, 1, 3));
}

// MaxPool2d kernel (1,2) stride (1,2): pools the frequency axis only.
ggml_tensor * maxpool_freq(ggml_context * ctx, ggml_tensor * x) {
    return ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, /*k0=*/1, /*k1=*/2,
                        /*s0=*/1, /*s1=*/2, /*p0=*/0.0f, /*p1=*/0.0f);
}

// Depthwise 1-D conv over time on a [N, C] tensor (channels on ne0).
ggml_tensor * dwconv_time(ggml_context * ctx, const ConvW & k, ggml_tensor * x,
                          int kernel_size) {
    const int pad = (kernel_size - 1) / 2;
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));   // [T, C, N]
    ggml_tensor * y  = ops::dwconv_1d(ctx, k.w, xt, kernel_size, pad);
    y = add(ctx, y, bias_tcn(ctx, k.b, k.w->ne[2]));
    return ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));               // [C, T, N]
}

// ONNX Slice of the fused qkv output along ne0 (the three [D] chunks).
ggml_tensor * chunk_cont(ggml_context * ctx, ggml_tensor * x, int64_t dim, int64_t offset) {
    const std::size_t esize = ggml_element_size(x);
    ggml_tensor * v = ggml_view_4d(ctx, x, dim, x->ne[1], x->ne[2], x->ne[3],
                                   x->nb[1], x->nb[2], x->nb[3],
                                   static_cast<std::size_t>(offset) * esize);
    return ggml_cont(ctx, v);
}

// ---------------------------------------------------------------------------
// block
// ---------------------------------------------------------------------------

// Debug taps: BREATH_GGML_TAPS=<dir> copies the named intermediates out of the
// graph (ggml_cpy into a dedicated output tensor, see AGENT.md §3) so a parity
// run can diff them against the ONNX reference node by node.
struct TapSink {
    bool on = false;
    std::vector<std::pair<std::string, ggml_tensor *>> items;
    void add(const std::string & name, ggml_tensor * t) {
        if (on) items.emplace_back(name, t);
    }
};

// GLU-ish FFN: LayerNorm -> Linear(D->H) -> SiLU -> Linear(H->D), residual x0.5.
ggml_tensor * ffn_branch(ggml_context * ctx, ggml_tensor * x, const NormW & norm,
                         const LinearW & fc1, const LinearW & fc2, float eps,
                         TapSink * taps = nullptr, const char * tag = "") {
    ggml_tensor * h = layernorm(ctx, x, norm, eps);
    h = linear(ctx, h, fc1, "ffn-fc1");
    h = ggml_silu(ctx, h);                       // ONNX: Mul(z, Sigmoid(z))
    h = linear(ctx, h, fc2, "ffn-fc2");
    h = ggml_scale(ctx, h, 0.5f);
    if (taps) taps->add(std::string(tag) + ".half", h);
    return h;
}

// Attention branch: plain multi-head attention over the fused
// [Q | K | V] projection (Gather_2/3/4 of the qkv reshape are slices 0/1/2 of
// the third axis), RoPE on Q and K only.
//   scores = rope(q) @ rope(k)^T / sqrt(D)    (the ONNX scales Q and K each by
//                                              D^-1/4, i.e. the scores by D^-1/2)
//   out    = softmax(scores) @ v
ggml_tensor * attn_branch(ggml_context * ctx, ggml_tensor * x, const BlockW & L,
                          ggml_tensor * positions, const BreathNetConfig & cfg,
                          TapSink * taps = nullptr, const char * tag = "") {
    const int64_t D = cfg.n_heads * cfg.head_dim;
    ggml_tensor * h   = layernorm(ctx, x, L.ln, cfg.norm_eps);
    ggml_tensor * qkv = linear(ctx, h, L.qkv, "qkv");
    if (taps) {
        taps->add(std::string(tag) + ".ln", h);
        taps->add(std::string(tag) + ".qkv", qkv);
    }

    // split a [H*d, T] chunk into [d, H, T]; RoPE runs per (head, position)
    auto head_view = [&](ggml_tensor * c, bool rope) {
        ggml_tensor * r = reshape3(ctx, c, cfg.head_dim, cfg.n_heads, c->ne[1], "qkv-head");
        if (rope) r = rope_neox(ctx, r, positions, cfg.head_dim, cfg.rope_theta);
        return r;                                                    // [d, H, T]
    };
    ggml_tensor * q = head_view(chunk_cont(ctx, qkv, D, 0),     true);
    ggml_tensor * k = head_view(chunk_cont(ctx, qkv, D, D),     true);
    ggml_tensor * v = head_view(chunk_cont(ctx, qkv, D, 2 * D), false);

    ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));   // [d, T_q, H]
    ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));   // [d, T_k, H]

    ggml_tensor * scores = mul_mat(ctx, kp, qp, "scores");
    // the fused scale keeps the 1/sqrt(D) out of the score tensor
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, /*mask=*/nullptr,
                                            cfg.attn_scale, /*max_bias=*/0.0f);

    if (taps) {
        taps->add(std::string(tag) + ".q_rope", q);
        taps->add(std::string(tag) + ".k_rope", k);
        taps->add(std::string(tag) + ".v", v);
    }
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));   // [T_k, d, H]
    ggml_tensor * o  = mul_mat(ctx, vt, probs, "ctx");
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));            // [d, H, T_q]
    o = reshape3(ctx, o, D, o->ne[2], 1, "attn-out");                // [H*d, T]
    return linear(ctx, o, L.attn_out, "attn-out");
}

// Channel branch: pw1 -> (a * sigmoid(b)) -> depthwise -> SiLU -> pw2.
ggml_tensor * conv_branch(ggml_context * ctx, ggml_tensor * x, const BlockW & L,
                          const BreathNetConfig & cfg,
                          TapSink * taps = nullptr, const char * tag = "") {
    const int64_t D = cfg.dim;
    ggml_tensor * h = layernorm(ctx, x, L.ln, cfg.norm_eps);
    ggml_tensor * p = linear(ctx, h, L.conv_pw1, "conv-pw1");
    ggml_tensor * a = chunk_cont(ctx, p, D, 0);
    ggml_tensor * b = chunk_cont(ctx, p, D, D);
    ggml_tensor * g = ggml_mul(ctx, a, ggml_sigmoid(ctx, b));            // GLU (not SiLU)
    if (taps) taps->add(std::string(tag) + ".conv_glu", g);
    g = dwconv_time(ctx, L.conv_dw, g, cfg.conv_kernel);
    g = ggml_silu(ctx, g);                                               // dw(x)*sigmoid(dw(x))
    if (taps) taps->add(std::string(tag) + ".conv_dw_act", g);
    return linear(ctx, g, L.conv_pw2, "conv-pw2");
}

ggml_tensor * block(ggml_context * ctx, ggml_tensor * x, const BlockW & L,
                    ggml_tensor * positions, const BreathNetConfig & cfg, int bi,
                    TapSink * taps = nullptr) {
    const std::string tag = "b" + std::to_string(bi);
    if (taps) taps->add(tag + ".in", x);
    x = add(ctx, x, ffn_branch(ctx, x, L.ffn1_norm, L.ffn1_fc1, L.ffn1_fc2,
                               cfg.norm_eps, taps, (tag + ".ffn1").c_str()));
    if (taps) taps->add(tag + ".ffn1_res", x);

    ggml_tensor * a = attn_branch(ctx, x, L, positions, cfg, taps, tag.c_str());
    ggml_tensor * c = conv_branch(ctx, x, L, cfg, taps, tag.c_str());
    if (taps) {
        taps->add(tag + ".attn_out", a);
        taps->add(tag + ".conv_out", c);
    }

    // merge: concat on the channel axis, depthwise over the 2D channels,
    // residual, then project back to D.
    ggml_tensor * m = ggml_concat(ctx, a, c, /*dim=*/0);            // [2D, T]
    if (taps) taps->add(tag + ".merge_cat", m);
    ggml_tensor * d = dwconv_time(ctx, L.merge_dw, m, cfg.merge_kernel);
    if (taps) taps->add(tag + ".merge_dw", d);
    m = add(ctx, m, d);
    x = add(ctx, x, linear(ctx, m, L.merge_proj, "merge-proj"));
    if (taps) taps->add(tag + ".merge_res", x);

    x = add(ctx, x, ffn_branch(ctx, x, L.ffn2_norm, L.ffn2_fc1, L.ffn2_fc2,
                               cfg.norm_eps, taps, (tag + ".ffn2").c_str()));
    x = layernorm(ctx, x, L.ln_out, cfg.norm_eps);
    if (taps) taps->add(tag + ".out", x);
    return x;
}

// ---------------------------------------------------------------------------
// stage (one fixed window length)
// ---------------------------------------------------------------------------

struct Stage {
    ggml_context *  ctx   = nullptr;
    ggml_cgraph *   graph = nullptr;
    ggml_gallocr *  alloc = nullptr;
    ggml_tensor *   in_rows = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   out = nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> taps;   // debug only
    int             T = 0;

    bool matches(int t) const { return ctx != nullptr && T == t; }

    void reset() {
        if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
        graph = nullptr;
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        in_rows = positions = out = nullptr;
        T = 0;
    }
    ~Stage() { reset(); }
};

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct BreathNet::Impl {
    BreathNetConfig  cfg;
    ggml_backend_t   backend = nullptr;
    std::unique_ptr<LoadedWeights> weights;
    Weights          w;
    std::vector<std::unique_ptr<Stage>> stages;

    Stage & ensure_stage(int T);
    ~Impl() { free_backend(backend); }
};

Stage & BreathNet::Impl::ensure_stage(int T) {
    for (auto & s : stages) {
        if (s->matches(T)) return *s;
    }
    if (static_cast<int>(stages.size()) >= kMaxStages) {
        stages.erase(stages.begin());      // LRU-ish: drop the oldest window
    }
    stages.push_back(std::make_unique<Stage>());
    Stage & st = *stages.back();

    ggml_init_params ip{};
    ip.mem_size = kGraphCtxBytes;
    ip.no_alloc = true;
    st.ctx = ggml_init(ip);
    if (!st.ctx) throw Error("breath: ggml_init failed (graph context)");
    st.graph = ggml_new_graph_custom(st.ctx, kGraphNodes, /*grads=*/false);

    ggml_context * ctx = st.ctx;
    const BreathNetConfig & cfg = this->cfg;
    const Weights & W = this->w;

    // ---- input: [W=T, H=input_rows, IC=1, N=1] --------------------------
    st.in_rows = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T, cfg.input_rows, 1, 1);
    ggml_set_input(st.in_rows);

    st.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(st.positions);

    TapSink taps;
    taps.on = std::getenv("BREATH_GGML_TAPS") != nullptr;

    // ---- front ----------------------------------------------------------
    ggml_tensor * x = st.in_rows;
    taps.add("front.in", x);
    for (int i = 0; i < static_cast<int>(cfg.front_channels.size()); ++i) {
        const ConvW & c = W.front[static_cast<std::size_t>(i)];
        // 3x3 conv, same padding, H = frequency, W = time
        x = conv2d_f32(ctx, c.w, x, /*p0=*/1, /*p1=*/1);
        taps.add("front.c" + std::to_string(i), x);
        x = add(ctx, x, bias_whcn(ctx, c.b, c.w->ne[3]));
        x = ggml_relu(ctx, x);
        taps.add("front.r" + std::to_string(i), x);
        if (std::find(cfg.front_pools.begin(), cfg.front_pools.end(), i) !=
            cfg.front_pools.end()) {
            x = maxpool_freq(ctx, x);    // floor((H-k)/s)+1 == the ONNX formula
            taps.add("front.p" + std::to_string(i), x);
        }
    }
    // [OW=T, OH=F, OC=C, N=1] -> ONNX [N, T, C, F] -> flatten (C, F)
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));     // [F, C, T, 1]
    x = reshape3(ctx, x, cfg.proj_in, T, 1, "front-flatten");
    taps.add("front.flat", x);
    x = linear(ctx, x, W.proj, "proj");
    taps.add("front.out", x);

    // ---- trunk ----------------------------------------------------------
    for (int i = 0; i < cfg.n_blocks; ++i) {
        x = block(ctx, x, W.blocks[static_cast<std::size_t>(i)], st.positions, cfg,
                  i, &taps);
    }

    ggml_tensor * logits = linear(ctx, x, W.head, "head");
    taps.add("head.logits", logits);
    st.out = ggml_sigmoid(ctx, logits);
    ggml_set_output(st.out);
    ggml_build_forward_expand(st.graph, st.out);

    // Taps must be copied *before* the allocator runs, otherwise their memory
    // is reused by later nodes and the readback is garbage (AGENT.md §3).
    for (auto & t : taps.items) {
        ggml_tensor * dst = ggml_dup_tensor(ctx, t.second);
        ggml_tensor * cpy = ggml_cpy(ctx, t.second, dst);
        ggml_set_output(cpy);
        ggml_build_forward_expand(st.graph, cpy);
        st.taps.emplace_back(t.first, cpy);
    }

    st.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(st.alloc, st.graph)) {
        throw Error("breath: ggml_gallocr_alloc_graph failed (T=" + std::to_string(T) + ")");
    }
    st.T = T;
    return st;
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

BreathNet::BreathNet() : impl_(std::make_unique<Impl>()) {}
BreathNet::~BreathNet() = default;
BreathNet::BreathNet(BreathNet &&) noexcept = default;
BreathNet & BreathNet::operator=(BreathNet &&) noexcept = default;

const BreathNetConfig & BreathNet::config() const noexcept { return impl_->cfg; }
const char * BreathNet::backend_name() const noexcept {
    return internal::backend_name(impl_->backend);
}

namespace {

int get_int(const GgufFile & g, const std::string & key, int fallback) {
    const auto v = g.get_int_opt(key);
    return v ? static_cast<int>(*v) : fallback;
}

float get_float(const GgufFile & g, const std::string & key, float fallback) {
    const auto v = g.get_float_opt(key);
    return v ? static_cast<float>(*v) : fallback;
}

}  // namespace

BreathNet BreathNet::load(const std::string & gguf_path) {
    BreathNet net;
    auto & impl = *net.impl_;

    GgufFile gguf = GgufFile::open(gguf_path);
    const std::string arch = gguf.get_string("general.architecture");
    if (arch != "breath-ap") {
        throw NotImplemented("breath: architecture '" + arch + "' is not supported");
    }

    BreathNetConfig & cfg = impl.cfg;
    cfg.dim         = get_int(gguf, "breath.model.dim", 192);
    cfg.n_blocks    = get_int(gguf, "breath.model.blocks", 5);
    cfg.ffn_hidden  = get_int(gguf, "breath.model.ffn_hidden", 768);
    cfg.n_heads     = get_int(gguf, "breath.model.n_heads", 4);
    cfg.head_dim    = get_int(gguf, "breath.model.head_dim", 48);
    cfg.n_out       = get_int(gguf, "breath.model.n_out", 4);
    cfg.rope_theta  = get_float(gguf, "breath.model.rope_theta", 10000.0f);
    cfg.attn_scale  = get_float(gguf, "breath.model.attn_scale",
                                1.0f / std::sqrt(static_cast<float>(cfg.head_dim)));
    cfg.front_freq_bins = get_int(gguf, "breath.model.front_freq_bins", 8);
    cfg.input_rows      = get_int(gguf, "breath.feature.input_rows", 66);
    cfg.conv_kernel     = get_int(gguf, "breath.model.conv_kernel", 15);
    cfg.merge_kernel    = get_int(gguf, "breath.model.merge_kernel", 31);
    cfg.proj_in         = get_int(gguf, "breath.model.proj_in", 1024);

    for (int i = 0;; ++i) {
        const auto v = gguf.get_int_opt("breath.model.front_channels." + std::to_string(i));
        if (!v) break;
        cfg.front_channels.push_back(static_cast<int>(*v));
    }
    if (cfg.front_channels.empty()) {
        throw GgufError("breath: breath.model.front_channels.* missing from " + gguf_path);
    }
    for (int i = 0;; ++i) {
        const auto v = gguf.get_int_opt("breath.model.front_pools." + std::to_string(i));
        if (!v) break;
        cfg.front_pools.push_back(static_cast<int>(*v));
    }

    impl.backend = internal::init_best_backend();
    // The dedicated CONV_2D_DW kernel is CPU-only; every other backend uses the
    // im2col path (see ops_basic.cpp).
    ops::set_direct_dwconv(std::strcmp(internal::backend_name(impl.backend), "CPU") == 0);

    impl.weights = std::make_unique<LoadedWeights>(
        LoadedWeights::load_all(gguf, impl.backend));
    impl.w = bind_weights(*impl.weights, cfg);
    return net;
}

std::vector<float> BreathNet::run(const float * rows, int stride, int T) const {
    auto & impl = *impl_;
    if (!rows || T <= 0) throw InvalidArgument("breath: empty window");
    if (stride < T) throw InvalidArgument("breath: window wider than the row stride");

    Stage & st = impl.ensure_stage(T);

    // The front conv wants [W=T, H=input_rows, IC=1, N=1] — ggml ne=(T, rows)
    // with ne0 innermost, which is *element-wise identical* to the row-major
    // [input_rows, stride] layout the features use, so each feature row is one
    // contiguous run of the window (no transpose).
    static thread_local std::vector<float> window;
    window.resize(static_cast<std::size_t>(T) * impl.cfg.input_rows);
    for (int r = 0; r < impl.cfg.input_rows; ++r) {
        std::memcpy(window.data() + static_cast<std::size_t>(r) * T,
                    rows + static_cast<std::size_t>(r) * stride,
                    static_cast<std::size_t>(T) * sizeof(float));
    }
    static thread_local std::vector<std::int32_t> pos;
    pos.resize(static_cast<std::size_t>(T));
    for (int i = 0; i < T; ++i) pos[static_cast<std::size_t>(i)] = i;

    // Inputs are re-uploaded immediately before every compute: the arena can
    // alias a leaf with a temporary (AGENT.md §2).
    ggml_backend_tensor_set(st.in_rows, window.data(), 0,
                            window.size() * sizeof(float));
    ggml_backend_tensor_set(st.positions, pos.data(), 0,
                            pos.size() * sizeof(std::int32_t));

    if (ggml_backend_graph_compute(impl.backend, st.graph) != GGML_STATUS_SUCCESS) {
        throw Error("breath: graph compute failed");
    }

    std::vector<float> out(static_cast<std::size_t>(impl.cfg.n_out) * T);
    ggml_backend_tensor_get(st.out, out.data(), 0, out.size() * sizeof(float));

    if (!st.taps.empty()) {
        const char * dir = std::getenv("BREATH_GGML_TAPS");
        for (const auto & t : st.taps) {
            std::vector<float> buf(static_cast<std::size_t>(ggml_nelements(t.second)));
            ggml_backend_tensor_get(t.second, buf.data(), 0, buf.size() * sizeof(float));
            const std::string base = std::string(dir) + "/" + t.first;
            if (FILE * f = std::fopen((base + ".bin").c_str(), "wb")) {
                std::fwrite(buf.data(), sizeof(float), buf.size(), f);
                std::fclose(f);
            }
            if (FILE * f = std::fopen((base + ".shape").c_str(), "w")) {
                std::fprintf(f, "%lld %lld %lld %lld\n",
                             (long long) t.second->ne[0], (long long) t.second->ne[1],
                             (long long) t.second->ne[2], (long long) t.second->ne[3]);
                std::fclose(f);
            }
        }
    }
    return out;
}

}  // namespace tifa_ggml::internal
