// FoxBreatheLabeler network graph — see fbl_net.h for the block structure.

#include "fbl_net.h"

#include "backend.h"
#include "ops_basic.h"
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
constexpr int         kMaxStages     = 4;

// ---- weights ---------------------------------------------------------------

struct LinearW { ggml_tensor * w = nullptr; ggml_tensor * b = nullptr; };
struct NormW   { ggml_tensor * w = nullptr; ggml_tensor * b = nullptr; };
using ConvW = LinearW;      // conv kernels bind the same (weight, bias) pair

struct BlockW {
    NormW   norm[4];
    LinearW ffn1_a, ffn1_b;      // dim -> 4*dim -> dim
    LinearW ffn2_a, ffn2_b;
    LinearW att_q, att_kv, att_out;
    LinearW conv_pw1, conv_pw2;
    ConvW   conv_dw;
    ggml_tensor * bn_scale = nullptr;   // BatchNorm folded to y = x*s + t
    ggml_tensor * bn_shift = nullptr;
};

struct Weights {
    LinearW in, out;
    NormW   final_norm;
    std::vector<BlockW> blocks;
};

int get_int(const GgufFile & g, const std::string & key, int fallback) {
    const auto v = g.get_int_opt(key);
    return v ? static_cast<int>(*v) : fallback;
}

float get_float(const GgufFile & g, const std::string & key, float fallback) {
    const auto v = g.get_float_opt(key);
    return v ? *v : fallback;
}

ggml_tensor * need(const LoadedWeights & w, const std::string & name) {
    ggml_tensor * t = w.get(name);
    if (!t) throw Error("fbl: missing tensor '" + name + "'");
    return t;
}

void bind_linear(const LoadedWeights & w, const std::string & p, LinearW & l) {
    l.w = need(w, p + ".weight");
    // the attention projections are bias=False in the reference model
    l.b = w.try_get(p + ".bias");
}

void bind_norm(const LoadedWeights & w, const std::string & p, NormW & n) {
    n.w = need(w, p + ".weight");
    n.b = need(w, p + ".bias");
}

Weights bind_weights(const LoadedWeights & w, const FblConfig & cfg) {
    Weights out;
    bind_linear(w, "fbl.in",  out.in);
    bind_linear(w, "fbl.out", out.out);
    bind_norm(w, "fbl.final_norm", out.final_norm);
    out.blocks.resize(static_cast<std::size_t>(cfg.n_blocks));
    for (int i = 0; i < cfg.n_blocks; ++i) {
        const std::string p = "fbl.enc." + std::to_string(i);
        BlockW & b = out.blocks[static_cast<std::size_t>(i)];
        for (int n = 0; n < 4; ++n) {
            bind_norm(w, p + ".norm" + std::to_string(n + 1), b.norm[n]);
        }
        bind_linear(w, p + ".ffn1.1", b.ffn1_a);
        bind_linear(w, p + ".ffn1.2", b.ffn1_b);
        bind_linear(w, p + ".ffn2.1", b.ffn2_a);
        bind_linear(w, p + ".ffn2.2", b.ffn2_b);
        bind_linear(w, p + ".att.q",   b.att_q);
        bind_linear(w, p + ".att.kv",  b.att_kv);
        bind_linear(w, p + ".att.out", b.att_out);
        bind_linear(w, p + ".conv.pw1", b.conv_pw1);
        bind_linear(w, p + ".conv.pw2", b.conv_pw2);
        bind_linear(w, p + ".conv.dw",  b.conv_dw);
        b.bn_scale = need(w, p + ".conv.bn.scale");
        b.bn_shift = need(w, p + ".conv.bn.shift");
    }
    return out;
}

// ---- graph helpers ---------------------------------------------------------

// ggml_add aborts the process on an invalid broadcast; check first.
ggml_tensor * add(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (!ggml_can_repeat(b, a)) {
        throw Error("fbl: add shape mismatch");
    }
    return ggml_add(ctx, a, b);
}

ggml_tensor * mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                      const char * what) {
    if (a->ne[0] != b->ne[0]) {
        throw Error(std::string("fbl: ") + what + " inner dim mismatch");
    }
    return ggml_mul_mat(ctx, a, b);
}

ggml_tensor * reshape3(ggml_context * ctx, ggml_tensor * a, int64_t n0, int64_t n1,
                       int64_t n2, const char * what) {
    if (!ggml_is_contiguous(a) || ggml_nelements(a) != n0 * n1 * n2) {
        throw Error(std::string("fbl: ") + what + " cannot reshape");
    }
    return ggml_reshape_3d(ctx, a, n0, n1, n2);
}

// y = x @ W^T + b, with x [K, T] and W stored [K, N] (ne0 = in features).
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const LinearW & l,
                     const char * what) {
    // ggml stores PyTorch's Linear [OC, IC] as ne=[IC, OC], and a Conv1d
    // [OC, IC, K] as ne=[K, IC, OC]; both collapse to the [IC, OC] mul_mat wants.
    ggml_tensor * w = l.w;
    if (w->ne[2] == 1 && w->ne[3] == 1) {
        w = ggml_reshape_2d(ctx, w, w->ne[0], w->ne[1]);       // Linear
    } else if (w->ne[0] == 1 && w->ne[3] == 1) {
        w = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);       // 1x1 conv
    } else {
        throw Error("fbl: unsupported weight layout for a linear");
    }
    ggml_tensor * y = mul_mat(ctx, w, x, what);
    if (!l.b) return y;
    // [N] -> [N, 1] so it broadcasts along the time axis
    return add(ctx, y, ggml_reshape_2d(ctx, l.b, l.b->ne[0], 1));
}

// LayerNorm over ne0 (the channel axis), which is what ggml_norm does.
ggml_tensor * layernorm(ggml_context * ctx, ggml_tensor * x, const NormW & n,
                        float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    return add(ctx, ggml_mul(ctx, y, ggml_reshape_2d(ctx, n.w, n.w->ne[0], 1)),
               ggml_reshape_2d(ctx, n.b, n.b->ne[0], 1));
}

// ONNX Slice of a [2D, T] tensor along ne0.
ggml_tensor * chunk_cont(ggml_context * ctx, ggml_tensor * x, int64_t dim, int64_t offset) {
    const std::size_t esize = ggml_element_size(x);
    ggml_tensor * v = ggml_view_4d(ctx, x, dim, x->ne[1], x->ne[2], x->ne[3],
                                   x->nb[1], x->nb[2], x->nb[3],
                                   static_cast<std::size_t>(offset) * esize);
    return ggml_cont(ctx, v);
}

// Depthwise 1-D conv over time on a [C, T] tensor (channels on ne0).
ggml_tensor * dwconv_time(ggml_context * ctx, const ConvW & k, ggml_tensor * x,
                          int kernel_size) {
    const int pad = (kernel_size - 1) / 2;
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));   // [T, C]
    ggml_tensor * y  = ops::dwconv_1d(ctx, k.w, xt, kernel_size, pad);
    y = add(ctx, y, ggml_reshape_2d(ctx, k.b, 1, k.b->ne[0]));
    return ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));               // [C, T]
}

// ---- blocks ----------------------------------------------------------------

// conform_ffn: Linear -> SiLU -> Linear.
ggml_tensor * ffn(ggml_context * ctx, ggml_tensor * x, const LinearW & a,
                  const LinearW & b, const char * what) {
    ggml_tensor * h = linear(ctx, x, a, what);
    h = ggml_silu(ctx, h);
    return linear(ctx, h, b, what);
}

// Multi-head self-attention, no RoPE: q/k/v are three Linear projections and
// the scores are scaled by 1/sqrt(head_dim).
ggml_tensor * attn(ggml_context * ctx, ggml_tensor * x, const BlockW & L,
                   const FblConfig & cfg) {
    const int64_t D = static_cast<int64_t>(cfg.n_heads) * cfg.head_dim;
    ggml_tensor * q  = linear(ctx, x, L.att_q,  "att-q");
    ggml_tensor * kv = linear(ctx, x, L.att_kv, "att-kv");

    const auto head_view = [&](ggml_tensor * c) {
        return reshape3(ctx, c, cfg.head_dim, cfg.n_heads, c->ne[1], "att-head");
    };
    ggml_tensor * qh = head_view(q);
    ggml_tensor * kh = head_view(chunk_cont(ctx, kv, D, 0));
    ggml_tensor * vh = head_view(chunk_cont(ctx, kv, D, D));

    ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, qh, 0, 2, 1, 3));   // [d, T, H]
    ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, kh, 0, 2, 1, 3));

    ggml_tensor * scores = mul_mat(ctx, kp, qp, "att-scores");
    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, /*mask=*/nullptr, scale, 0.0f);

    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 2, 0, 3));   // [T, d, H]
    ggml_tensor * o  = mul_mat(ctx, vt, probs, "att-ctx");
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                   // [d, H, T]
    o = reshape3(ctx, o, D, o->ne[2], 1, "att-out");
    return linear(ctx, o, L.att_out, "att-out");
}

// conform_conv: pw1 -> GLU -> depthwise -> BatchNorm -> SiLU -> pw2.
ggml_tensor * conv_branch(ggml_context * ctx, ggml_tensor * x, const BlockW & L,
                          const FblConfig & cfg) {
    const int64_t C = cfg.dim;
    ggml_tensor * p = linear(ctx, x, L.conv_pw1, "conv-pw1");
    ggml_tensor * a = chunk_cont(ctx, p, C, 0);
    ggml_tensor * b = chunk_cont(ctx, p, C, C);
    ggml_tensor * g = ggml_mul(ctx, a, ggml_sigmoid(ctx, b));       // GLU
    g = dwconv_time(ctx, L.conv_dw, g, cfg.kernel);
    // BatchNorm1d folded to y = x * scale + shift (conversion-time).
    g = add(ctx, ggml_mul(ctx, g, ggml_reshape_2d(ctx, L.bn_scale, C, 1)),
            ggml_reshape_2d(ctx, L.bn_shift, C, 1));
    g = ggml_silu(ctx, g);
    return linear(ctx, g, L.conv_pw2, "conv-pw2");
}

ggml_tensor * block(ggml_context * ctx, ggml_tensor * x, const BlockW & L,
                    const FblConfig & cfg) {
    ggml_tensor * h = ffn(ctx, layernorm(ctx, x, L.norm[0], cfg.norm_eps),
                          L.ffn1_a, L.ffn1_b, "ffn1");
    x = add(ctx, ggml_scale(ctx, h, 0.5f), x);

    h = attn(ctx, layernorm(ctx, x, L.norm[1], cfg.norm_eps), L, cfg);
    x = add(ctx, h, x);

    h = conv_branch(ctx, layernorm(ctx, x, L.norm[2], cfg.norm_eps), L, cfg);
    x = add(ctx, h, x);

    h = ffn(ctx, layernorm(ctx, x, L.norm[3], cfg.norm_eps),
            L.ffn2_a, L.ffn2_b, "ffn2");
    return add(ctx, ggml_scale(ctx, h, 0.5f), x);
}

}  // namespace

// ---- Stage -----------------------------------------------------------------

struct Stage {
    ggml_context *  ctx   = nullptr;
    ggml_cgraph *   graph = nullptr;
    ggml_gallocr *  alloc = nullptr;
    ggml_tensor *   in    = nullptr;
    ggml_tensor *   out   = nullptr;
    int             T     = 0;

    bool matches(int t) const { return ctx != nullptr && T == t; }

    void reset() {
        if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
        graph = nullptr;
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        in = out = nullptr;
        T = 0;
    }
    ~Stage() { reset(); }
};

struct FblNet::Impl {
    FblConfig  cfg;
    ggml_backend_t backend = nullptr;
    std::unique_ptr<LoadedWeights> weights;
    Weights w;
    std::vector<std::unique_ptr<Stage>> stages;

    Stage & ensure_stage(int T);
    ~Impl() { free_backend(backend); }
};

Stage & FblNet::Impl::ensure_stage(int T) {
    for (auto & s : stages) {
        if (s->matches(T)) return *s;
    }
    if (static_cast<int>(stages.size()) >= kMaxStages) stages.erase(stages.begin());
    stages.push_back(std::make_unique<Stage>());
    Stage & st = *stages.back();

    ggml_init_params ip{};
    ip.mem_size = kGraphCtxBytes;
    ip.no_alloc = true;
    st.ctx = ggml_init(ip);
    if (!st.ctx) throw Error("fbl: ggml_init failed (graph context)");
    st.graph = ggml_new_graph_custom(st.ctx, kGraphNodes, /*grads=*/false);

    ggml_context * ctx = st.ctx;
    const FblConfig & cfg = this->cfg;

    st.in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.spec_win, T);
    ggml_set_input(st.in);

    ggml_tensor * x = linear(ctx, st.in, this->w.in, "in");
    for (int i = 0; i < cfg.n_blocks; ++i) {
        x = block(ctx, x, this->w.blocks[static_cast<std::size_t>(i)], cfg);
    }
    x = layernorm(ctx, x, this->w.final_norm, cfg.norm_eps);
    ggml_tensor * logits = linear(ctx, x, this->w.out, "out");
    st.out = ggml_sigmoid(ctx, logits);
    ggml_set_output(st.out);
    ggml_build_forward_expand(st.graph, st.out);

    st.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(st.alloc, st.graph)) {
        throw Error("fbl: gallocr could not allocate the graph");
    }
    st.T = T;
    return st;
}

// ---- public ----------------------------------------------------------------

FblNet::FblNet() : impl_(new Impl()) {}

FblNet::~FblNet() = default;
FblNet::FblNet(FblNet &&) noexcept = default;
FblNet & FblNet::operator=(FblNet &&) noexcept = default;

const FblConfig & FblNet::config() const noexcept { return impl_->cfg; }
const char * FblNet::backend_name() const noexcept {
    return impl_->backend ? ::tifa_ggml::internal::backend_name(impl_->backend) : "<null>";
}

FblNet FblNet::load(const std::string & gguf_path) {
    FblNet m;
    Impl & impl = *m.impl_;

    GgufFile gguf = GgufFile::open(gguf_path);
    const std::string kind = gguf.get_string_opt("breath.model_kind").value_or("breathlab");
    if (kind != "fbl") {
        throw NotImplemented("fbl: model kind '" + kind + "' is not the FBL net");
    }
    impl.cfg.spec_win    = get_int(gguf, "fbl.spec_win", 1024);
    impl.cfg.hop         = get_int(gguf, "fbl.hop", 882);
    impl.cfg.sample_rate = get_int(gguf, "fbl.sample_rate", 44100);
    impl.cfg.dim         = get_int(gguf, "fbl.dim", 512);
    impl.cfg.n_blocks    = get_int(gguf, "fbl.layers", 6);
    impl.cfg.n_heads     = get_int(gguf, "fbl.heads", 8);
    impl.cfg.head_dim    = get_int(gguf, "fbl.head_dim", 64);
    impl.cfg.kernel      = get_int(gguf, "fbl.kernel", 31);
    impl.cfg.fps         = get_float(gguf, "fbl.fps", 50.0f);
    impl.cfg.threshold   = get_float(gguf, "fbl.threshold", 0.4f);
    impl.cfg.min_dur     = get_int(gguf, "fbl.min_dur_frames", 4);
    impl.cfg.max_gap     = get_int(gguf, "fbl.max_gap", 5);

    impl.backend = init_best_backend();
    impl.weights = std::make_unique<LoadedWeights>(
        LoadedWeights::load_all(gguf, impl.backend));
    impl.w = bind_weights(*impl.weights, impl.cfg);
    return m;
}

std::vector<float> FblNet::run(const float * frames, int T) const {
    auto & impl = *impl_;
    if (!frames || T <= 0) throw InvalidArgument("fbl: empty window");

    Stage & st = impl.ensure_stage(T);
    ggml_backend_tensor_set(st.in, frames, 0,
                            static_cast<std::size_t>(T) * impl.cfg.spec_win * sizeof(float));

    if (ggml_backend_graph_compute(impl.backend, st.graph) != GGML_STATUS_SUCCESS) {
        throw Error("fbl: graph compute failed");
    }
    std::vector<float> out(static_cast<std::size_t>(T));
    ggml_backend_tensor_get(st.out, out.data(), 0, out.size() * sizeof(float));
    return out;
}

}  // namespace tifa_ggml::internal
