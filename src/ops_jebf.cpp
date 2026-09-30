#include "ops_jebf.h"

#include "ops_ffn.h"
#include "ops_rope.h"

#include <ggml.h>

#include <cmath>
#include <cstdint>

namespace tifa_ggml::internal::ops {

namespace {

// Chunk a (3*D_total, n_tokens, 1) tensor into q, k, v (PyTorch chunk order).
struct QKVTriple { ggml_tensor * q; ggml_tensor * k; ggml_tensor * v; };

QKVTriple chunk_qkv_3(ggml_context * ctx, ggml_tensor * qkv) {
    const int64_t      Dtot  = qkv->ne[0] / 3;
    const std::size_t  esize = ggml_element_size(qkv);
    QKVTriple r;
    r.q = ggml_view_4d(ctx, qkv, Dtot, qkv->ne[1], qkv->ne[2], qkv->ne[3],
                       qkv->nb[1], qkv->nb[2], qkv->nb[3], 0);
    r.k = ggml_view_4d(ctx, qkv, Dtot, qkv->ne[1], qkv->ne[2], qkv->ne[3],
                       qkv->nb[1], qkv->nb[2], qkv->nb[3], Dtot * esize);
    r.v = ggml_view_4d(ctx, qkv, Dtot, qkv->ne[1], qkv->ne[2], qkv->ne[3],
                       qkv->nb[1], qkv->nb[2], qkv->nb[3], 2 * Dtot * esize);
    return r;
}

// Multiply by a (1, n, 1, 1) f32 padding mask (no-op when null).
ggml_tensor * apply_mask(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mask) {
    if (!mask) return x;
    return ggml_mul(ctx, x, mask);
}

}  // namespace

// ---------------------------------------------------------------------------
// Joint attention (TIFA modules/backbones/attention.py:JointAttention)
// ---------------------------------------------------------------------------

JoinResult joint_attention(
    ggml_context * ctx,
    ggml_tensor * token,
    ggml_tensor * x,
    const JointAttentionWeights & W,
    ggml_tensor * token_positions,
    ggml_tensor * x_positions,
    int num_heads,
    int head_dim,
    float theta,
    ggml_tensor * n_mask,
    ggml_tensor * t_mask,
    JebfTaps * taps)
{
    const int64_t N    = token->ne[1];
    const int64_t T    = x->ne[1];
    const int64_t Dtot = num_heads * head_dim;

    // ---- per-stream pre-norm + fused QKV --------------------------------
    ggml_tensor * token_norm = rms_norm(ctx, token, W.w_token_norm, 1e-6f);
    ggml_tensor * x_norm     = rms_norm(ctx, x,     W.w_x_norm,     1e-6f);

    ggml_tensor * token_qkv = linear(ctx, token_norm, W.w_token_qkv, W.b_token_qkv);
    ggml_tensor * x_qkv     = linear(ctx, x_norm,     W.w_x_qkv,     W.b_x_qkv);
    if (taps) {
        taps->attn_qkv_token = token_qkv;
        taps->attn_qkv_x     = x_qkv;
        taps->attn_norm_token = token_norm;
        taps->attn_norm_x     = x_norm;
    }

    QKVTriple tq = chunk_qkv_3(ctx, token_qkv);
    QKVTriple xq = chunk_qkv_3(ctx, x_qkv);

    // ---- (D, H, n, 1) views ---------------------------------------------
    auto to_heads = [&](ggml_tensor * t, int64_t n) {
        return ggml_reshape_4d(ctx, ggml_cont(ctx, t), head_dim, num_heads, n, 1);
    };
    ggml_tensor * token_q = to_heads(tq.q, N);
    ggml_tensor * token_k = to_heads(tq.k, N);
    ggml_tensor * token_v = to_heads(tq.v, N);
    ggml_tensor * x_q     = to_heads(xq.q, T);
    ggml_tensor * x_k     = to_heads(xq.k, T);
    ggml_tensor * x_v     = to_heads(xq.v, T);

    if (taps) taps->attn_q_token_pre = token_q;
    // ---- per-head Q/K RMSNorm (along head_dim = ne0) ---------------------
    token_q = rms_norm(ctx, token_q, W.w_token_q_norm, 1e-6f);
    if (taps) taps->attn_q_token_normed = token_q;
    token_k = rms_norm(ctx, token_k, W.w_token_k_norm, 1e-6f);
    x_q     = rms_norm(ctx, x_q,     W.w_x_q_norm,     1e-6f);
    x_k     = rms_norm(ctx, x_k,     W.w_x_k_norm,     1e-6f);

    // ---- plain RoPE per stream (positions start at 0 for both) ----------
    token_q = apply_rope(ctx, token_q, token_positions, head_dim, theta);
    token_k = apply_rope(ctx, token_k, token_positions, head_dim, theta);
    x_q     = apply_rope(ctx, x_q,     x_positions,     head_dim, theta);
    x_k     = apply_rope(ctx, x_k,     x_positions,     head_dim, theta);

    if (taps) {
        taps->attn_q_token = token_q; taps->attn_k_token = token_k;
        taps->attn_q_x = x_q;         taps->attn_k_x = x_k;
        taps->attn_v_token = token_v; taps->attn_v_x = x_v;
    }

    // ---- joint SDPA over concat([token, x]) -----------------------------
    ggml_tensor * q = ggml_concat(ctx, token_q, x_q, /*dim=*/2);   // (D, H, S, 1)
    ggml_tensor * k = ggml_concat(ctx, token_k, x_k, /*dim=*/2);
    ggml_tensor * v = ggml_concat(ctx, token_v, x_v, /*dim=*/2);

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));          // (D, S, H, 1)
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    // No attention mask: padded positions stay in the key/value set, exactly
    // as in the PyTorch reference (padding is zeroed, not excluded).
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr,
                                            scale, /*max_bias=*/0.0f,
                                            /*logit_softcap=*/0.0f);  // (D, S, H, 1)

    out = ggml_cont(ctx, out);
    out = ggml_reshape_3d(ctx, out, Dtot, N + T, 1);

    ggml_tensor * token_rows = ggml_cont(ctx, ggml_view_3d(ctx, out,
        Dtot, N, 1, out->nb[1], out->nb[2], 0));
    ggml_tensor * x_rows = ggml_cont(ctx, ggml_view_3d(ctx, out,
        Dtot, T, 1, out->nb[1], out->nb[2], static_cast<std::size_t>(N) * out->nb[1]));

    JoinResult r;
    r.token = linear(ctx, token_rows, W.w_token_out, W.b_token_out);
    r.x     = linear(ctx, x_rows,     W.w_x_out,     W.b_x_out);
    // JointAttention._mask_outputs: zero padded rows before the residual add.
    r.token = apply_mask(ctx, r.token, n_mask);
    r.x     = apply_mask(ctx, r.x,     t_mask);
    return r;
}

// ---------------------------------------------------------------------------
// PJAC
// ---------------------------------------------------------------------------

namespace {

// Merge one stream: concat([attn ; cgmlp]) -> optional depthwise conv + residual
// -> linear -> layer scale.
ggml_tensor * merge_stream(
    ggml_context * ctx,
    ggml_tensor * a_s,
    ggml_tensor * c_s,
    const JebfMergeWeights & M,
    ggml_tensor * w_lay_scale)
{
    ggml_tensor * m = ggml_concat(ctx, a_s, c_s, /*dim=*/0);      // (2D, n, 1)
    if (M.kernel != 0 && M.w_dw) {
        ggml_tensor * m_tr = ggml_cont(ctx, ggml_permute(ctx, m, 1, 0, 2, 3));
        const int pad = (M.kernel - 1) / 2;
        ggml_tensor * cv = dwconv_1d(ctx, M.w_dw, m_tr, M.kernel, pad);
        if (M.b_dw) {
            const std::size_t es = ggml_element_size(M.b_dw);
            const int64_t     C  = M.b_dw->ne[0];
            cv = ggml_add(ctx, cv, ggml_view_4d(ctx, M.b_dw, 1, C, 1, 1, es, C * es, C * es, 0));
        }
        ggml_tensor * cv_back = ggml_cont(ctx, ggml_permute(ctx, cv, 1, 0, 2, 3));
        m = ggml_add(ctx, cv_back, m);
    }
    m = linear(ctx, m, M.w_linear, M.b_linear);
    if (w_lay_scale) m = layer_scale(ctx, m, w_lay_scale);
    return m;
}

}  // namespace

JoinResult pjac(
    ggml_context * ctx,
    ggml_tensor * token,
    ggml_tensor * x,
    const PjacWeights & W,
    ggml_tensor * token_positions,
    ggml_tensor * x_positions,
    int num_heads,
    int head_dim,
    float theta,
    ggml_tensor * n_mask,
    ggml_tensor * t_mask,
    JebfTaps * taps)
{
    // Attention branch (JointAttention has its own per-stream pre-norms).
    JoinResult a = joint_attention(ctx, token, x, W.jattn,
                                   token_positions, x_positions,
                                   num_heads, head_dim, theta, n_mask, t_mask, taps);
    if (taps) { taps->attn_token = a.token; taps->attn_x = a.x; }
    // (joint_attention already applied the padding masks to its outputs.)

    // CgMLP branches (own pre-norms).
    ggml_tensor * c_token = cgmlp(ctx,
        rms_norm(ctx, token, W.w_c_norm_token, 1e-6f),
        W.c_token.w_pw1, W.c_token.b_pw1, W.c_token.w_norm,
        W.c_token.w_dw,  W.c_token.b_dw,
        W.c_token.w_pw2, W.c_token.b_pw2,
        W.c_token.kernel);
    ggml_tensor * c_x = cgmlp(ctx,
        rms_norm(ctx, x, W.w_c_norm_x, 1e-6f),
        W.c_x.w_pw1, W.c_x.b_pw1, W.c_x.w_norm,
        W.c_x.w_dw,  W.c_x.b_dw,
        W.c_x.w_pw2, W.c_x.b_pw2,
        W.c_x.kernel);

    JoinResult r;
    r.token = merge_stream(ctx, a.token, c_token, W.merge_token, W.w_lay_scale_token);
    r.x     = merge_stream(ctx, a.x,     c_x,     W.merge_x,     W.w_lay_scale_x);
    return r;
}

// ---------------------------------------------------------------------------
// JEBF block
// ---------------------------------------------------------------------------

namespace {

ggml_tensor * apply_ffn(ggml_context * ctx, ggml_tensor * x,
                        const JebfFfnWeights & F,
                        ggml_tensor ** norm_out = nullptr,
                        ggml_tensor ** glu_out = nullptr)
{
    ggml_tensor * h = rms_norm(ctx, x, F.w_norm, 1e-6f);
    if (norm_out) *norm_out = h;
    if (F.w_ln1_a) {
        h = glu_ffn_split(ctx, h, F.w_ln1_a, F.b_ln1_a, F.w_ln1_b, F.b_ln1_b,
                          F.w_ln2, F.b_ln2);
    } else {
        h = glu_ffn(ctx, h, F.w_ln1, F.b_ln1, F.w_ln2, F.b_ln2);
    }
    if (glu_out) *glu_out = h;
    if (F.w_lay_scale) h = layer_scale(ctx, h, F.w_lay_scale);
    // JEBF residual: x + branch (no 0.5 factor — that is EBF, not JEBF).
    return ggml_add(ctx, x, h);
}

}  // namespace

JoinResult jebf_block(
    ggml_context * ctx,
    ggml_tensor * token,
    ggml_tensor * x,
    const JebfBlockWeights & W,
    ggml_tensor * token_positions,
    ggml_tensor * x_positions,
    int num_heads,
    int head_dim,
    float theta,
    ggml_tensor * n_mask,
    ggml_tensor * t_mask,
    JebfTaps * taps)
{
    // ---- entry padding mask (JEBF.forward applies it before FFN1) --------
    x     = apply_mask(ctx, x,     t_mask);
    token = apply_mask(ctx, token, n_mask);

    if (W.has_ffn1) {
        ggml_tensor * xn = nullptr, * xg = nullptr;
        x     = apply_ffn(ctx, x,     W.ffn1_x, taps ? &xn : nullptr, taps ? &xg : nullptr);
        token = apply_ffn(ctx, token, W.ffn1_token);
        if (taps) { taps->ffn1_x_norm = xn; taps->ffn1_x_glu = xg; }
    }
    if (taps) { taps->ffn1_token = token; taps->ffn1_x = x; }

    // ---- padding mask before PJAC ---------------------------------------
    x     = apply_mask(ctx, x,     t_mask);
    token = apply_mask(ctx, token, n_mask);

    JoinResult a = pjac(ctx, token, x, W.pjac,
                        token_positions, x_positions,
                        num_heads, head_dim, theta, n_mask, t_mask, taps);
    if (taps) { taps->pjac_token = a.token; taps->pjac_x = a.x; }
    x     = ggml_add(ctx, x,     a.x);
    token = ggml_add(ctx, token, a.token);

    // ---- padding mask before FFN2 ---------------------------------------
    x     = apply_mask(ctx, x,     t_mask);
    token = apply_mask(ctx, token, n_mask);

    if (W.has_ffn2) {
        x     = apply_ffn(ctx, x,     W.ffn2_x);
        token = apply_ffn(ctx, token, W.ffn2_token);
    }

    return {token, x};
}

}  // namespace tifa_ggml::internal::ops
