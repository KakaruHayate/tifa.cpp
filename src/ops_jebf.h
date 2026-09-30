#pragma once

// Two-stream JEBF block: JointAttention(token, x) + PJAC + JEBF layer.
//
// Ported from openvpi/TIFA modules/backbones/jebf.py + attention.py.
// Structurally the same block GAME's estimator uses (see game.cpp
// ops_joint_attn.*), with three TIFA-specific differences:
//
//   1. No attention mask at all.  Padding is handled by *multiply* masks
//      (`x = x * t_mask`, `token = token * n_mask`) applied before each
//      sub-block and on the attention outputs.  Padded positions still
//      participate as attention keys/values (their K/V are the projection
//      bias terms), so the graph must not "helpfully" exclude them.
//   2. Plain RoPE per stream: token positions 0..N-1, frame positions 0..T-1,
//      theta = 10000, full head_dim (GGML_ROPE_TYPE_NORMAL / interleaved).
//   3. FFN residuals are `x + branch` — there is NO 0.5 factor (that is the
//      single-stream EBF convention, not JEBF).
//
// Shape convention (ggml ne order, innermost first): activations are
// (D, n_tokens, 1).  Masks are f32 with ne = (1, n_tokens, 1, 1) so that
// ggml_mul broadcasts along ne0 (ggml_can_repeat requires equal-or-1 dims).

#include "ops_basic.h"

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace tifa_ggml::internal::ops {

struct JoinResult { ggml_tensor * token; ggml_tensor * x; };

// Optional per-sub-block taps for layer-by-layer parity debugging.
struct JebfTaps {
    ggml_tensor * ffn1_token = nullptr;
    ggml_tensor * ffn1_x     = nullptr;
    ggml_tensor * attn_token = nullptr;   // JointAttention outputs
    ggml_tensor * attn_x     = nullptr;
    ggml_tensor * pjac_token = nullptr;   // PJAC merge outputs
    ggml_tensor * pjac_x     = nullptr;
    // FFN1_x pieces (norm output, GLU output before layer scale/residual)
    ggml_tensor * ffn1_x_norm = nullptr;
    ggml_tensor * ffn1_x_glu  = nullptr;
    // Post-RoPE Q/K per stream, shape (head_dim, heads, n, 1)
    ggml_tensor * attn_q_token = nullptr;
    ggml_tensor * attn_k_token = nullptr;
    ggml_tensor * attn_q_x     = nullptr;
    ggml_tensor * attn_k_x     = nullptr;
    ggml_tensor * attn_v_token = nullptr;
    ggml_tensor * attn_v_x     = nullptr;
    // Raw pre-chunk QKV linear outputs (3*head_dim*heads wide) and the
    // per-stream pre-norms.
    ggml_tensor * attn_qkv_token = nullptr;
    ggml_tensor * attn_qkv_x     = nullptr;
    ggml_tensor * attn_norm_token = nullptr;
    ggml_tensor * attn_norm_x     = nullptr;
    // Q pipeline pieces for the token stream: pre-norm (post head reshape)
    // and post-qk_norm (pre-rope).
    ggml_tensor * attn_q_token_pre    = nullptr;
    ggml_tensor * attn_q_token_normed = nullptr;
};

// ----------------------------------------------------------------------------
// Weight bundles (tensor names are the PyTorch module paths, see converter)
// ----------------------------------------------------------------------------
struct JointAttentionWeights {
    // Pre-QKV RMSNorms (per stream, dim = D_embed).
    ggml_tensor * w_token_norm   = nullptr;   // .jattn.token_norm.weight
    ggml_tensor * w_x_norm       = nullptr;   // .jattn.x_norm.weight
    // Fused QKV projections (D_embed -> 3*H*D_head), chunk order q, k, v.
    ggml_tensor * w_token_qkv    = nullptr;
    ggml_tensor * b_token_qkv    = nullptr;
    ggml_tensor * w_x_qkv        = nullptr;
    ggml_tensor * b_x_qkv        = nullptr;
    // Per-head Q/K RMSNorms (dim = D_head).
    ggml_tensor * w_token_q_norm = nullptr;
    ggml_tensor * w_token_k_norm = nullptr;
    ggml_tensor * w_x_q_norm     = nullptr;
    ggml_tensor * w_x_k_norm     = nullptr;
    // Output projections (H*D_head -> D_embed).
    ggml_tensor * w_token_out    = nullptr;
    ggml_tensor * b_token_out    = nullptr;
    ggml_tensor * w_x_out        = nullptr;
    ggml_tensor * b_x_out        = nullptr;
};

struct JebfCgmlpWeights {
    ggml_tensor * w_pw1  = nullptr;   // Conv1d(dim, 2*dim, 1)  ne=(1, dim, 2*dim)
    ggml_tensor * b_pw1  = nullptr;
    ggml_tensor * w_norm = nullptr;   // RMSNorm(dim)
    ggml_tensor * w_dw   = nullptr;   // Conv1d(dim, dim, K, groups=dim) ne=(K, 1, dim)
    ggml_tensor * b_dw   = nullptr;
    ggml_tensor * w_pw2  = nullptr;   // Conv1d(dim, dim, 1)  ne=(1, dim, dim)
    ggml_tensor * b_pw2  = nullptr;
    int           kernel = 31;
};

struct JebfMergeWeights {
    ggml_tensor * w_linear = nullptr;      // Linear(2*dim -> dim)
    ggml_tensor * b_linear = nullptr;
    ggml_tensor * w_dw     = nullptr;      // Conv1d(2*dim, 2*dim, K, groups=2*dim)
    ggml_tensor * b_dw     = nullptr;
    int           kernel   = 31;           // 0 = no depthwise merge conv
};

struct PjacWeights {
    JointAttentionWeights jattn{};
    // Per-stream CgMLP pre-norms (.attn.c_norm_{x,token}.weight)
    ggml_tensor * w_c_norm_token = nullptr;
    ggml_tensor * w_c_norm_x     = nullptr;
    JebfCgmlpWeights c_token{};            // kernel 7
    JebfCgmlpWeights c_x{};                // kernel 31
    JebfMergeWeights merge_token{};        // kernel 5
    JebfMergeWeights merge_x{};            // kernel 31
    // layer_scale_{jpac_token,jpac_x}.scale — applied to the merge outputs.
    ggml_tensor * w_lay_scale_token = nullptr;
    ggml_tensor * w_lay_scale_x     = nullptr;
};

struct JebfFfnWeights {
    ggml_tensor * w_norm = nullptr;        // norm_ffn{1,2}_{x,token}.weight
    ggml_tensor * w_ln1  = nullptr;        // ffn.ln1.weight (dim -> 2*latent)
    ggml_tensor * b_ln1  = nullptr;
    // Optional load-time pre-split halves of ln1 (see tensor_utils):
    ggml_tensor * w_ln1_a = nullptr;
    ggml_tensor * b_ln1_a = nullptr;
    ggml_tensor * w_ln1_b = nullptr;
    ggml_tensor * b_ln1_b = nullptr;
    ggml_tensor * w_ln2  = nullptr;
    ggml_tensor * b_ln2  = nullptr;
    ggml_tensor * w_lay_scale = nullptr;   // layer_scale_*.scale (dim,)
};

struct JebfBlockWeights {
    bool has_ffn1 = true;
    bool has_ffn2 = true;
    JebfFfnWeights ffn1_token{};
    JebfFfnWeights ffn1_x{};
    PjacWeights   pjac{};
    JebfFfnWeights ffn2_token{};
    JebfFfnWeights ffn2_x{};
};

// ----------------------------------------------------------------------------
// Graph builders
// ----------------------------------------------------------------------------

// Joint attention across the token and frame streams.
//   token        : (D, N, 1)      x            : (D, T, 1)
//   token_pos    : int32 (N,)     x_pos        : int32 (T,)
//   n_mask/t_mask: f32 (1, N|T, 1, 1), 1 = valid, 0 = padding (may be null)
// Returns the two *output projections* (before the residual add), already
// multiplied by the padding masks.
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
    ggml_tensor * n_mask = nullptr,
    ggml_tensor * t_mask = nullptr);

// PJAC: parallel joint attention + CgMLP, per-stream depthwise merge.
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
    ggml_tensor * n_mask = nullptr,
    ggml_tensor * t_mask = nullptr,
    JebfTaps * taps = nullptr);

// Full JEBF block: FFN1 -> PJAC -> FFN2 with per-block plain residuals and
// the TIFA padding-mask placement.
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
    ggml_tensor * n_mask = nullptr,
    ggml_tensor * t_mask = nullptr,
    JebfTaps * taps = nullptr);

}  // namespace tifa_ggml::internal::ops
