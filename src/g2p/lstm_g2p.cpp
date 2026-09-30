// English LSTM G2P: GGUF loading + ggml-graph inference.
//
// Port of openvpi/TIFA's g2p/converters/lstm.py over the ONNX pair that
// scripts/convert_lstm_g2p_to_gguf.py re-lays out into a "lstm-g2p" GGUF.
// The recurrent cores run through GGML_OP_LSTM (cmake/patches/ggml-lstm-op.md,
// following the fused-RNN pattern of ggml-audio-patch's GRU); everything
// around them — embeddings, input projections, Bahdanau attention, the ReLU
// head — is plain ggml graph nodes:
//
//   encoder (one graph per word):
//     char embedding -> BiLSTM(hidden, 2 layers) -> projection, with the
//     decoder start state = per-layer mean of the two directional final
//     states (the op emits final h/c as extra output planes).
//   decoder (one graph per word, batched over the beam):
//     Bahdanau attention over the encoder outputs (query = top-layer hidden)
//     feeding an embedding+context LSTM cell (2 layers) and a ReLU head
//     producing phoneme logits; the host only re-uploads the token ids and
//     the carried state each step.
//
// Gate order inside the 4*hidden axis is the ONNX convention i, o, f, c.  The
// beam search mirrors the reference exactly: per-parent top-k pruning on the
// log-softmax (stable, ties resolved by the lower token id), length-normalised
// cumulative score over the surviving beams (excluding BOS, including EOS),
// finished beams carried unchanged, and a final order-preserving dedup of the
// produced pronunciations.

#include "lstm_g2p.h"

#include "../backend.h"
#include "../gguf_io.h"
#include "../json.h"
#include "../ops_basic.h"
#include "../tensor_utils.h"
#include "tifa_ggml/errors.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tifa_ggml::internal::g2p {

namespace {

[[noreturn]] void bad_model(const std::string & why) {
    throw InvalidArgument("lstm-g2p GGUF: " + why);
}

int64_t require_int(const GgufFile & file, const std::string & key) {
    if (!file.has(key)) bad_model("missing key '" + key + "'");
    return file.get_int(key);
}

// {grapheme: id} / {phoneme: id} JSON embedded by the converter.
std::unordered_map<std::string, int> load_vocab(const std::string & json_text,
                                                const std::string & what) {
    const json::Value doc = json::parse(json_text);
    if (!doc.is_object()) bad_model(what + " vocabulary is not a JSON object");
    std::unordered_map<std::string, int> out;
    for (const auto & entry : doc.obj) {
        if (!entry.second.is_number()) bad_model(what + " vocabulary has a non-numeric id");
        out.emplace(entry.first, static_cast<int>(entry.second.number));
    }
    return out;
}

// RAII for one built graph: a no-alloc ggml context, the graph itself and its
// arena allocator.  Weight tensors live in the LoadedWeights buffer and are
// referenced, not owned, by every graph built here.
//
// Graph INPUTS must not live in the arena: the gallocr is free to hand a dead
// tensor's span to a later node, and a decoder input that is re-uploaded every
// step then silently aliases the node it was reused for.  Inputs are therefore
// created in a small dedicated context (`ictx`) backed by its own buffer
// (`ibuf`) — the same treatment the weights get — which the allocator skips.
struct GraphRun {
    ggml_context * ctx   = nullptr;   // graph + intermediates (no_alloc)
    ggml_context * ictx  = nullptr;   // graph inputs
    ggml_cgraph *  graph = nullptr;
    ggml_gallocr * alloc = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    ggml_backend_t backend = nullptr;

    // Call after every input tensor has been created in `ictx`.
    void allocate_inputs() {
        ibuf = ggml_backend_alloc_ctx_tensors(ictx, backend);
        if (ibuf == nullptr) throw Error("lstm-g2p: cannot allocate the input buffer");
    }

    void allocate() {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, graph)) {
            throw Error("lstm-g2p: ggml_gallocr_alloc_graph failed");
        }
    }

    ~GraphRun() {
        if (ibuf != nullptr) ggml_backend_buffer_free(ibuf);
        if (alloc != nullptr) ggml_gallocr_free(alloc);
        if (ictx != nullptr) ggml_free(ictx);
        if (ctx != nullptr) ggml_free(ctx);
    }
};

constexpr std::size_t kGraphCtxBytes = 8u << 20;   // tensor structs + node list
constexpr int         kGraphNodes    = 512;

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct LstmG2p::Impl {
    struct Config {
        int hidden = 0;
        int input_dim = 0;        // encoder embedding width
        int embedding_dim = 0;    // decoder embedding width
        int num_chars = 0;
        int num_phonemes = 0;
        int max_len = 0;
        int beam_size = 0;
        int phoneme_unk = 0, phoneme_pad = 0, phoneme_bos = 0, phoneme_eos = 0;
        int char_unk = 0;
    } cfg;

    std::unordered_map<std::string, int> char_vocab;
    std::vector<std::string> phoneme_list;                    // id -> phoneme string

    ggml_backend_t                 backend = nullptr;  // CPU: the LSTM op is CPU-side
    std::unique_ptr<LoadedWeights> weights;

    // bound weight tensors
    ggml_tensor * enc_embed  = nullptr;                 // [input_dim, num_chars]
    ggml_tensor * enc_w0     = nullptr;                 // [in, 4H, 2]
    ggml_tensor * enc_r0     = nullptr;                 // [H, 4H, 2]
    ggml_tensor * enc_b0     = nullptr;                 // [8H, 2]
    ggml_tensor * enc_w1     = nullptr;                 // [2H, 4H, 2]
    ggml_tensor * enc_r1     = nullptr;                 // [H, 4H, 2]
    ggml_tensor * enc_b1     = nullptr;                 // [8H, 2]
    ggml_tensor * enc_proj_w = nullptr;                 // [2H, H]
    ggml_tensor * enc_proj_b = nullptr;                 // [H]
    ggml_tensor * dec_embed  = nullptr;                 // [embedding_dim, num_phonemes]
    ggml_tensor * dec_w0     = nullptr;                 // [E+H, 4H]
    ggml_tensor * dec_r0     = nullptr;                 // [H, 4H]
    ggml_tensor * dec_b0     = nullptr;                 // [8H]  (a [8H, 1] direction slice)
    ggml_tensor * dec_w1     = nullptr;                 // [H, 4H]
    ggml_tensor * dec_r1     = nullptr;                 // [H, 4H]
    ggml_tensor * dec_b1     = nullptr;                 // [8H]
    ggml_tensor * att_k_w    = nullptr;                 // [2H, H]
    ggml_tensor * att_k_b    = nullptr;                 // [H]
    ggml_tensor * att_v_w    = nullptr;                 // [H]  (a [H, 1] out-major mat)
    ggml_tensor * fc0_w      = nullptr;                 // [2H, H]
    ggml_tensor * fc0_b      = nullptr;                 // [H]
    ggml_tensor * fc3_w      = nullptr;                 // [H, num_phonemes]
    ggml_tensor * fc3_b      = nullptr;                 // [num_phonemes]

    ~Impl() { free_backend(backend); }

    // One bidirectional encoder layer: runs both directions of ggml_lstm over
    // `x` [in, T] and returns the time-ordered direction-concatenated outputs
    // [2H, T]; the per-direction final states come back through the four
    // reference out-parameters (graph nodes).
    ggml_tensor * bi_layer(ggml_context * ctx, int layer, ggml_tensor * x,
                           ggml_tensor * & h_final_f, ggml_tensor * & h_final_b,
                           ggml_tensor * & c_final_f, ggml_tensor * & c_final_b) const {
        const int H = cfg.hidden;
        const int T = x->ne[1];
        const int in_dim = x->ne[0];

        ggml_tensor * w = layer == 0 ? enc_w0 : enc_w1;
        ggml_tensor * r = layer == 0 ? enc_r0 : enc_r1;
        ggml_tensor * b = layer == 0 ? enc_b0 : enc_b1;

        ggml_tensor * h_seq[2];
        for (int d = 0; d < 2; ++d) {
            // direction slice of the [in, 4H, 2] / [H, 4H, 2] / [8H, 2] weights
            ggml_tensor * wd = ggml_view_3d(ctx, w, in_dim, 4 * H, 1,
                                            in_dim * sizeof(float),
                                            in_dim * 4 * H * sizeof(float),
                                            static_cast<std::size_t>(d) * in_dim * 4 * H * sizeof(float));
            ggml_tensor * rd = ggml_view_3d(ctx, r, H, 4 * H, 1,
                                            H * sizeof(float),
                                            H * 4 * H * sizeof(float),
                                            static_cast<std::size_t>(d) * H * 4 * H * sizeof(float));
            ggml_tensor * bi  = ggml_view_1d(ctx, b, 4 * H,
                                             static_cast<std::size_t>(d) * 8 * H * sizeof(float));
            ggml_tensor * bhh = ggml_view_1d(ctx, b, 4 * H,
                                             (static_cast<std::size_t>(d) * 8 + 4) * H * sizeof(float));

            // input projection: [4H, T], gate rows in ONNX iofc order
            ggml_tensor * gi = ggml_add(ctx, ggml_mul_mat(ctx, wd, x), bi);
            gi = ggml_reshape_3d(ctx, gi, 4 * H, 1, T);

            // initial state zeros come from the op itself (null h0/c0)
            ggml_tensor * out = ggml_lstm(ctx, rd, gi, bhh, nullptr, nullptr, d == 1);
            h_seq[d] = ggml_view_3d(ctx, out, H, 1, T, H * sizeof(float),
                                    H * sizeof(float), 0);                     // [H, 1, T]
            const std::size_t plane = static_cast<std::size_t>(T) * H * sizeof(float);
            ggml_tensor * & hf = d == 0 ? h_final_f : h_final_b;
            ggml_tensor * & cf = d == 0 ? c_final_f : c_final_b;
            hf = ggml_view_2d(ctx, out, H, 1, H * sizeof(float), plane);
            cf = ggml_view_2d(ctx, out, H, 1, H * sizeof(float),
                              plane + H * sizeof(float));
        }

        return ggml_reshape_2d(ctx, ggml_concat(ctx, h_seq[0], h_seq[1], /*dim=*/0),
                               2 * H, T);
    }

    // ---- encoder graph ----------------------------------------------------
    // src ids [T] -> encoder outputs [H, T] + decoder start states [2H]
    struct EncoderGraph : GraphRun {
        ggml_tensor * in_ids  = nullptr;
        ggml_tensor * enc_out = nullptr;
        ggml_tensor * h_init  = nullptr;
        ggml_tensor * c_init  = nullptr;
    };

    std::unique_ptr<EncoderGraph> build_encoder(int T) const {
        auto g = std::make_unique<EncoderGraph>();
        g->backend = backend;

        ggml_init_params ip{};
        ip.mem_size = kGraphCtxBytes;
        ip.no_alloc = true;
        g->ctx = ggml_init(ip);
        if (g->ctx == nullptr) throw Error("lstm-g2p: ggml_init failed (encoder)");
        ggml_init_params iip{};
        iip.mem_size = 1u << 20;
        iip.no_alloc = true;
        g->ictx = ggml_init(iip);
        if (g->ictx == nullptr) throw Error("lstm-g2p: ggml_init failed (encoder inputs)");
        g->graph = ggml_new_graph_custom(g->ctx, kGraphNodes, /*grads=*/false);
        ggml_context * ctx = g->ctx;

        g->in_ids = ggml_new_tensor_1d(g->ictx, GGML_TYPE_I32, T);
        g->allocate_inputs();
        ggml_set_input(g->in_ids);

        ggml_tensor * x = ops::embedding(ctx, enc_embed, g->in_ids);       // [E, T]

        ggml_tensor * hT0f, * hT0b, * cT0f, * cT0b;
        ggml_tensor * y0 = bi_layer(ctx, 0, x, hT0f, hT0b, cT0f, cT0b);    // [2H, T]
        ggml_tensor * hT1f, * hT1b, * cT1f, * cT1b;
        ggml_tensor * y1 = bi_layer(ctx, 1, y0, hT1f, hT1b, cT1f, cT1b);   // [2H, T]

        // decoder start state: per-layer mean over the two directions,
        // concatenated layer-major ([layer0; layer1] on ne0).
        auto mean2 = [&](ggml_tensor * a, ggml_tensor * b) {
            return ggml_scale(ctx, ggml_add(ctx, a, b), 0.5f);             // [H, 1]
        };
        g->h_init = ggml_concat(ctx, mean2(hT0f, hT0b), mean2(hT1f, hT1b), 0);
        g->c_init = ggml_concat(ctx, mean2(cT0f, cT0b), mean2(cT1f, cT1b), 0);
        g->enc_out = ops::linear(ctx, y1, enc_proj_w, enc_proj_b);         // [H, T]

        for (ggml_tensor * out : { g->enc_out, g->h_init, g->c_init }) {
            ggml_set_output(out);
            ggml_build_forward_expand(g->graph, out);
        }
        g->allocate();
        return g;
    }

    // ---- decoder graph ----------------------------------------------------
    // One step, batched over the beam.  Inputs: prev token ids [B], carried
    // state [2H, B] (layer-major), encoder outputs [H, T].  Outputs: phoneme
    // logits [V, B] and the next state [2H, B].
    struct DecoderGraph : GraphRun {
        ggml_tensor * in_tok     = nullptr;
        ggml_tensor * in_h       = nullptr;
        ggml_tensor * in_c       = nullptr;
        ggml_tensor * in_enc     = nullptr;
        ggml_tensor * out_logits = nullptr;
        ggml_tensor * out_h      = nullptr;
        ggml_tensor * out_c      = nullptr;
    };

    std::unique_ptr<DecoderGraph> build_decoder(int T, int B) const {
        auto g = std::make_unique<DecoderGraph>();
        g->backend = backend;

        ggml_init_params ip{};
        ip.mem_size = kGraphCtxBytes;
        ip.no_alloc = true;
        g->ctx = ggml_init(ip);
        if (g->ctx == nullptr) throw Error("lstm-g2p: ggml_init failed (decoder)");
        ggml_init_params iip{};
        iip.mem_size = 1u << 20;
        iip.no_alloc = true;
        g->ictx = ggml_init(iip);
        if (g->ictx == nullptr) throw Error("lstm-g2p: ggml_init failed (decoder inputs)");
        g->graph = ggml_new_graph_custom(g->ctx, kGraphNodes, /*grads=*/false);
        ggml_context * ctx = g->ctx;

        const int H = cfg.hidden;
        const int E = cfg.embedding_dim;
        const int V = cfg.num_phonemes;

        g->in_tok = ggml_new_tensor_1d(g->ictx, GGML_TYPE_I32, B);
        g->in_h   = ggml_new_tensor_2d(g->ictx, GGML_TYPE_F32, 2 * H, B);
        g->in_c   = ggml_new_tensor_2d(g->ictx, GGML_TYPE_F32, 2 * H, B);
        g->in_enc = ggml_new_tensor_2d(g->ictx, GGML_TYPE_F32, H, T);
        g->allocate_inputs();
        for (ggml_tensor * t : { g->in_tok, g->in_h, g->in_c, g->in_enc }) {
            ggml_set_input(t);
        }

        // Bahdanau attention: k_j = tanh(Wk [h_top; enc_j] + bk),
        // a = softmax(v . k_j), context = sum_j a_j enc_j.
        ggml_tensor * q = ggml_view_2d(ctx, g->in_h, H, B, H * sizeof(float),
                                       H * sizeof(float));                     // top layer
        ggml_tensor * q_rep = ggml_repeat(ctx,
            ggml_reshape_3d(ctx, q, H, B, 1),
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, B, T));
        ggml_tensor * enc_rep = ggml_repeat(ctx,
            ggml_reshape_3d(ctx, g->in_enc, H, 1, T),
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, B, T));
        ggml_tensor * kv = ggml_concat(ctx, q_rep, enc_rep, 0);                // [2H, B, T]
        ggml_tensor * k = ggml_tanh(ctx,
            ggml_add(ctx, ggml_mul_mat(ctx, att_k_w, kv), att_k_b));           // [H, B, T]
        ggml_tensor * scores = ggml_mul_mat(ctx, att_v_w, k);                  // [1, B, T]
        ggml_tensor * att = ggml_soft_max(ctx, ggml_cont(ctx,
            ggml_permute(ctx, scores, 2, 1, 0, 3)));                           // [T, B]
        ggml_tensor * context = ggml_mul_mat(ctx,
            ggml_cont(ctx, ggml_transpose(ctx, g->in_enc)), att);              // [H, B]

        // layer 0 over [embedding; context]
        ggml_tensor * emb = ops::embedding(ctx, dec_embed, g->in_tok);         // [E, B]
        ggml_tensor * x0 = ggml_concat(ctx, emb, context, 0);                  // [E+H, B]
        ggml_tensor * gi0 = ggml_add(ctx, ggml_mul_mat(ctx, dec_w0, x0),
                                     ggml_view_1d(ctx, dec_b0, 4 * H, 0));
        ggml_tensor * lstm0 = ggml_lstm(ctx, dec_r0,
            ggml_reshape_3d(ctx, gi0, 4 * H, B, 1),
            ggml_view_1d(ctx, dec_b0, 4 * H, 4 * H * sizeof(float)),
            ggml_view_2d(ctx, g->in_h, H, B, H * sizeof(float), 0),
            ggml_view_2d(ctx, g->in_c, H, B, H * sizeof(float), 0), false);
        ggml_tensor * h0n = ggml_view_2d(ctx, lstm0, H, B, H * sizeof(float), 0);
        ggml_tensor * c0n = ggml_view_2d(ctx, lstm0, H, B, H * sizeof(float),
                                         2 * static_cast<std::size_t>(H) * B * sizeof(float));

        // layer 1 over the layer-0 output, seeded from the carried state
        ggml_tensor * gi1 = ggml_add(ctx, ggml_mul_mat(ctx, dec_w1, h0n),
                                     ggml_view_1d(ctx, dec_b1, 4 * H, 0));
        ggml_tensor * lstm1 = ggml_lstm(ctx, dec_r1,
            ggml_reshape_3d(ctx, gi1, 4 * H, B, 1),
            ggml_view_1d(ctx, dec_b1, 4 * H, 4 * H * sizeof(float)),
            ggml_view_2d(ctx, g->in_h, H, B, H * sizeof(float), H * sizeof(float)),
            ggml_view_2d(ctx, g->in_c, H, B, H * sizeof(float), H * sizeof(float)), false);
        ggml_tensor * h1n = ggml_view_2d(ctx, lstm1, H, B, H * sizeof(float), 0);
        ggml_tensor * c1n = ggml_view_2d(ctx, lstm1, H, B, H * sizeof(float),
                                         2 * static_cast<std::size_t>(H) * B * sizeof(float));

        // ReLU head over [layer-1 hidden; context]
        ggml_tensor * fc_in = ggml_concat(ctx, h1n, context, 0);               // [2H, B]
        ggml_tensor * fc = ggml_relu(ctx,
            ggml_add(ctx, ggml_mul_mat(ctx, fc0_w, fc_in), fc0_b));            // [H, B]
        g->out_logits = ggml_add(ctx, ggml_mul_mat(ctx, fc3_w, fc), fc3_b);    // [V, B]
        g->out_h = ggml_concat(ctx, h0n, h1n, 0);                              // [2H, B]
        g->out_c = ggml_concat(ctx, c0n, c1n, 0);

        for (ggml_tensor * out : { g->out_logits, g->out_h, g->out_c }) {
            ggml_set_output(out);
            ggml_build_forward_expand(g->graph, out);
        }
        g->allocate();
        return g;
    }
};

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

LstmG2p LstmG2p::from_file(const std::string & path) {
    GgufFile file = GgufFile::open(path);

    const std::string arch = file.has("general.architecture")
        ? file.get_string("general.architecture") : std::string();
    if (arch != "lstm-g2p") {
        bad_model("general.architecture is '" + arch + "', expected 'lstm-g2p'");
    }

    LstmG2p model;
    model.impl_.reset(new Impl());
    LstmG2p::Impl & I = *model.impl_;
    I.cfg.hidden        = static_cast<int>(require_int(file, "lstm_g2p.hidden"));
    I.cfg.input_dim     = static_cast<int>(require_int(file, "lstm_g2p.input_dim"));
    I.cfg.embedding_dim = static_cast<int>(require_int(file, "lstm_g2p.embedding_dim"));
    I.cfg.num_chars     = static_cast<int>(require_int(file, "lstm_g2p.num_chars"));
    I.cfg.num_phonemes  = static_cast<int>(require_int(file, "lstm_g2p.num_phonemes"));
    I.cfg.max_len       = static_cast<int>(require_int(file, "lstm_g2p.max_len"));
    I.cfg.beam_size     = static_cast<int>(require_int(file, "lstm_g2p.beam_size"));
    I.cfg.phoneme_unk   = static_cast<int>(require_int(file, "lstm_g2p.phoneme_unk"));
    I.cfg.phoneme_pad   = static_cast<int>(require_int(file, "lstm_g2p.phoneme_pad"));
    I.cfg.phoneme_bos   = static_cast<int>(require_int(file, "lstm_g2p.phoneme_bos"));
    I.cfg.phoneme_eos   = static_cast<int>(require_int(file, "lstm_g2p.phoneme_eos"));
    I.cfg.char_unk      = static_cast<int>(require_int(file, "lstm_g2p.char_unk"));

    if (require_int(file, "lstm_g2p.encoder_layers") != 2 ||
        require_int(file, "lstm_g2p.decoder_layers") != 2) {
        bad_model("only 2-layer encoder/decoder models are supported");
    }
    if (!file.has("lstm_g2p.encoder_bidirectional") ||
        !file.get_bool("lstm_g2p.encoder_bidirectional")) {
        bad_model("the encoder must be bidirectional");
    }
    if (I.cfg.hidden <= 0 || I.cfg.num_phonemes <= 0 || I.cfg.max_len <= 0) {
        bad_model("hyper-parameters are out of range");
    }

    I.char_vocab = load_vocab(file.get_string("lstm_g2p.char_vocab"), "char");
    const std::unordered_map<std::string, int> phoneme_vocab =
        load_vocab(file.get_string("lstm_g2p.phoneme_vocab"), "phoneme");
    I.phoneme_list.assign(static_cast<std::size_t>(I.cfg.num_phonemes), std::string());
    for (const auto & entry : phoneme_vocab) {
        if (entry.second >= 0 && entry.second < I.cfg.num_phonemes) {
            I.phoneme_list[static_cast<std::size_t>(entry.second)] = entry.first;
        }
    }

    // The G2P LSTM is tiny and the sweep op is CPU-side; keep the whole net on
    // a dedicated CPU backend regardless of what the aligner runs on.
    I.backend = init_backend(Backend::CPU);
    if (I.backend == nullptr) throw Error("lstm-g2p: cannot initialise the CPU backend");
    I.weights = std::make_unique<LoadedWeights>(
        LoadedWeights::load_all(file, I.backend));

    auto need = [&](const std::string & name) -> ggml_tensor * {
        return I.weights->get(name);   // throws GgufError naming the tensor
    };
    I.enc_embed  = need("encoder.embed.weight");
    I.enc_w0     = need("encoder.lstm.0.weight");
    I.enc_r0     = need("encoder.lstm.0.recurrence");
    I.enc_b0     = need("encoder.lstm.0.bias");
    I.enc_w1     = need("encoder.lstm.1.weight");
    I.enc_r1     = need("encoder.lstm.1.recurrence");
    I.enc_b1     = need("encoder.lstm.1.bias");
    I.enc_proj_w = need("encoder.projection.weight");
    I.enc_proj_b = need("encoder.projection.bias");
    I.dec_embed  = need("decoder.embed.weight");
    I.dec_w0     = need("decoder.lstm.0.weight");
    I.dec_r0     = need("decoder.lstm.0.recurrence");
    I.dec_b0     = need("decoder.lstm.0.bias");
    I.dec_w1     = need("decoder.lstm.1.weight");
    I.dec_r1     = need("decoder.lstm.1.recurrence");
    I.dec_b1     = need("decoder.lstm.1.bias");
    I.att_k_w    = need("decoder.attention.key.weight");
    I.att_k_b    = need("decoder.attention.key.bias");
    I.att_v_w    = need("decoder.attention.value.weight");
    I.fc0_w      = need("decoder.fc.0.weight");
    I.fc0_b      = need("decoder.fc.0.bias");
    I.fc3_w      = need("decoder.fc.3.weight");
    I.fc3_b      = need("decoder.fc.3.bias");

    // Shape gates: the tensors drive hard-coded view arithmetic below, so a
    // mismatch must fail at load time, not corrupt memory mid-decode.  ggml
    // hides trailing 1-dims from ggml_n_dims, so a [8H, 1] direction slice
    // checks as {8H}.
    auto check = [&](ggml_tensor * t, std::initializer_list<int64_t> ne,
                     const char * name) {
        bool ok = ggml_n_dims(t) == static_cast<int>(ne.size());
        std::size_t i = 0;
        for (const int64_t d : ne) ok = ok && t->ne[i++] == d;
        if (!ok) {
            std::string got = std::to_string(t->ne[0]);
            for (int k = 1; k < ggml_n_dims(t); ++k) {
                got += "x" + std::to_string(t->ne[k]);
            }
            bad_model(std::string("tensor '") + name + "' has shape " + got);
        }
    };
    const int64_t H = I.cfg.hidden, E = I.cfg.input_dim, D = I.cfg.embedding_dim,
                   V = I.cfg.num_phonemes;
    check(I.enc_embed,  { E, I.cfg.num_chars }, "encoder.embed.weight");
    check(I.enc_w0,     { E, 4 * H, 2 }, "encoder.lstm.0.weight");
    check(I.enc_r0,     { H, 4 * H, 2 }, "encoder.lstm.0.recurrence");
    check(I.enc_b0,     { 8 * H, 2 }, "encoder.lstm.0.bias");
    check(I.enc_w1,     { 2 * H, 4 * H, 2 }, "encoder.lstm.1.weight");
    check(I.enc_r1,     { H, 4 * H, 2 }, "encoder.lstm.1.recurrence");
    check(I.enc_b1,     { 8 * H, 2 }, "encoder.lstm.1.bias");
    check(I.enc_proj_w, { 2 * H, H }, "encoder.projection.weight");
    check(I.enc_proj_b, { H }, "encoder.projection.bias");
    check(I.dec_embed,  { D, V }, "decoder.embed.weight");
    check(I.dec_w0,     { D + H, 4 * H }, "decoder.lstm.0.weight");
    check(I.dec_r0,     { H, 4 * H }, "decoder.lstm.0.recurrence");
    check(I.dec_b0,     { 8 * H }, "decoder.lstm.0.bias");
    check(I.dec_w1,     { H, 4 * H }, "decoder.lstm.1.weight");
    check(I.dec_r1,     { H, 4 * H }, "decoder.lstm.1.recurrence");
    check(I.dec_b1,     { 8 * H }, "decoder.lstm.1.bias");
    check(I.att_k_w,    { 2 * H, H }, "decoder.attention.key.weight");
    check(I.att_k_b,    { H }, "decoder.attention.key.bias");
    check(I.att_v_w,    { H }, "decoder.attention.value.weight");
    check(I.fc0_w,      { 2 * H, H }, "decoder.fc.0.weight");
    check(I.fc0_b,      { H }, "decoder.fc.0.bias");
    check(I.fc3_w,      { H, V }, "decoder.fc.3.weight");
    check(I.fc3_b,      { V }, "decoder.fc.3.bias");

    return model;
}

// ---------------------------------------------------------------------------
// inference
// ---------------------------------------------------------------------------

LstmG2p::LstmG2p() = default;
LstmG2p::~LstmG2p() = default;
LstmG2p::LstmG2p(LstmG2p &&) noexcept = default;
LstmG2p & LstmG2p::operator=(LstmG2p &&) noexcept = default;

int LstmG2p::num_phonemes() const noexcept { return impl_->cfg.num_phonemes; }

bool LstmG2p::can_encode(const std::string & word) const {
    // Port of the `find` gate in g2p/converters/lstm.py: the word (lower-cased,
    // stripped, exactly like `predict` normalises) must not contain a
    // character outside the char vocabulary.
    const Impl & I = *impl_;
    std::size_t begin = 0, end = word.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(word[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(word[end - 1]))) --end;
    if (end == begin) return false;
    for (std::size_t i = begin; i < end; ++i) {
        const char ch = word[i];
        const char lowered = ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch + 32) : ch;
        if (I.char_vocab.find(std::string(1, lowered)) == I.char_vocab.end()) {
            return false;
        }
    }
    return true;
}

namespace {

struct Beam {
    std::vector<int>   tokens;
    double             score = 0.0;
    std::vector<float> h;        // [2H] layer-major, ready for an in_h column
    std::vector<float> c;
    bool               finished = false;

    double normalized_score() const noexcept {
        // Port of _Beam.normalized_score: cumulative log probability divided by
        // the generated length (BOS excluded, EOS included).
        return score / static_cast<double>(std::max<std::size_t>(1, tokens.size() - 1));
    }
};

}  // namespace

std::vector<std::vector<std::string>> LstmG2p::predict(const std::string & word,
                                                       int beam_size) const {
    const Impl & I = *impl_;
    const int beam = beam_size > 0 ? beam_size : I.cfg.beam_size;
    const int H = I.cfg.hidden;
    const int V = I.cfg.num_phonemes;

    // The reference lower-cases and strips before the char lookup; characters
    // outside the vocabulary (which is ASCII-only) fall to <unk> either way.
    std::string w;
    std::size_t begin = 0, end = word.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(word[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(word[end - 1]))) --end;
    w.reserve(end - begin);
    for (std::size_t i = begin; i < end; ++i) {
        const char ch = word[i];
        w.push_back(ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch + 32) : ch);
    }

    std::vector<int> src;
    src.reserve(w.size() + 2);
    src.push_back(I.cfg.phoneme_bos);
    for (const char ch : w) {
        const auto it = I.char_vocab.find(std::string(1, ch));
        src.push_back(it == I.char_vocab.end() ? I.cfg.char_unk : it->second);
    }
    src.push_back(I.cfg.phoneme_eos);
    const int T = static_cast<int>(src.size());

    // ---- encoder ----------------------------------------------------------
    std::unique_ptr<LstmG2p::Impl::EncoderGraph> enc = I.build_encoder(T);
    ggml_backend_tensor_set(enc->in_ids, src.data(), 0, src.size() * sizeof(int32_t));
    ggml_backend_graph_compute(I.backend, enc->graph);

    std::vector<float> enc_out(static_cast<std::size_t>(T) * H);
    std::vector<float> h_init(static_cast<std::size_t>(2) * H);
    std::vector<float> c_init(static_cast<std::size_t>(2) * H);
    ggml_backend_tensor_get(enc->enc_out, enc_out.data(), 0, enc_out.size() * sizeof(float));
    ggml_backend_tensor_get(enc->h_init, h_init.data(), 0, h_init.size() * sizeof(float));
    ggml_backend_tensor_get(enc->c_init, c_init.data(), 0, c_init.size() * sizeof(float));
    enc.reset();   // free the encoder graph before decoding

    // ---- decoder ----------------------------------------------------------
    const int B = beam;
    std::unique_ptr<LstmG2p::Impl::DecoderGraph> dec = I.build_decoder(T, B);
    ggml_backend_tensor_set(dec->in_enc, enc_out.data(), 0, enc_out.size() * sizeof(float));

    std::vector<Beam> beams;
    Beam first;
    first.tokens = { I.cfg.phoneme_bos };
    first.h = h_init;
    first.c = c_init;
    beams.push_back(std::move(first));

    const std::size_t state2H = static_cast<std::size_t>(2) * H;
    std::vector<int>   tok(static_cast<std::size_t>(B));
    std::vector<float> h_in(static_cast<std::size_t>(B) * state2H);
    std::vector<float> c_in(static_cast<std::size_t>(B) * state2H);
    std::vector<float> logits(static_cast<std::size_t>(B) * V);
    std::vector<float> log_probs(static_cast<std::size_t>(V));
    std::vector<float> h_out(static_cast<std::size_t>(B) * state2H);
    std::vector<float> c_out(static_cast<std::size_t>(B) * state2H);

    for (int step = 0; step < I.cfg.max_len; ++step) {
        const bool all_done =
            std::all_of(beams.begin(), beams.end(),
                        [](const Beam & b) { return b.finished; });
        if (all_done) break;

        // Pack the inputs; columns beyond the active beams duplicate the last
        // one (the graph is fixed-shape and their results are never read).
        const std::size_t n_active = beams.size();
        for (std::size_t b = 0; b < static_cast<std::size_t>(B); ++b) {
            const Beam & src_beam = beams[std::min(b, n_active - 1)];
            tok[b] = src_beam.tokens.back();
            std::memcpy(h_in.data() + b * state2H, src_beam.h.data(),
                        state2H * sizeof(float));
            std::memcpy(c_in.data() + b * state2H, src_beam.c.data(),
                        state2H * sizeof(float));
        }
        ggml_backend_tensor_set(dec->in_tok, tok.data(), 0, tok.size() * sizeof(int32_t));
        ggml_backend_tensor_set(dec->in_h, h_in.data(), 0, h_in.size() * sizeof(float));
        ggml_backend_tensor_set(dec->in_c, c_in.data(), 0, c_in.size() * sizeof(float));
        ggml_backend_graph_compute(I.backend, dec->graph);
        ggml_backend_tensor_get(dec->out_logits, logits.data(), 0, logits.size() * sizeof(float));
        ggml_backend_tensor_get(dec->out_h, h_out.data(), 0, h_out.size() * sizeof(float));
        ggml_backend_tensor_get(dec->out_c, c_out.data(), 0, c_out.size() * sizeof(float));


        std::vector<Beam> candidates;
        candidates.reserve(n_active * static_cast<std::size_t>(std::min(beam, V)));
        for (std::size_t bi = 0; bi < n_active; ++bi) {
            const Beam & b = beams[bi];
            if (b.finished) {
                candidates.push_back(b);
                continue;
            }
            const float * col = logits.data() + bi * V;

            // log-softmax in double, exactly like the reference
            const float l_max = *std::max_element(col, col + V);
            double sum = 0.0;
            for (int v = 0; v < V; ++v) {
                sum += std::exp(static_cast<double>(col[v]) - l_max);
            }
            const double lse = std::log(sum) + l_max;
            for (int v = 0; v < V; ++v) {
                log_probs[static_cast<std::size_t>(v)] =
                    static_cast<float>(static_cast<double>(col[v]) - lse);
            }

            std::vector<int> ids(static_cast<std::size_t>(V));
            std::iota(ids.begin(), ids.end(), 0);
            std::stable_sort(ids.begin(), ids.end(), [&](int a, int b_id) {
                return log_probs[static_cast<std::size_t>(a)] >
                       log_probs[static_cast<std::size_t>(b_id)];
            });

            const int keep = std::min(beam, V);
            for (int k = 0; k < keep; ++k) {
                const int id = ids[static_cast<std::size_t>(k)];
                Beam next;
                next.tokens = b.tokens;
                next.tokens.push_back(id);
                next.score    = b.score + log_probs[static_cast<std::size_t>(id)];
                next.h.assign(h_out.begin() + static_cast<std::ptrdiff_t>(bi * state2H),
                              h_out.begin() + static_cast<std::ptrdiff_t>((bi + 1) * state2H));
                next.c.assign(c_out.begin() + static_cast<std::ptrdiff_t>(bi * state2H),
                              c_out.begin() + static_cast<std::ptrdiff_t>((bi + 1) * state2H));
                next.finished = id == I.cfg.phoneme_eos;
                candidates.push_back(std::move(next));
            }
        }

        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const Beam & a, const Beam & b) {
                             return a.normalized_score() > b.normalized_score();
                         });
        if (static_cast<int>(candidates.size()) > beam) candidates.resize(beam);
        beams = std::move(candidates);
    }

    // Order-preserving dedup of the produced pronunciations (dict.fromkeys in
    // the reference): special tokens are dropped, distinct readings keep their
    // beam rank.
    std::vector<std::vector<std::string>> out;
    for (const Beam & b : beams) {
        std::vector<std::string> pron;
        for (std::size_t k = 1; k < b.tokens.size(); ++k) {
            const int id = b.tokens[k];
            if (id == I.cfg.phoneme_unk || id == I.cfg.phoneme_pad ||
                id == I.cfg.phoneme_bos || id == I.cfg.phoneme_eos) {
                continue;
            }
            if (id >= 0 && id < static_cast<int>(I.phoneme_list.size()) &&
                !I.phoneme_list[static_cast<std::size_t>(id)].empty()) {
                pron.push_back(I.phoneme_list[static_cast<std::size_t>(id)]);
            }
        }
        if (std::find(out.begin(), out.end(), pron) == out.end()) {
            out.push_back(std::move(pron));
        }
    }
    return out;
}

}  // namespace tifa_ggml::internal::g2p
