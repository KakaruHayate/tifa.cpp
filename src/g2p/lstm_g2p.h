#pragma once

// English LSTM G2P (port of openvpi/TIFA's g2p/converters/lstm.py).
//
// The converter is dictionary-backed with an encoder/decoder LSTM for
// out-of-vocabulary words.  The weights come from a GGUF file produced by
// scripts/convert_lstm_g2p_to_gguf.py (architecture "lstm-g2p"); inference
// runs as ggml graphs through GGML_OP_LSTM (cmake/patches/ggml-lstm-op.md) on
// a dedicated CPU backend — the net is small (~4M parameters) and its
// sequential beam loop gives a GPU nothing to do.
//
// Internal API (not exported from the library).

#include "tifa_ggml/errors.h"

#include <memory>
#include <string>
#include <vector>

namespace tifa_ggml::internal::g2p {

class LstmG2p {
public:
    // Loads the model, hyper-parameters and both vocabularies from a
    // lstm-g2p GGUF.  Throws GgufError / InvalidArgument on a malformed file.
    static LstmG2p from_file(const std::string & gguf_path);

    LstmG2p(LstmG2p &&) noexcept;
    LstmG2p & operator=(LstmG2p &&) noexcept;
    ~LstmG2p();
    LstmG2p(const LstmG2p &) = delete;
    LstmG2p & operator=(const LstmG2p &) = delete;

    // Phoneme strings of one pronunciation, ranked best first; distinct
    // readings only (port of LSTMConverter._predict, including its beam
    // search and the dedup that collapses equal pronunciations).
    // `beam_size <= 0` selects the value recorded in the GGUF.
    std::vector<std::vector<std::string>> predict(const std::string & word,
                                                  int beam_size = 0) const;

    // Number of rows of the phoneme vocabulary (the decoder's output space).
    int num_phonemes() const noexcept;

    // Python `find` gate: true when every (lower-cased, stripped) character of
    // `word` is inside the char vocabulary, i.e. the encoder can represent it
    // without an implicit <unk>.  Callers use it to decide whether a word that
    // misses the dictionary can still be inferred by the network.
    bool can_encode(const std::string & word) const;

private:
    LstmG2p();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tifa_ggml::internal::g2p
