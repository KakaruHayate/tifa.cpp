// MeCab tagger wrapper for the `japanese-mecab` G2P converter.
//
// Port of g2pflow's JapaneseMecabConverter backend (fugashi + UniDic) as a
// thin RAII shell over MeCab's C++ API:
//   * `segment()`       — what fugashi does with `tagger(text)`
//   * `pronunciations()` — what fugashi does with `tagger.nbestToNodeList`,
//     keeping only parses where the whole word stays a single morpheme and
//     reading UniDic's `pron` column (the kana reading).
//
// One instance is not safe for concurrent use (MeCab's tagger holds a mutable
// lattice); the CLI drives it sequentially.
//
// The compiled UniDic dictionary is external data (shipped as a separate
// release asset, ~260 MB extracted); `open()` expects the dicdir that holds
// mecabrc, sys.dic, matrix.bin, char.bin and unk.dic.
#ifndef TIFA_GGML_G2P_MECAB_H_
#define TIFA_GGML_G2P_MECAB_H_

#include <memory>
#include <string>
#include <vector>

namespace tifa_ggml::internal::g2p {

class MecabTagger {
public:
    struct Morph {
        std::string surface;   // UTF-8, not NUL-terminated in MeCab -- copied here
        std::string feature;   // raw UniDic CSV feature string
    };

    // Throws InvalidArgument (with MeCab's message) when the dictionary
    // cannot be loaded -- missing dicdir, missing mecabrc/sys.dic, charset
    // mismatch, ...
    static std::unique_ptr<MecabTagger> open(const std::string & dicdir);

    ~MecabTagger();
    MecabTagger(const MecabTagger &)            = delete;
    MecabTagger & operator=(const MecabTagger &) = delete;

    // Segment a UTF-8 sentence into morphemes (BOS/EOS dropped).  Callers
    // re-join kana digraphs if they care (see JapaneseMecabConverter).
    std::vector<Morph> segment(const std::string & text) const;

    // Distinct UniDic pronunciations of `word`, in N-best order and capped at
    // `nbest` lattice walks: parses where the whole word stays a single
    // morpheme contribute their `pron` (feature field 9); multi-morpheme
    // parses are skipped, exactly like upstream.  `*`/`-` pron fields are
    // dropped, duplicates de-duplicated.
    std::vector<std::string> pronunciations(const std::string & word, int nbest) const;

private:
    MecabTagger();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tifa_ggml::internal::g2p

#endif  // TIFA_GGML_G2P_MECAB_H_
