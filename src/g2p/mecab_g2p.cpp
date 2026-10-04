#include "mecab_g2p.h"

#include <mecab.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "tifa_ggml/errors.h"

namespace tifa_ggml::internal::g2p {

namespace {

// UniDic's `pron` column.  The dicrc shipped with the dictionary documents
// the feature layout: f[0..3] POS, f[4] cType, f[5] cForm, f[6] lForm,
// f[7] lemma, f[8] orth, f[9] pron.  Quoted CSV fields (aType/aConType)
// only appear past index 22, so a plain comma split is safe here.
constexpr int k_pron_feature_index = 9;

std::string csv_field(const std::string & feature, int index) {
    int current = 0;
    std::size_t begin = 0;
    while (true) {
        const std::size_t comma = feature.find(',', begin);
        const std::string field =
            comma == std::string::npos ? feature.substr(begin)
                                       : feature.substr(begin, comma - begin);
        if (current == index) return field;
        if (comma == std::string::npos) return std::string();
        begin = comma + 1;
        ++current;
    }
}

}  // namespace

struct MecabTagger::Impl {
    MeCab::Tagger * tagger = nullptr;
    ~Impl() { delete tagger; }
};

MecabTagger::MecabTagger() = default;

MecabTagger::~MecabTagger() = default;

std::unique_ptr<MecabTagger> MecabTagger::open(const std::string & dicdir) {
    std::string rc  = dicdir + "/mecabrc";
    std::string dic = dicdir;
    std::string r_flag = "-r", d_flag = "-d";
    char           prog[] = "tifa_ggml";
    // argv-style options: MeCab's string form does not strip quotes, so a
    // dicdir containing spaces would only survive this spelling.
    char * argv[] = { prog, &r_flag[0], &rc[0], &d_flag[0], &dic[0] };

    auto tagger = std::unique_ptr<MecabTagger>(new MecabTagger());
    tagger->impl_ = std::make_unique<Impl>();
    tagger->impl_->tagger = MeCab::Tagger::create(5, argv);
    if (tagger->impl_->tagger == nullptr) {
        std::string reason = MeCab::getTaggerError();
        if (reason.empty() || reason == "Unknown Error") {
            reason = "no usable MeCab dictionary in '" + dicdir +
                     "' (need mecabrc, sys.dic, matrix.bin, char.bin, unk.dic)";
        }
        throw InvalidArgument("cannot load MeCab dictionary from '" + dicdir +
                              "': " + reason);
    }
    return tagger;
}

std::vector<MecabTagger::Morph> MecabTagger::segment(const std::string & text) const {
    std::vector<Morph> morphs;
    const MeCab::Node * node = impl_->tagger->parseToNode(text.c_str());
    if (node == nullptr) {
        throw InvalidArgument("MeCab failed to segment '" + text + "': " +
                              MeCab::getTaggerError());
    }
    for (node = node->next; node != nullptr; node = node->next) {
        if (node->stat == MECAB_EOS_NODE) break;
        morphs.push_back(Morph{ std::string(node->surface, node->length),
                                std::string(node->feature) });
    }
    return morphs;
}

std::vector<std::string> MecabTagger::pronunciations(const std::string & word,
                                                     int nbest) const {
    std::vector<std::string> readings;
    MeCab::Lattice * lattice = MeCab::createLattice();
    if (lattice == nullptr) {
        throw InvalidArgument("MeCab: cannot create a lattice for '" + word + "'");
    }
    lattice->add_request_type(MECAB_NBEST);
    lattice->set_sentence(word.c_str());
    if (!impl_->tagger->parse(lattice)) {
        const std::string reason = lattice->what();  // before the lattice is freed
        MeCab::deleteLattice(lattice);
        throw InvalidArgument("MeCab failed to parse '" + word + "': " + reason);
    }
    for (int i = 0; i < nbest && lattice->next(); ++i) {
        // Upstream keeps only N-best parses where the word is one morpheme:
        // those are alternative whole-word readings, not accidental splits.
        std::size_t nodes    = 0;
        bool        single   = true;
        std::string pron;
        const MeCab::Node * bos = lattice->bos_node();
        for (const MeCab::Node * node = bos != nullptr ? bos->next : nullptr;
             node != nullptr; node = node->next) {
            if (node->stat == MECAB_EOS_NODE) break;
            if (++nodes > 1) { single = false; break; }
            if (std::string(node->surface, node->length) != word) {
                single = false;
                break;
            }
            pron = csv_field(std::string(node->feature), k_pron_feature_index);
        }
        if (!single || pron.empty() || pron == "*" || pron == "-") continue;
        if (std::find(readings.begin(), readings.end(), pron) == readings.end()) {
            readings.push_back(std::move(pron));
        }
    }
    MeCab::deleteLattice(lattice);
    return readings;
}

}  // namespace tifa_ggml::internal::g2p
