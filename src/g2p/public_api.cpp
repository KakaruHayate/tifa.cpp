// Public G2P / PFML API — see include/tifa_ggml/g2p.h.
//
// A thin adapter over the internal g2p module: the public value types are
// structurally identical to the internal ones, so this file only converts
// between them.  Keeping the adapter separate means the internal headers stay
// free to change without breaking downstream consumers.

#include "tifa_ggml/g2p.h"

#include "g2p/g2p.h"
#include "g2p/pfml.h"

#include <utility>

namespace tifa_ggml {

namespace {

namespace ig2p = internal::g2p;

G2PWord to_public(const ig2p::Word & word) {
    G2PWord out;
    out.text     = word.text;
    out.language = word.language;
    out.readings.reserve(word.readings.size());
    for (const ig2p::Reading & reading : word.readings) {
        G2PReading converted;
        converted.paths.reserve(reading.paths.size());
        for (const ig2p::Path & path : reading.paths) {
            G2PPath converted_path;
            converted_path.reserve(path.size());
            for (const ig2p::Group & group : path) {
                G2PGroup converted_group;
                converted_group.script   = group.script;
                converted_group.phonemes = group.phonemes;
                converted_path.push_back(std::move(converted_group));
            }
            converted.paths.push_back(std::move(converted_path));
        }
        out.readings.push_back(std::move(converted));
    }
    return out;
}

ig2p::Word to_internal(const G2PWord & word) {
    ig2p::Word out;
    out.text     = word.text;
    out.language = word.language;
    out.readings.reserve(word.readings.size());
    for (const G2PReading & reading : word.readings) {
        ig2p::Reading converted;
        converted.paths.reserve(reading.paths.size());
        for (const G2PPath & path : reading.paths) {
            ig2p::Path converted_path;
            converted_path.reserve(path.size());
            for (const G2PGroup & group : path) {
                ig2p::Group converted_group;
                converted_group.script   = group.script;
                converted_group.phonemes = group.phonemes;
                converted_path.push_back(std::move(converted_group));
            }
            converted.paths.push_back(std::move(converted_path));
        }
        out.readings.push_back(std::move(converted));
    }
    return out;
}

std::string join_with_spaces(const std::vector<std::string> & parts) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) out.push_back(' ');
        out += parts[i];
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// G2PPipeline
// ---------------------------------------------------------------------------

struct G2PPipeline::Impl {
    explicit Impl(ig2p::Pipeline value) : pipeline(std::move(value)) {}
    ig2p::Pipeline pipeline;
};

G2PPipeline::G2PPipeline() = default;
G2PPipeline::~G2PPipeline() = default;
G2PPipeline::G2PPipeline(G2PPipeline &&) noexcept = default;
G2PPipeline & G2PPipeline::operator=(G2PPipeline &&) noexcept = default;

G2PPipeline G2PPipeline::from_config(const std::string & g2p_json,
                                     const std::string & dict_dir) {
    G2PPipeline result;
    result.impl_ = std::make_unique<Impl>(ig2p::Pipeline::from_config(g2p_json, dict_dir));
    return result;
}

std::vector<G2PWord> G2PPipeline::convert(const std::string & text,
                                          const std::vector<std::string> & languages) const {
    if (!impl_) return {};
    std::vector<G2PWord> out;
    for (const ig2p::Word & word : impl_->pipeline.convert(text, languages)) {
        out.push_back(to_public(word));
    }
    return out;
}

std::vector<G2PWord> G2PPipeline::convert_pfml(const std::string & text,
                                               const std::vector<std::string> & languages) const {
    if (!impl_) return {};
    std::vector<G2PWord> out;
    for (const ig2p::Word & word : ig2p::convert_pfml(impl_->pipeline, text, languages)) {
        out.push_back(to_public(word));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

bool looks_like_pfml(const std::string & text) {
    return ig2p::looks_like_pfml(text);
}

void validate_pfml(const std::string & text) {
    ig2p::validate_pfml(text);
}

std::string to_pfml(const std::vector<G2PWord> & words) {
    std::vector<ig2p::Word> internal;
    internal.reserve(words.size());
    for (const G2PWord & word : words) internal.push_back(to_internal(word));
    return ig2p::to_pfml(internal);
}

std::vector<G2PWordCandidates> candidates(const G2PPipeline & pipeline,
                                          const std::string & text,
                                          const std::vector<std::string> & languages) {
    std::vector<G2PWordCandidates> out;
    for (const G2PWord & word : pipeline.convert(text, languages)) {
        G2PWordCandidates entry;
        entry.text     = word.text;
        entry.language = word.language;
        for (std::size_t r = 0; r < word.readings.size(); ++r) {
            const std::vector<G2PPath> & paths = word.readings[r].paths;
            for (std::size_t p = 0; p < paths.size(); ++p) {
                G2PCandidate candidate;
                candidate.reading = static_cast<int>(r);
                candidate.path    = static_cast<int>(p);
                std::vector<std::string> scripts;
                for (const G2PGroup & group : paths[p]) {
                    if (!group.script.empty()) scripts.push_back(group.script);
                    candidate.phonemes.insert(candidate.phonemes.end(),
                                              group.phonemes.begin(),
                                              group.phonemes.end());
                }
                candidate.script = join_with_spaces(scripts);
                entry.candidates.push_back(std::move(candidate));
            }
        }
        out.push_back(std::move(entry));
    }
    return out;
}

ResolvedPhoneme resolve_phoneme(const std::string & phoneme,
                                const std::vector<std::string> & languages,
                                const G2PLookup & lookup,
                                const std::vector<std::string> & global_symbols,
                                const std::vector<std::string> & stop_symbols) {
    const ig2p::ResolvedSymbol resolved =
        ig2p::resolve_phoneme(phoneme, languages, lookup, global_symbols, stop_symbols);
    ResolvedPhoneme out;
    out.id     = resolved.id;
    out.symbol = resolved.symbol;
    return out;
}

}  // namespace tifa_ggml
