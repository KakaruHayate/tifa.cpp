#pragma once

// Public C++ API for grapheme-to-phoneme conversion and PFML
// (Pronunciation Flow Markup Language, openvpi/g2pflow) handling.
//
// This is the ggml-free subset of tifa_ggml: it is what the standalone
// `tifa_ggml_g2p` target provides, and what downstream editors (MaxLabel) link
// against when they need G2P and PFML but not the aligner model.  The English
// OOV LSTM converter needs ggml and is only present in the full build.

#include "tifa_ggml/errors.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tifa_ggml {

// ---------------------------------------------------------------------------
// Data model — mirrors g2pflow: word -> readings -> paths -> groups -> phonemes
// ---------------------------------------------------------------------------

struct G2PGroup {
    std::string              script;     // e.g. "zhong"; filled from the phonemes when empty
    std::vector<std::string> phonemes;   // e.g. {"zh", "ong"}
};

using G2PPath = std::vector<G2PGroup>;

struct G2PReading {
    std::vector<G2PPath> paths;          // alternative complete pronunciations
};

struct G2PWord {
    std::string                 text;
    std::string                 language;   // resolved tag ("" = any)
    std::vector<G2PReading>     readings;
};

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

// Vocabulary lookup: the token id for a symbol, or -1 when it is unknown.
using G2PLookup = std::function<int(const std::string &)>;

class G2PPipeline {
public:
    // Build from the JSON embedded in a TIFA GGUF (`tifa.vocab.json` -> "g2p"),
    // which follows configs/g2p.yaml.  `dict_dir` resolves "@dictionaries/..."
    // style relative paths.  An empty JSON builds an inert pipeline (no
    // preprocessors, no converters) that still handles direct-phoneme PFML.
    static G2PPipeline from_config(const std::string & g2p_json,
                                   const std::string & dict_dir);

    G2PPipeline(G2PPipeline &&) noexcept;
    G2PPipeline & operator=(G2PPipeline &&) noexcept;
    ~G2PPipeline();

    // Plain text -> words.  `languages` is the ordered allow-list from the
    // command line (empty = every converter is active).
    std::vector<G2PWord> convert(const std::string & text,
                                 const std::vector<std::string> & languages) const;

    // PFML fragment -> words.
    std::vector<G2PWord> convert_pfml(const std::string & text,
                                      const std::vector<std::string> & languages) const;

private:
    G2PPipeline();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// True when `text` contains a PFML element tag (`<scope`, `<word`, `<reading`,
// `<path`, `<group` or `<phoneme`).  Plain prose with a stray '<' is not PFML.
bool looks_like_pfml(const std::string & text);

// Parse-only check of a PFML fragment: throws InvalidArgument on malformed
// markup and returns normally otherwise.  No pipeline, converters or
// vocabulary are involved, so an editor can gate its exports on this without
// loading a model — the aligner skips a sample whose PFML does not parse, with
// no fallback, which makes this the difference between a typo and a lost line.
void validate_pfml(const std::string & text);

// Words -> PFML fragment.  A word with a single reading / path / group uses the
// compact form (`<word text="重" language="zh" script="zhong"
// phonemes="zh ong"/>`); anything carrying alternatives emits the explicit
// <reading>/<path>/<group>/<phoneme> tree.
//
// Round-trip guarantee: convert_pfml(to_pfml(words)) reproduces `words` — the
// data model, not the original spelling or attribute order (the same contract
// as g2pflow's to_pfml).
std::string to_pfml(const std::vector<G2PWord> & words);

// ---------------------------------------------------------------------------
// Pronunciation candidates and phoneme validation
// ---------------------------------------------------------------------------

// One pronunciation the pipeline offers for a word: a (reading, path) pair.
struct G2PCandidate {
    int                      reading = 0;   // index into the word's readings
    int                      path    = 0;   // index into that reading's paths
    std::string              script;        // group scripts joined with spaces
    std::vector<std::string> phonemes;      // groups flattened, in order
};

struct G2PWordCandidates {
    std::string               text;
    std::string               language;
    std::vector<G2PCandidate> candidates;
};

// Every pronunciation the pipeline offers, per word — this is what a
// pronunciation picker shows.  A word with no alternatives still yields one
// candidate, so callers can treat "ambiguous" as candidates.size() > 1.
std::vector<G2PWordCandidates> candidates(const G2PPipeline & pipeline,
                                          const std::string & text,
                                          const std::vector<std::string> & languages);

// A phoneme resolved against the model vocabulary.
struct ResolvedPhoneme {
    int         id = -1;    // -1 = not in the vocabulary
    std::string symbol;     // the symbol that was looked up
};

// Resolve one phoneme: literal first, then "<language>/<phoneme>" for each
// language in order, then the global and stop symbol sets.  `id == -1` means
// the editor should flag the phoneme as invalid.
ResolvedPhoneme resolve_phoneme(const std::string & phoneme,
                                const std::vector<std::string> & languages,
                                const G2PLookup & lookup,
                                const std::vector<std::string> & global_symbols = {},
                                const std::vector<std::string> & stop_symbols = {});

}  // namespace tifa_ggml
