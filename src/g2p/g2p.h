#pragma once

// Grapheme-to-phoneme conversion (port of openvpi/TIFA's `g2p` package).
//
// The pipeline mirrors g2p/pipeline.py: a list of preprocessors produces text
// fragments, each fragment is routed to the first converter that claims a
// substring, and the claimed substrings are converted into G2PWords holding
// alternative pronunciations.
//
// Internal API (not exported from the library).

#include "tifa_ggml/errors.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tifa_ggml::internal::g2p {

// ---------------------------------------------------------------------------
// Data model (mirrors g2p/converters/base.py)
// ---------------------------------------------------------------------------

struct Group {
    std::string              script;     // e.g. "zhong" (pinyin) or the raw word
    std::vector<std::string> phonemes;   // e.g. {"zh", "ong"}
};

using Path = std::vector<Group>;

struct Reading {
    std::vector<Path> paths;             // alternative complete pronunciations
};

struct Word {
    std::string         text;
    std::string         language;        // resolved language tag ("" = any)
    std::vector<Reading> readings;
};

// ---------------------------------------------------------------------------
// Converters
// ---------------------------------------------------------------------------

// Substring match: [begin, end) within `text`, or false when the converter
// cannot handle anything in `text`.
struct Match {
    bool        ok = false;
    std::size_t begin = 0;
    std::size_t end   = 0;
};

class Converter {
public:
    virtual ~Converter() = default;
    // Language tags this converter serves; empty = any language.
    virtual const std::vector<std::string> & languages() const = 0;
    // Locate a claimed run inside `text` (UTF-8 codepoint indices).
    virtual Match find(const std::u32string & text) const = 0;
    // Convert a claimed run into words.
    virtual std::vector<Word> convert(const std::u32string & text) const = 0;
    // Converter-local preprocessors (applied to the claimed run).
    virtual std::vector<std::string> preprocess(const std::string & text) const { return { text }; }
};

// ---------------------------------------------------------------------------
// Pinyin engine (port of g2p/converters/cpp_pinyin)
// ---------------------------------------------------------------------------

// Faithful port of cpp-pinyin's EngineImpl: dictionary-driven phrase
// disambiguation over hanzi input.
class PinyinEngine {
public:
    // `dict_dir` must contain word.txt, phrases_dict.txt, phrases_map.txt,
    // trans_word.txt and (optionally) user_dict.txt.
    explicit PinyinEngine(const std::string & dict_dir);

    bool ready() const noexcept;

    // Traditional -> simplified for each character.
    std::vector<char32_t> simplify(const std::vector<char32_t> & chars) const;

    // Port of EngineImpl::queryRaw() with STYLE_NORMAL output (tone digits
    // stripped, which is all the released converter uses).
    std::vector<std::vector<std::string>> query_raw(const std::vector<char32_t> & chars) const;

    // All readings of one character (STYLE_NORMAL), or {ch} when unknown.
    std::vector<std::string> readings(char32_t ch) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

class Pipeline {
public:
    // Build from the JSON embedded in the GGUF (`tifa.vocab.json` → "g2p"),
    // which follows configs/g2p.yaml.  `dict_dir` resolves "@dictionaries/..."
    // style paths.
    static Pipeline from_config(const std::string & g2p_json,
                                const std::string & dict_dir);

    ~Pipeline();
    Pipeline(Pipeline &&) noexcept;
    Pipeline & operator=(Pipeline &&) noexcept;

    // Convert text into words.  `languages` is the ordered allow-list from the
    // command line (empty = every converter is active).
    std::vector<Word> convert(const std::string & text,
                              const std::vector<std::string> & languages) const;

private:
    Pipeline();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Encoding (g2p/encoding.py)
// ---------------------------------------------------------------------------

// Token resolution result: id (or -1) plus the symbol that was looked up.
struct ResolvedSymbol {
    int         id = -1;
    std::string symbol;
};

// Resolve one phoneme against a vocabulary: literal first, then
// "<language>/<phoneme>" for each language in order.
ResolvedSymbol resolve_phoneme(
    const std::string & phoneme,
    const std::vector<std::string> & languages,
    const std::function<int(const std::string &)> & lookup,
    const std::vector<std::string> & global_symbols,
    const std::vector<std::string> & stop_symbols);

// Candidate grid, laid out exactly like encode_paths()'s output:
//   paths[P][C]      token id per grid row/column (0 = gap or padding)
//   groups[P][C]     1-based group id per row/column (0 = gap)
//   words[P]         1-based word id per row (0 = padding)
//   candidates[W][C] valid-candidate mask (a valid empty path has a true row)
//   lexicon           per word: the candidate metadata (labels + scripts)
struct CandidateGrid {
    std::vector<int>  paths;        // [P * C]
    std::vector<int>  groups;       // [P * C]
    std::vector<int>  words;        // [P]
    std::vector<char> candidates;   // [W * C]
    int P = 0, C = 0, W = 0;

    struct Candidate {
        int                      reading = 0;
        std::vector<std::string> phonemes;
        std::vector<std::string> scripts;
    };
    std::vector<std::vector<Candidate>> lexicon;   // [W]
    std::vector<std::string>            texts;     // [W]
};

// Port of g2p/encoding.py:encode_paths.  `oov_handling` is raise|discard|force.
// Throws G2PEncodingError (an InvalidArgument subclass) for OOV words whose
// handling is "raise".
struct G2PEncodingError : InvalidArgument {
    explicit G2PEncodingError(const std::string & msg) : InvalidArgument(msg) {}
};

CandidateGrid encode_paths(
    const std::vector<Word> & words,
    const std::function<int(const std::string &)> & lookup,
    const std::vector<std::string> & global_symbols,
    const std::vector<std::string> & stop_symbols,
    const std::vector<std::string> & languages,
    const std::string & oov_handling,
    int num_reserved_tokens);

// Levenshtein multi-sequence alignment (lib/levenshtein.py).
// Returns, for each path, its row in the shared profile: `profile[row][col]`
// is the token (or -1 for a gap).
std::vector<std::vector<int>> align_multiple_sequences(const std::vector<std::vector<int>> & paths);

}  // namespace tifa_ggml::internal::g2p
