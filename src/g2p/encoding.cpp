// Vocabulary encoding: candidate grids and multi-path alignment.
//
// Port of openvpi/TIFA's `g2p/encoding.py` and `lib/levenshtein.py`.  Each
// word's pronunciation candidates are aligned into a shared profile, and the
// profiles are stacked into the row-major grids the token classifier consumes.

#include "g2p.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tifa_ggml::internal::g2p {

namespace {

// Alignment gap marker; tokens are always >= NUM_RESERVED_TOKENS.
constexpr int kGap = -1;

bool contains(const std::vector<std::string> & values, const std::string & value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

// lib/vocabulary.py:qualify_symbol; empty `default_language` means "none".
std::string qualify_symbol(const std::string & symbol,
                           const std::string & default_language,
                           const std::vector<std::string> & global_symbols) {
    if (!contains(global_symbols, symbol) && symbol.find('/') == std::string::npos &&
        !default_language.empty()) {
        return default_language + "/" + symbol;
    }
    return symbol;
}

std::string format_phonemes(const std::vector<std::string> & phonemes) {
    std::string out = "[";
    for (std::size_t i = 0; i < phonemes.size(); ++i) {
        if (i != 0) out += ", ";
        out += "'" + phonemes[i] + "'";
    }
    return out + "]";
}

// ---------------------------------------------------------------------------
// levenshtein alignment (lib/levenshtein.py)
// ---------------------------------------------------------------------------

// Levenshtein DP: cost 0 for equality, 1 for substitution.  The backtrack
// prefers match over delete over insert.
std::pair<std::vector<int>, std::vector<int>> align_sequences(
    const std::vector<int> & orig, const std::vector<int> & mut) {
    const std::size_t n = orig.size();
    const std::size_t m = mut.size();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 0; i <= n; ++i) dp[i][0] = static_cast<int>(i);
    for (std::size_t j = 0; j <= m; ++j) dp[0][j] = static_cast<int>(j);
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= m; ++j) {
            const int cost       = orig[i - 1] == mut[j - 1] ? 0 : 1;
            const int match_val  = dp[i - 1][j - 1] + cost;
            const int delete_val = dp[i - 1][j] + 1;
            const int insert_val = dp[i][j - 1] + 1;
            if (match_val <= delete_val && match_val <= insert_val) {
                dp[i][j] = match_val;
            } else if (delete_val <= insert_val) {
                dp[i][j] = delete_val;
            } else {
                dp[i][j] = insert_val;
            }
        }
    }

    std::vector<int> aligned_orig;
    std::vector<int> aligned_mut;
    std::size_t i = n;
    std::size_t j = m;
    while (i > 0 || j > 0) {
        bool match_ok = false;
        if (i > 0 && j > 0) {
            const int cost = orig[i - 1] == mut[j - 1] ? 0 : 1;
            match_ok       = dp[i][j] == dp[i - 1][j - 1] + cost;
        }
        if (match_ok) {
            aligned_orig.push_back(orig[i - 1]);
            aligned_mut.push_back(mut[j - 1]);
            --i;
            --j;
        } else if (i > 0 && dp[i][j] == dp[i - 1][j] + 1) {
            aligned_orig.push_back(orig[i - 1]);
            aligned_mut.push_back(kGap);
            --i;
        } else {
            aligned_orig.push_back(kGap);
            aligned_mut.push_back(mut[j - 1]);
            --j;
        }
    }
    std::reverse(aligned_orig.begin(), aligned_orig.end());
    std::reverse(aligned_mut.begin(), aligned_mut.end());
    return { std::move(aligned_orig), std::move(aligned_mut) };
}

// Column-wise consensus of a row-oriented profile; the first token reaching
// the highest count wins, matching Python's dict-ordered max().
std::vector<int> build_consensus(const std::vector<std::vector<int>> & rows) {
    std::vector<int> consensus;
    if (rows.empty()) return consensus;
    const std::size_t columns = rows[0].size();
    consensus.reserve(columns);
    for (std::size_t column = 0; column < columns; ++column) {
        std::vector<std::pair<int, int>> counts;             // token, count
        for (const std::vector<int> & row : rows) {
            const int token = row[column];
            if (token == kGap) continue;
            auto it = std::find_if(counts.begin(), counts.end(),
                                   [token](const std::pair<int, int> & entry) {
                                       return entry.first == token;
                                   });
            if (it == counts.end()) {
                counts.emplace_back(token, 1);
            } else {
                ++it->second;
            }
        }
        if (counts.empty()) {
            consensus.push_back(kGap);
            continue;
        }
        int best_token = counts[0].first;
        int best_count = counts[0].second;
        for (const std::pair<int, int> & entry : counts) {
            if (entry.second > best_count) {
                best_token = entry.first;
                best_count = entry.second;
            }
        }
        consensus.push_back(best_token);
    }
    return consensus;
}

// Aligns `new_path` to the profile and merges it in; every source row is
// preserved, including duplicates.
std::vector<std::vector<int>> merge_profile(const std::vector<std::vector<int>> & rows,
                                            const std::vector<int> & new_path) {
    const std::vector<int> consensus = build_consensus(rows);
    const auto aligned = align_sequences(consensus, new_path);
    const std::vector<int> & aligned_consensus = aligned.first;

    std::vector<std::vector<int>> merged;
    merged.reserve(rows.size() + 1);
    for (const std::vector<int> & old_row : rows) {
        std::vector<int> new_row;
        new_row.reserve(aligned_consensus.size());
        std::size_t old_index = 0;
        for (int token : aligned_consensus) {
            if (token != kGap) {
                new_row.push_back(old_row[old_index]);
                ++old_index;
            } else {
                new_row.push_back(kGap);
            }
        }
        merged.push_back(std::move(new_row));
    }
    merged.push_back(aligned.second);
    return merged;
}

// One word's encodable pronunciation candidates and their metadata.
struct BuiltCandidate {
    int                      reading = 0;
    std::vector<std::string> phonemes;
    std::vector<std::string> scripts;
    std::vector<int>         tokens;
    std::vector<int>         groups;      // 1-based group id per token
};

}  // namespace

// ---------------------------------------------------------------------------
// phoneme resolution
// ---------------------------------------------------------------------------

ResolvedSymbol resolve_phoneme(const std::string & phoneme,
                               const std::vector<std::string> & languages,
                               const std::function<int(const std::string &)> & lookup,
                               const std::vector<std::string> & global_symbols,
                               const std::vector<std::string> & stop_symbols) {
    // `languages` are the prefixes to try, most specific first; a global
    // symbol is never qualified.
    std::vector<std::string> candidates = languages;
    if (contains(global_symbols, phoneme)) candidates.clear();
    const std::string fallback = qualify_symbol(
        phoneme, candidates.empty() ? std::string() : candidates[0], global_symbols);

    if (contains(stop_symbols, phoneme)) return ResolvedSymbol{ -1, phoneme };
    if (contains(stop_symbols, fallback)) return ResolvedSymbol{ -1, fallback };

    const int literal = lookup(phoneme);
    if (literal >= 0) return ResolvedSymbol{ literal, phoneme };
    if (phoneme.find('/') == std::string::npos) {
        for (const std::string & language : candidates) {
            const std::string symbol = language + "/" + phoneme;
            const int id = lookup(symbol);
            if (id >= 0) return ResolvedSymbol{ id, symbol };
        }
    }
    return ResolvedSymbol{ -1, fallback };
}

// ---------------------------------------------------------------------------
// multi-sequence alignment
// ---------------------------------------------------------------------------

std::vector<std::vector<int>> align_multiple_sequences(
    const std::vector<std::vector<int>> & paths) {
    if (paths.empty()) return {};
    std::vector<std::vector<int>> rows{ paths[0] };
    for (std::size_t i = 1; i < paths.size(); ++i) {
        rows = merge_profile(rows, paths[i]);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// encode_paths
// ---------------------------------------------------------------------------

CandidateGrid encode_paths(const std::vector<Word> & words,
                           const std::function<int(const std::string &)> & lookup,
                           const std::vector<std::string> & global_symbols,
                           const std::vector<std::string> & stop_symbols,
                           const std::vector<std::string> & languages,
                           const std::string & oov_handling,
                           int num_reserved_tokens) {
    if (oov_handling != "raise" && oov_handling != "discard" && oov_handling != "force") {
        throw InvalidArgument("Unknown OOV handling: " + oov_handling);
    }

    std::vector<std::vector<BuiltCandidate>> per_word;
    per_word.reserve(words.size());
    for (const Word & word : words) {
        // A word with a language resolves within it; an untagged word ("any")
        // resolves through the requested languages.  The value may list several
        // languages (`language="zh,yue"`), so split it rather than use it whole.
        const std::vector<std::string> word_languages =
            word.language.empty() ? languages : split_language_tags(word.language);

        std::vector<BuiltCandidate> candidates;
        for (std::size_t reading_index = 0; reading_index < word.readings.size(); ++reading_index) {
            for (const Path & path : word.readings[reading_index].paths) {
                BuiltCandidate candidate;
                candidate.reading = static_cast<int>(reading_index);
                std::vector<std::string> unknown;
                for (const Group & group : path) {
                    if (group.phonemes.empty()) {
                        throw InvalidArgument(
                            "An empty pronunciation must be an empty path, not an empty group.");
                    }
                    std::vector<int>         group_tokens;
                    std::vector<std::string> group_phonemes;
                    for (const std::string & phoneme : group.phonemes) {
                        const ResolvedSymbol resolved = resolve_phoneme(
                            phoneme, word_languages, lookup, global_symbols, stop_symbols);
                        if (contains(stop_symbols, resolved.symbol)) continue;
                        if (resolved.id < 0) {
                            unknown.push_back(phoneme);
                            continue;
                        }
                        if (resolved.id < num_reserved_tokens) {
                            throw InvalidArgument("Reserved token in G2P output: '" + phoneme + "'");
                        }
                        group_tokens.push_back(resolved.id);
                        group_phonemes.push_back(resolved.symbol);
                    }
                    if (!group_tokens.empty()) {
                        candidate.scripts.push_back(group.script);
                        candidate.tokens.insert(candidate.tokens.end(),
                                                group_tokens.begin(), group_tokens.end());
                        candidate.phonemes.insert(candidate.phonemes.end(),
                                                  group_phonemes.begin(), group_phonemes.end());
                        candidate.groups.insert(candidate.groups.end(), group_tokens.size(),
                                                static_cast<int>(candidate.scripts.size()));
                    }
                }
                if (!unknown.empty()) {
                    // In force mode the whole candidate is dropped; otherwise
                    // the word (Python: the sample) cannot be encoded.  The
                    // reference raises for both "raise" and "discard" -- the
                    // two modes only differ at the call site, which reports
                    // the error or drops the sample.
                    if (oov_handling == "force") continue;
                    throw G2PEncodingError("Unknown phonemes " + format_phonemes(unknown) +
                                           " in word '" + word.text + "'");
                }
                candidates.push_back(std::move(candidate));
            }
        }
        if (candidates.empty() && oov_handling != "force") {
            throw G2PEncodingError("No pronunciation paths for word '" + word.text + "'");
        }
        per_word.push_back(std::move(candidates));
    }

    // ---- align each word's candidates into a shared profile ----
    std::vector<std::vector<std::vector<int>>> profiles;
    profiles.reserve(per_word.size());
    for (const std::vector<BuiltCandidate> & candidates : per_word) {
        std::vector<std::vector<int>> tokens;
        tokens.reserve(candidates.size());
        for (const BuiltCandidate & candidate : candidates) tokens.push_back(candidate.tokens);
        profiles.push_back(align_multiple_sequences(tokens));
    }

    // ---- lay the profiles out as grids (g2p/encoding.py:_pack_paths) ----
    std::size_t capacity = 0;
    for (const std::vector<std::vector<int>> & profile : profiles) {
        capacity += profile.empty() ? 0 : profile[0].size();
    }
    std::size_t width = 0;
    for (const std::vector<BuiltCandidate> & candidates : per_word) {
        width = std::max(width, candidates.size());
    }

    CandidateGrid grid;
    grid.W = static_cast<int>(per_word.size());
    grid.P = static_cast<int>(std::max<std::size_t>(1, capacity));
    grid.C = static_cast<int>(std::max<std::size_t>(1, width));
    grid.paths.assign(static_cast<std::size_t>(grid.P) * grid.C, 0);
    grid.groups.assign(static_cast<std::size_t>(grid.P) * grid.C, 0);
    grid.words.assign(static_cast<std::size_t>(grid.P), 0);
    grid.candidates.assign(static_cast<std::size_t>(grid.W) * grid.C, 0);
    grid.lexicon.resize(per_word.size());
    grid.texts.reserve(words.size());

    std::size_t offset = 0;
    for (std::size_t w = 0; w < per_word.size(); ++w) {
        const std::vector<BuiltCandidate> & candidates = per_word[w];
        const std::vector<std::vector<int>> & profile  = profiles[w];
        const std::size_t size  = profile.empty() ? 0 : profile[0].size();
        const std::size_t count = candidates.size();

        for (std::size_t c = 0; c < count; ++c) {
            grid.candidates[w * width + c] = 1;              // this candidate exists
        }
        for (std::size_t row = 0; row < size; ++row) {
            grid.words[offset + row] = static_cast<int>(w) + 1;
        }
        for (std::size_t c = 0; c < count; ++c) {
            for (std::size_t row = 0; row < size; ++row) {
                const int token = profile[c][row];
                grid.paths[(offset + row) * width + c] = token == kGap ? 0 : token;
            }
            std::size_t token_index = 0;
            for (std::size_t row = 0; row < size; ++row) {
                if (profile[c][row] == kGap) continue;
                grid.groups[(offset + row) * width + c] = candidates[c].groups[token_index++];
            }

            CandidateGrid::Candidate entry;
            entry.reading  = candidates[c].reading;
            entry.phonemes = candidates[c].phonemes;
            entry.scripts  = candidates[c].scripts;
            grid.lexicon[w].push_back(std::move(entry));
        }
        offset += size;
    }
    for (const Word & word : words) grid.texts.push_back(word.text);

    return grid;
}

}  // namespace tifa_ggml::internal::g2p
