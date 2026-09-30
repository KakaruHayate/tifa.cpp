#pragma once

// Candidate selection + compaction (port of lib/path_traversal.py's
// `extract_tokens` / `compact_sequences` / `materialize_paths`).
//
// Used after pronunciation scoring to turn the candidate grid plus 1-based
// per-word choices into the flat token sequence the aligner consumes.

#include "g2p/g2p.h"

#include <vector>

namespace tifa_ggml::internal::g2p {

struct MaterializedPaths {
    std::vector<int> tokens;   // compacted token ids (0 = padding tail)
    std::vector<int> words;    // 1-based semantic word id per token
    std::vector<int> groups;   // 1-based global group id per token
    int              count = 0;  // number of non-zero tokens
};

// `choices` has one 1-based candidate id per word (0 = absent word).
MaterializedPaths materialize_paths(const CandidateGrid & grid,
                                    const std::vector<int> & choices);

// First valid candidate per word (lib/path_traversal.py:first_choices).
std::vector<int> first_choices(const CandidateGrid & grid);

}  // namespace tifa_ggml::internal::g2p
