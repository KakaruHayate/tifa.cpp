#pragma once

// Viterbi decoding of the frame/token similarity matrix (port of
// openvpi/TIFA modules/decoding.py `decode_alignment_flat`).
//
// Maximises the summed raw cosine similarity minus `skip_penalty` per skipped
// token.  Every token emits at least one frame or pays the penalty, including
// prefixes and suffixes.  Exits and skips consume no frames.  `groups`
// restricts gap waiting (phones inside one pronunciation group may not be
// separated by a gap).

#include <cstdint>
#include <utility>
#include <vector>

namespace tifa_ggml::internal {

// sim: row-major [T, N].  groups: N entries (1-based, non-decreasing) or null.
// Returns N (onset, offset) frame spans; skipped tokens get zero-width spans.
std::vector<std::pair<int, int>> decode_alignment_flat(
    const float * sim, int T, int N,
    const std::int32_t * groups, float skip_penalty);

}  // namespace tifa_ggml::internal
