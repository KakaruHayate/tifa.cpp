#pragma once

// Reference-free alignment diagnostics (port of TIFA's
// modules/metrics/reference_free.py).  Single sample (B = 1) only.

#include <utility>
#include <vector>

namespace tifa_ggml::internal {

struct DiagnosisMetrics {
    float confidence    = 0.0f;
    float determinacy   = 0.0f;
    float monotonicity  = 0.0f;
};

// sim      : row-major [T, N] frame/token cosine similarity
// spans    : N (onset, offset) frame spans (offsets exclusive)
// t_mask   : T flags (all 1 for exact-length inference)
// n_mask   : N flags (all 1)
DiagnosisMetrics compute_diagnosis(const float * sim, int T, int N,
                                   const std::vector<std::pair<int, int>> & spans,
                                   const std::vector<char> & t_mask,
                                   const std::vector<char> & n_mask);

}  // namespace tifa_ggml::internal
