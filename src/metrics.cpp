#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tifa_ggml::internal {

namespace {

// Per-frame token assignment shared by determinacy and monotonicity:
//   pos      = last token whose onset <= t (searchsorted side="right" - 1)
//   in_token = pos >= 0 && t < offsets[pos]
// `token_idx` is the token index or -1 (gap); `gap_pos` is the clamped
// neighbouring token index used for the gap columns.
void assign_frames(const std::vector<std::pair<int, int>> & spans, int T,
                   const std::vector<char> & n_mask,
                   std::vector<int> & token_idx, std::vector<int> & gap_pos)
{
    const int N = static_cast<int>(spans.size());
    token_idx.assign(static_cast<std::size_t>(T), -1);
    gap_pos.assign(static_cast<std::size_t>(T), 0);

    // onsets are non-decreasing, so a linear sweep is equivalent to
    // searchsorted(side="right") - 1.
    int valid_last = -1;
    for (int n = 0; n < N; ++n) {
        if (n_mask[static_cast<std::size_t>(n)]) valid_last = n;
    }
    const int n_per_item = std::max(0, valid_last);   // n_mask.sum() - 1

    for (int t = 0; t < T; ++t) {
        int pos = -1;
        for (int n = 0; n < N; ++n) {
            if (!n_mask[static_cast<std::size_t>(n)]) continue;
            if (spans[static_cast<std::size_t>(n)].first <= t) pos = n;
            else break;
        }
        const bool in_token =
            pos >= 0 && t < spans[static_cast<std::size_t>(pos)].second;
        token_idx[static_cast<std::size_t>(t)] = in_token ? pos : -1;
        const int gp = std::min(std::max(pos + 1, 0), n_per_item);
        gap_pos[static_cast<std::size_t>(t)] = in_token ? pos : gp;
    }
}

}  // namespace

DiagnosisMetrics compute_diagnosis(const float * sim, int T, int N,
                                   const std::vector<std::pair<int, int>> & spans,
                                   const std::vector<char> & t_mask,
                                   const std::vector<char> & n_mask)
{
    DiagnosisMetrics out;
    if (T <= 0 || N <= 0) return out;

    constexpr float kPower = 2.0f;
    constexpr int   kWidth = 5;      // determinacy neighbourhood half-width

    auto at = [&](int t, int n) { return sim[static_cast<std::size_t>(t) * N + n]; };
    auto t_ok = [&](int t) { return t_mask[static_cast<std::size_t>(t)] != 0; };
    auto n_ok = [&](int n) { return n_mask[static_cast<std::size_t>(n)] != 0; };

    // ---- confidence: mean similarity inside each predicted span ----------
    {
        double acc = 0.0;
        int    count = 0;
        for (int n = 0; n < N; ++n) {
            if (!n_ok(n)) continue;
            const int on  = spans[static_cast<std::size_t>(n)].first;
            const int off = spans[static_cast<std::size_t>(n)].second;
            double sum = 0.0;
            int    cnt = 0;
            for (int t = std::max(on, 0); t < std::min(off, T); ++t) {
                if (!t_ok(t)) continue;
                sum += at(t, n);
                ++cnt;
            }
            if (cnt > 0) acc += sum / cnt;   // zero-width spans contribute 0
            ++count;
        }
        out.confidence = count > 0 ? static_cast<float>(acc / count) : 0.0f;
    }

    // ---- a[t][n] = relu(sim)^power, masked --------------------------------
    std::vector<float> a(static_cast<std::size_t>(T) * N, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            if (!t_ok(t) || !n_ok(n)) continue;
            const float v = at(t, n);
            a[static_cast<std::size_t>(t) * N + n] =
                v > 0.0f ? std::pow(v, kPower) : 0.0f;
        }
    }

    std::vector<int> token_idx, gap_pos;
    assign_frames(spans, T, n_mask, token_idx, gap_pos);

    // ---- determinacy ------------------------------------------------------
    {
        // local_sum[t][n] = sum of a[t][n-width .. n+width]
        std::vector<float> local_sum(static_cast<std::size_t>(T) * N, 0.0f);
        for (int t = 0; t < T; ++t) {
            for (int n = 0; n < N; ++n) {
                double s = 0.0;
                for (int k = std::max(0, n - kWidth); k <= std::min(N - 1, n + kWidth); ++k) {
                    s += a[static_cast<std::size_t>(t) * N + k];
                }
                local_sum[static_cast<std::size_t>(t) * N + n] = static_cast<float>(s);
            }
        }
        double num = 0.0, den = 0.0;
        for (int t = 0; t < T; ++t) {
            if (!t_ok(t)) continue;
            const int ti = token_idx[static_cast<std::size_t>(t)];
            if (ti >= 0) {
                num += a[static_cast<std::size_t>(t) * N + ti];
            }   // gap frames contribute 0 (the reference's zero-padded column)
            den += local_sum[static_cast<std::size_t>(t) * N + gap_pos[static_cast<std::size_t>(t)]];
        }
        out.determinacy = static_cast<float>(num / (den + 1e-8));
    }

    // ---- monotonicity (width = unlimited) ---------------------------------
    {
        std::vector<float> suffix(static_cast<std::size_t>(T) * N, 0.0f);
        for (int t = 0; t < T; ++t) {
            double run = 0.0;
            for (int n = N - 1; n >= 0; --n) {
                run += a[static_cast<std::size_t>(t) * N + n];
                suffix[static_cast<std::size_t>(t) * N + n] = static_cast<float>(run);
            }
        }
        double acc = 0.0;
        int    count = 0;
        for (int t = 0; t < T; ++t) {
            if (!t_ok(t)) continue;
            const int ti = token_idx[static_cast<std::size_t>(t)];
            if (ti < 0) continue;
            double total = 0.0;
            for (int n = 0; n < N; ++n) total += a[static_cast<std::size_t>(t) * N + n];
            const double fwd = suffix[static_cast<std::size_t>(t) * N + ti];
            const double bwd = total - fwd;
            const double denom = fwd + bwd;
            if (denom <= 0.0) continue;      // valid requires mass > 0
            acc += fwd / std::max(denom, 1e-8);
            ++count;
        }
        out.monotonicity = count > 0 ? static_cast<float>(acc / count) : 0.0f;
    }

    return out;
}

}  // namespace tifa_ggml::internal
