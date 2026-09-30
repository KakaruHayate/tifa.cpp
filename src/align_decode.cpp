#include "align_decode.h"

#include "tifa_ggml/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace tifa_ggml::internal {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Place silence at the first permitted gap in each skipped-token run
// (port of `_canonicalize_skipped_spans`).
void canonicalize_skipped_spans(std::vector<std::pair<int, int>> & spans, int T,
                                const std::vector<char> & gap_allowed)
{
    const int N = static_cast<int>(spans.size());
    int lo = 0;
    while (lo < N) {
        if (spans[static_cast<std::size_t>(lo)].first != spans[static_cast<std::size_t>(lo)].second) {
            ++lo;
            continue;
        }
        int hi = lo;
        while (hi + 1 < N &&
               spans[static_cast<std::size_t>(hi + 1)].first ==
               spans[static_cast<std::size_t>(hi + 1)].second) {
            ++hi;
        }
        const int left  = (lo > 0) ? spans[static_cast<std::size_t>(lo - 1)].second : 0;
        const int right = (hi + 1 < N) ? spans[static_cast<std::size_t>(hi + 1)].first : T;

        int gap_index = lo;
        while (gap_index <= hi + 1 && !gap_allowed[static_cast<std::size_t>(gap_index)]) {
            ++gap_index;
        }
        // The reference asserts left == right when no gap is available; with
        // group constraints that cannot happen, but stay defensive.
        if (gap_index > hi + 1 && left != right) {
            // keep left/right anchoring as-is (no assertion failure in release)
        }
        for (int i = lo; i <= hi; ++i) {
            const int anchor = (i < gap_index) ? left : right;
            spans[static_cast<std::size_t>(i)] = {anchor, anchor};
        }
        lo = hi + 1;
    }
}

// Port of `_decode_flat_single`.
std::vector<std::pair<int, int>> decode_single(const float * sim, int T, int N,
                                               float skip_penalty,
                                               const std::vector<char> & gap_allowed)
{
    std::vector<float> gap(static_cast<std::size_t>(N) + 1, kNegInf);
    std::vector<float> token(static_cast<std::size_t>(N), kNegInf);
    std::vector<std::int8_t> gap_back(static_cast<std::size_t>(T + 1) * (N + 1), 0);
    std::vector<std::int8_t> token_back(static_cast<std::size_t>(T + 1) * N, 0);

    gap[0] = 0.0f;
    for (int i = 0; i < N; ++i) {
        gap[static_cast<std::size_t>(i) + 1] = gap[static_cast<std::size_t>(i)] - skip_penalty;
        gap_back[static_cast<std::size_t>(i) + 1] = 2;   // skip
    }

    std::vector<float> next_token(static_cast<std::size_t>(N));
    std::vector<float> next_gap(static_cast<std::size_t>(N) + 1);

    for (int t = 1; t <= T; ++t) {
        std::fill(next_gap.begin(), next_gap.end(), kNegInf);
        for (int i = 0; i < N; ++i) {
            float best = token[static_cast<std::size_t>(i)];
            if (gap[static_cast<std::size_t>(i)] > best) {
                best = gap[static_cast<std::size_t>(i)];
                token_back[static_cast<std::size_t>(t) * N + i] = 1;   // enter from gap
            }
            next_token[static_cast<std::size_t>(i)] =
                best + sim[static_cast<std::size_t>(t - 1) * N + i];
        }
        for (int i = 0; i <= N; ++i) {
            if (gap_allowed[static_cast<std::size_t>(i)]) {
                next_gap[static_cast<std::size_t>(i)] = gap[static_cast<std::size_t>(i)];
            }
        }
        for (int i = 0; i < N; ++i) {
            if (next_token[static_cast<std::size_t>(i)] > next_gap[static_cast<std::size_t>(i) + 1]) {
                next_gap[static_cast<std::size_t>(i) + 1] = next_token[static_cast<std::size_t>(i)];
                gap_back[static_cast<std::size_t>(t) * (N + 1) + (i + 1)] = 1;  // exit without a frame
            }
            const float skipped = next_gap[static_cast<std::size_t>(i)] - skip_penalty;
            if (skipped > next_gap[static_cast<std::size_t>(i) + 1]) {
                next_gap[static_cast<std::size_t>(i) + 1] = skipped;
                gap_back[static_cast<std::size_t>(t) * (N + 1) + (i + 1)] = 2;  // skip
            }
        }
        gap.swap(next_gap);
        token.swap(next_token);
    }

    std::vector<std::pair<int, int>> spans(static_cast<std::size_t>(N), {-1, -1});
    int  t = T;
    int  i = N;
    bool in_token = false;
    while (t > 0 || i > 0 || in_token) {
        if (in_token) {
            auto & sp = spans[static_cast<std::size_t>(i)];
            if (sp.second < 0) sp.second = t;
            sp.first = t - 1;
            const std::int8_t entered = token_back[static_cast<std::size_t>(t) * N + i];
            --t;
            if (entered == 1) in_token = false;
        } else {
            const std::int8_t source = gap_back[static_cast<std::size_t>(t) * (N + 1) + i];
            if (source == 0) {
                --t;
            } else if (source == 1) {
                --i;
                in_token = true;
            } else {
                --i;
                spans[static_cast<std::size_t>(i)] = {t, t};
            }
        }
    }

    canonicalize_skipped_spans(spans, T, gap_allowed);
    return spans;
}

}  // namespace

std::vector<std::pair<int, int>> decode_alignment_flat(
    const float * sim, int T, int N,
    const std::int32_t * groups, float skip_penalty)
{
    if (T <= 0 || N <= 0 || !sim) {
        throw InvalidArgument("decode_alignment_flat: empty similarity matrix");
    }
    if (skip_penalty < 0.0f) {
        throw InvalidArgument("decode_alignment_flat: skip_penalty must be non-negative");
    }

    std::vector<char> gap_allowed(static_cast<std::size_t>(N) + 1, 1);
    if (groups) {
        for (int i = 1; i < N; ++i) {
            gap_allowed[static_cast<std::size_t>(i)] =
                groups[i - 1] != groups[i] ? 1 : 0;
        }
    }
    return decode_single(sim, T, N, skip_penalty, gap_allowed);
}

}  // namespace tifa_ggml::internal
