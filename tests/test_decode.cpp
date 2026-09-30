// Viterbi decoder parity with modules/decoding.py:decode_alignment_flat.

#include "align_decode.h"

#include "tifa_ggml/errors.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using tifa_ggml::internal::decode_alignment_flat;

namespace {

// Build a similarity matrix from a per-token "true" frame range: high inside,
// low outside.  Diagonals make the intended alignment unambiguous.
std::vector<float> diag_sim(int T, int N, float high = 1.0f, float low = -1.0f) {
    std::vector<float> sim(static_cast<std::size_t>(T) * N, low);
    for (int n = 0; n < N; ++n) {
        const int on  = (T * n) / N;
        const int off = (T * (n + 1)) / N;
        for (int t = on; t < off; ++t) {
            sim[static_cast<std::size_t>(t) * N + n] = high;
        }
    }
    return sim;
}

}  // namespace

TEST(Decode, RecoversDiagonalSpans) {
    const int T = 20, N = 4;
    const auto sim = diag_sim(T, N);
    const auto spans = decode_alignment_flat(sim.data(), T, N, nullptr, 0.5f);
    ASSERT_EQ(spans.size(), 4u);
    EXPECT_EQ(spans[0], std::make_pair(0, 5));
    EXPECT_EQ(spans[1], std::make_pair(5, 10));
    EXPECT_EQ(spans[2], std::make_pair(10, 15));
    EXPECT_EQ(spans[3], std::make_pair(15, 20));
}

TEST(Decode, SpansCoverEveryFrameForStrictlyPreferredDiagonal) {
    const int T = 30, N = 5;
    const auto sim = diag_sim(T, N);
    const auto spans = decode_alignment_flat(sim.data(), T, N, nullptr, 0.5f);
    // Spans are monotone and contiguous when every token is worth aligning.
    int cursor = 0;
    for (const auto & s : spans) {
        EXPECT_EQ(s.first, cursor);
        EXPECT_GT(s.second, s.first);
        cursor = s.second;
    }
    EXPECT_EQ(cursor, T);
}

TEST(Decode, SkipsATokenWhenThePenaltyBeatsTheEvidence) {
    // Token 1 has no positive evidence anywhere: rather than eating frames it
    // should be skipped (zero width) and the span canonicalised into the gap.
    const int T = 12, N = 3;
    std::vector<float> sim(static_cast<std::size_t>(T) * N, -1.0f);
    for (int t = 0; t < 5; ++t)  sim[static_cast<std::size_t>(t) * N + 0] = 1.0f;
    for (int t = 7; t < 12; ++t) sim[static_cast<std::size_t>(t) * N + 2] = 1.0f;
    const auto spans = decode_alignment_flat(sim.data(), T, N, nullptr, 0.5f);
    ASSERT_EQ(spans.size(), 3u);
    EXPECT_EQ(spans[1].first, spans[1].second);          // skipped => zero width
    EXPECT_GT(spans[0].second, spans[0].first);
    EXPECT_GT(spans[2].second, spans[2].first);
    EXPECT_GE(spans[1].first, spans[0].second);
    EXPECT_LE(spans[1].first, spans[2].first);
}

TEST(Decode, GroupConstraintForbidsGapsInsideAGroup) {
    // Two phones of the same pronunciation group must be adjacent: the gap
    // between them is disallowed, so no zero-width span may sit in the middle.
    const int T = 10, N = 3;
    const auto sim = diag_sim(T, N);
    const std::vector<std::int32_t> groups = {1, 1, 2};
    const auto spans = decode_alignment_flat(sim.data(), T, N, groups.data(), 0.5f);
    ASSERT_EQ(spans.size(), 3u);
    EXPECT_EQ(spans[0].second, spans[1].first);   // no gap allowed between them
}

TEST(Decode, RejectsDegenerateInput) {
    std::vector<float> sim(4, 0.0f);
    EXPECT_THROW(decode_alignment_flat(sim.data(), 0, 2, nullptr, 0.5f),
                 tifa_ggml::InvalidArgument);
    EXPECT_THROW(decode_alignment_flat(sim.data(), 2, 2, nullptr, -1.0f),
                 tifa_ggml::InvalidArgument);
}
