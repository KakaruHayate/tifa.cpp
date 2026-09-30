// TextGrid writer + skip-handling behaviour.

#include "tifa_ggml/textgrid.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace tifa_ggml;

namespace {

AlignResult make_result(const std::vector<std::pair<float, float>> & spans) {
    AlignResult r;
    r.num_frames = 100;
    r.labels = { "AP", "zh", "e" };
    r.labels.resize(spans.size());
    r.words.assign(spans.size(), 1);
    r.groups.assign(spans.size(), 0);
    for (std::size_t i = 0; i < spans.size(); ++i) {
        r.spans.push_back({spans[i].first, spans[i].second});
        r.groups[i] = static_cast<std::int32_t>(i + 1);
    }
    r.word_texts = { "ni hao" };
    r.group_scripts = r.labels;
    return r;
}

}  // namespace

TEST(TextGrid, FormatsThreeTiersInOrder) {
    std::vector<TextGridTier> tiers = {
        { "texts",  {{0.0, 1.0, "hello"}} },
        { "words",  {{0.0, 1.0, "he"}} },
        { "phones", {{0.0, 0.5, "h"}, {0.5, 1.0, "e"}} },
    };
    const std::string text = format_textgrid(tiers, 1.0);
    EXPECT_NE(text.find("Object class = \"TextGrid\""), std::string::npos);
    const auto pos_texts  = text.find("name = \"texts\"");
    const auto pos_words  = text.find("name = \"words\"");
    const auto pos_phones = text.find("name = \"phones\"");
    EXPECT_LT(pos_texts, pos_words);
    EXPECT_LT(pos_words, pos_phones);
    EXPECT_NE(text.find("intervals: size = 2"), std::string::npos);
    EXPECT_NE(text.find("text = \"h\""), std::string::npos);
}

TEST(TextGrid, EscapesEmbeddedQuotes) {
    std::vector<TextGridTier> tiers = { { "phones", {{0.0, 1.0, "a\"b"}} } };
    const std::string text = format_textgrid(tiers, 1.0);
    EXPECT_NE(text.find("text = \"a\"\"b\""), std::string::npos);
}

TEST(TextGrid, OmitsZeroWidthSpans) {
    auto r = make_result({{0.0f, 0.5f}, {0.5f, 0.5f}, {0.5f, 1.0f}});
    std::vector<TextGridTier> tiers;
    double xmax = 0.0;
    ASSERT_TRUE(build_alignment_tiers(r, SkipHandling::Omit, 1.0, tiers, xmax));
    ASSERT_EQ(tiers.size(), 3u);
    EXPECT_EQ(tiers[2].name, "phones");
    EXPECT_EQ(tiers[2].intervals.size(), 2u);        // the zero-width one is gone
    EXPECT_EQ(tiers[2].intervals[0].text, "AP");
    EXPECT_EQ(tiers[2].intervals[1].text, "e");
}

TEST(TextGrid, DiscardsTheWholeSample) {
    auto r = make_result({{0.0f, 0.5f}, {0.5f, 0.5f}});
    std::vector<TextGridTier> tiers;
    double xmax = 0.0;
    EXPECT_FALSE(build_alignment_tiers(r, SkipHandling::Discard, 1.0, tiers, xmax));
}

TEST(TextGrid, PreservesZeroWidthSpansAtOneMillisecond) {
    auto r = make_result({{0.0f, 0.5f}, {0.5f, 0.5f}, {0.5f, 1.0f}});
    std::vector<TextGridTier> tiers;
    double xmax = 0.0;
    ASSERT_TRUE(build_alignment_tiers(r, SkipHandling::Preserve, 1.0, tiers, xmax));
    ASSERT_EQ(tiers.size(), 3u);
    EXPECT_EQ(tiers[2].intervals.size(), 3u);
    for (const auto & iv : tiers[2].intervals) {
        EXPECT_GE(iv.xmax, iv.xmin + 0.0009);
    }
}

TEST(TextGrid, ClampsMonotoneAndMatchesWordLabel) {
    auto r = make_result({{0.0f, 0.4f}, {0.3f, 0.8f}});   // overlapping spans
    std::vector<TextGridTier> tiers;
    double xmax = 0.0;
    ASSERT_TRUE(build_alignment_tiers(r, SkipHandling::Omit, 1.0, tiers, xmax));
    const auto & phones = tiers[2].intervals;
    ASSERT_EQ(phones.size(), 2u);
    EXPECT_LE(phones[0].xmax, phones[1].xmin + 1e-9);
    EXPECT_EQ(tiers[0].intervals[0].text, "ni hao");
}
