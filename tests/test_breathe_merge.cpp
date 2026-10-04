// 2PASS merge: folding the breath AP/SP segments into a phones tier.
//
// The dataset workflow is align -> breathe --merge -> align --textgrid; the
// merge decides where the detected breaths land in the phone timeline, so its
// rules (skip already-labelled stretches, split straddling phones, insert
// gap-living segments, drop unlabelled stretches, collapse adjacent
// same-label intervals) are pinned here.

#include "cli/phone_input.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using tifa_cli::merge_breath_into_phones;
using tifa_cli::TimedInterval;

namespace {

TimedInterval iv(double a, double b, const char * text) {
    return TimedInterval{a, b, text};
}

tifa_ggml::BreathSegment seg(const char * label, double a, double b) {
    tifa_ggml::BreathSegment s;
    s.label = label;
    s.start = a;
    s.end = b;
    return s;
}

std::string render(const std::vector<TimedInterval> & xs) {
    std::string out;
    for (const TimedInterval & x : xs) {
        if (!out.empty()) out += ' ';
        out += x.text + "@" + std::to_string(x.xmin) + "-" + std::to_string(x.xmax);
    }
    return out;
}

}  // namespace

TEST(BreatheMerge, SplitsStraddlingPhoneAndKeepsOrder) {
    const std::vector<TimedInterval> phones{
        iv(0.0, 1.0, "a"), iv(1.0, 2.0, "b")};
    // breath inside the second phone
    std::size_t inserted = 0;
    const auto out = merge_breath_into_phones(
        phones, {seg("AP", 1.2, 1.5)}, 0.05, &inserted);
    ASSERT_EQ(render(out), render(std::vector<TimedInterval>{
        iv(0.0, 1.0, "a"), iv(1.0, 1.2, "b"), iv(1.2, 1.5, "AP"),
        iv(1.5, 2.0, "b")}));
    EXPECT_EQ(inserted, 1u);
}

TEST(BreatheMerge, SkipsAlreadyLabelledStretch) {
    // the annotation already placed an AP where the detector found one
    const std::vector<TimedInterval> phones{
        iv(0.0, 0.4, "AP"), iv(0.4, 1.4, "a")};
    std::size_t inserted = 99;
    const auto out = merge_breath_into_phones(
        phones, {seg("AP", 0.0, 0.4)}, 0.05, &inserted);
    ASSERT_EQ(render(out), render(std::vector<TimedInterval>{
        iv(0.0, 0.4, "AP"), iv(0.4, 1.4, "a")}));
    EXPECT_EQ(inserted, 0u);
}

TEST(BreatheMerge, InsertsIntoGapsAndDropsUnlabelled) {
    // leading silence 0.0-0.5, a phone, trailing silence with an SP
    const std::vector<TimedInterval> phones{
        iv(0.0, 0.5, ""), iv(0.5, 2.0, "a"), iv(2.0, 3.0, "")};
    std::size_t inserted = 0;
    const auto out = merge_breath_into_phones(
        phones, {seg("SP", 0.0, 0.5), seg("AP", 2.2, 2.6)}, 0.05, &inserted);
    // the leading SP is inside an *unlabelled* stretch: it is inserted
    ASSERT_EQ(render(out), render(std::vector<TimedInterval>{
        iv(0.0, 0.5, "SP"), iv(0.5, 2.0, "a"), iv(2.2, 2.6, "AP")}));
    EXPECT_EQ(inserted, 2u);
}

TEST(BreatheMerge, CollapsesAdjacentSameLabels) {
    // detected AP abutting an annotation AP -> one interval
    const std::vector<TimedInterval> phones{
        iv(0.0, 0.3, "AP"), iv(0.3, 1.0, "a")};
    std::size_t inserted = 0;
    const auto out = merge_breath_into_phones(
        phones, {seg("AP", 0.3, 0.55)}, 0.05, &inserted);
    ASSERT_EQ(render(out), render(std::vector<TimedInterval>{
        iv(0.0, 0.55, "AP"), iv(0.55, 1.0, "a")}));
}

TEST(BreatheMerge, RespectsMinimumDuration) {
    const std::vector<TimedInterval> phones{iv(0.0, 1.0, "a")};
    std::size_t inserted = 0;
    const auto out = merge_breath_into_phones(
        phones, {seg("AP", 0.4, 0.44)}, 0.05, &inserted);   // 40 ms < 50 ms
    ASSERT_EQ(render(out), render(std::vector<TimedInterval>{iv(0.0, 1.0, "a")}));
    EXPECT_EQ(inserted, 0u);
}

TEST(BreatheMerge, VoiceSegmentsAreNeverInserted) {
    const std::vector<TimedInterval> phones{iv(0.0, 1.0, "a")};
    std::size_t inserted = 0;
    const auto out = merge_breath_into_phones(
        phones, {seg("V", 0.2, 0.8)}, 0.05, &inserted);
    ASSERT_EQ(render(out), render(std::vector<TimedInterval>{iv(0.0, 1.0, "a")}));
    EXPECT_EQ(inserted, 0u);
}

TEST(BreatheMerge, InsertsExhaleSegmentsWhenPresent) {
    // phone 'a' followed by silence, with an EP detected between them
    const std::vector<TimedInterval> phones{
        iv(0.0, 1.0, "a"), iv(1.0, 2.0, "SP")};
    std::size_t inserted = 0;
    const auto out = merge_breath_into_phones(
        phones, {seg("EP", 0.9, 1.2)}, 0.05, &inserted);
    // straddling phone boundary [0.9, 1.2] vs [0.0, 1.0]: not inside phone, but if inside SP or phone:
    // let's test EP inside the trailing SP [1.0, 1.3]
    const auto out2 = merge_breath_into_phones(
        phones, {seg("EP", 1.0, 1.3)}, 0.05, &inserted);
    ASSERT_EQ(render(out2), render(std::vector<TimedInterval>{
        iv(0.0, 1.0, "a"), iv(1.0, 1.3, "EP"), iv(1.3, 2.0, "SP")}));
    EXPECT_EQ(inserted, 1u);
}

