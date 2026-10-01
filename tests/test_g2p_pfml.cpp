// PFML (openvpi/g2pflow's Pronunciation Flow Markup Language): parsing and
// conversion into G2P words.
//
// Upstream requires PFML as the G2P input format, so the semantics pinned
// here mirror g2pflow's docs: language scopes route converter selection,
// <word phonemes="..."> bypasses G2P with final phonemes, bare <phoneme>
// runs group into one word, and plain text inside a fragment still goes
// through the configured converters.

#include "g2p/pfml.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace tifa_ggml::internal::g2p;

namespace {

// The g2pflow quick-start pipeline: a characters converter, no dictionaries.
Pipeline characters_pipeline() {
    return Pipeline::from_config(
        R"({"converters":[{"id":"characters","language":"en",
             "kwargs":{"mapping":{"h":["HH"],"i":["AY"]," ":["SP"]}}}]})",
        "");
}

std::string path_text(const Word & word) {
    std::string out;
    for (const Reading & reading : word.readings) {
        for (const Path & path : reading.paths) {
            if (!out.empty()) out += " | ";
            for (const Group & group : path) {
                if (!out.empty() && &group != &path.front()) out += " ";
                for (const std::string & phoneme : group.phonemes) {
                    if (!out.empty() && out.back() != ' ' && out.back() != '|') out += ' ';
                    out += phoneme;
                }
            }
        }
    }
    return out;
}

}  // namespace

TEST(Pfml, DetectsFragmentsOnly) {
    EXPECT_FALSE(looks_like_pfml("hello world"));
    EXPECT_FALSE(looks_like_pfml("3 < 5 and 5 > 3"));
    EXPECT_FALSE(looks_like_pfml("<notanelement>x</notanelement>"));
    EXPECT_TRUE(looks_like_pfml("<phoneme>ong</phoneme>"));
    EXPECT_TRUE(looks_like_pfml("hi <word phonemes=\"HH AY\">hi</word>"));
    EXPECT_TRUE(looks_like_pfml("<scope language=\"ja\">x</scope>"));
}

TEST(Pfml, DirectPhonemesBypassG2p) {
    const Pipeline pipeline = characters_pipeline();
    const std::vector<Word> words =
        convert_pfml(pipeline, "<word phonemes=\"zh ong\">重</word>", {});
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0].text, "重");
    EXPECT_EQ(path_text(words[0]), "zh ong");
    ASSERT_EQ(words[0].readings.size(), 1u);
    ASSERT_EQ(words[0].readings[0].paths.size(), 1u);
    EXPECT_EQ(words[0].readings[0].paths[0][0].script, "重");   // script defaults to text
}

TEST(Pfml, ScriptAttributeLabelsTheGroup) {
    const Pipeline pipeline = characters_pipeline();
    const std::vector<Word> words = convert_pfml(
        pipeline, "<word script=\"zhong\" phonemes=\"zh ong\">重</word>", {});
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0].readings[0].paths[0][0].script, "zhong");
}

TEST(Pfml, ScopeRoutesConversion) {
    const Pipeline pipeline = characters_pipeline();
    // outside the scope no converter claims the text; inside `en` the
    // characters converter maps h/i
    const std::vector<Word> words =
        convert_pfml(pipeline, "<scope language=\"en\">hi</scope>", {"en"});
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0].text, "hi");
    EXPECT_EQ(path_text(words[0]), "HH AY");
    EXPECT_EQ(words[0].language, "en");
}

TEST(Pfml, BarePhonemeRunsGroupIntoOneWord) {
    const Pipeline pipeline = characters_pipeline();
    const std::vector<Word> words = convert_pfml(
        pipeline, "<phoneme>AP</phoneme><phoneme>zh</phoneme><phoneme>ong</phoneme>", {});
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(path_text(words[0]), "AP zh ong");
}

TEST(Pfml, MixedTextAndDirectWords) {
    const Pipeline pipeline = characters_pipeline();
    const std::vector<Word> words = convert_pfml(
        pipeline, "hi<word phonemes=\"AP\">br</word>hi", {"en"});
    ASSERT_EQ(words.size(), 3u);
    EXPECT_EQ(words[0].text, "hi");
    EXPECT_EQ(words[1].text, "br");
    EXPECT_EQ(path_text(words[1]), "AP");
    EXPECT_EQ(words[2].text, "hi");
}

TEST(Pfml, EntitiesAndSelfClosing) {
    const Pipeline pipeline = characters_pipeline();
    const std::vector<Word> words =
        convert_pfml(pipeline, "<word phonemes=\"x\">&amp;lt;</word>", {});
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0].text, "&lt;");   // entities decode once
    const std::vector<Word> self_closed =
        convert_pfml(pipeline, "<word phonemes=\"zh ong\"/>", {});
    ASSERT_EQ(self_closed.size(), 1u);
    EXPECT_EQ(self_closed[0].text, "zh ong");   // no inner text: phonemes are the label
}

TEST(Pfml, MalformedMarkupRaises) {
    const Pipeline pipeline = characters_pipeline();
    EXPECT_THROW(convert_pfml(pipeline, "<word phonemes=\"x\">unclosed", {}),
                 tifa_ggml::InvalidArgument);
    EXPECT_THROW(convert_pfml(pipeline, "<scope language=\"en\"></phone>", {}),
                 tifa_ggml::InvalidArgument);
    EXPECT_THROW(convert_pfml(pipeline, "<bogus>x</bogus>", {}),
                 tifa_ggml::InvalidArgument);
}
