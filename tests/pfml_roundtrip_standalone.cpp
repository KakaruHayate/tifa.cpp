// Standalone round-trip check for the PFML parser/serializer.
// Builds without ggml: only json.cpp + src/g2p/*.cpp are needed.

#include "g2p/g2p.h"
#include "g2p/pfml.h"

#include <iostream>
#include <string>
#include <vector>

using namespace tifa_ggml::internal::g2p;

static int failures = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        std::cout << "FAIL: " << what << "\n";
        ++failures;
    } else {
        std::cout << "ok:   " << what << "\n";
    }
}

static bool same_words(const std::vector<Word> & a, const std::vector<Word> & b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].text != b[i].text || a[i].language != b[i].language) return false;
        if (a[i].readings.size() != b[i].readings.size()) return false;
        for (std::size_t r = 0; r < a[i].readings.size(); ++r) {
            const std::vector<Path> & pa = a[i].readings[r].paths;
            const std::vector<Path> & pb = b[i].readings[r].paths;
            if (pa.size() != pb.size()) return false;
            for (std::size_t p = 0; p < pa.size(); ++p) {
                if (pa[p].size() != pb[p].size()) return false;
                for (std::size_t g = 0; g < pa[p].size(); ++g) {
                    if (pa[p][g].script != pb[p][g].script) return false;
                    if (pa[p][g].phonemes != pb[p][g].phonemes) return false;
                }
            }
        }
    }
    return true;
}

static std::vector<std::string> v(std::initializer_list<const char *> items) {
    std::vector<std::string> out;
    for (const char * s : items) out.emplace_back(s);
    return out;
}

int main() {
    Pipeline pipeline = Pipeline::from_config("{}", "");

    // 1. compact direct word (the documented form) — must round-trip verbatim.
    {
        const std::string src =
            "<word text=\"重\" language=\"zh\" script=\"zhong\" phonemes=\"zh ong\"/>";
        const std::vector<Word> words = convert_pfml(pipeline, src, {});
        check(words.size() == 1, "compact: one word");
        check(words[0].text == "重", "compact: text");
        check(words[0].language == "zh", "compact: language");
        check(words[0].readings.size() == 1 && words[0].readings[0].paths.size() == 1 &&
                  words[0].readings[0].paths[0].size() == 1,
              "compact: one reading / path / group");
        check(words[0].readings[0].paths[0][0].phonemes == v({"zh", "ong"}),
              "compact: phonemes");
        const std::string back = to_pfml(words);
        check(back == src, "compact: round-trip text identical");
        if (back != src) std::cout << "      got: " << back << "\n";
        check(same_words(words, convert_pfml(pipeline, back, {})),
              "compact: round-trip model equal");
    }

    // 2. inner-text form
    {
        const std::vector<Word> words =
            convert_pfml(pipeline, "<word phonemes=\"zh ong\">重</word>", {});
        check(words.size() == 1 && words[0].text == "重", "inner-text: text");
        check(words[0].readings[0].paths[0][0].phonemes == v({"zh", "ong"}),
              "inner-text: phonemes");
    }

    // 3. explicit reading/path/group tree with alternative paths
    {
        const std::string src =
            "<word text=\"重\" language=\"zh\">"
            "<reading>"
            "<path><group script=\"chong\">"
            "<phoneme symbol=\"ch\"/><phoneme symbol=\"ong\"/></group></path>"
            "<path><group script=\"zhong\">"
            "<phoneme symbol=\"zh\"/><phoneme symbol=\"ong\"/></group></path>"
            "</reading></word>";
        const std::vector<Word> words = convert_pfml(pipeline, src, {});
        check(words.size() == 1, "tree: one word");
        check(words[0].readings.size() == 1, "tree: one reading");
        check(words[0].readings[0].paths.size() == 2, "tree: two paths");
        check(words[0].readings[0].paths[0][0].script == "chong", "tree: path0 script");
        check(words[0].readings[0].paths[1][0].phonemes == v({"zh", "ong"}),
              "tree: path1 phonemes");
        const std::string back = to_pfml(words);
        check(same_words(words, convert_pfml(pipeline, back, {})),
              "tree: round-trip model equal");
    }

    // 4. multiple readings
    {
        const std::string src =
            "<word text=\"重\" language=\"zh\">"
            "<reading><group script=\"chong\" phonemes=\"ch ong\"/></reading>"
            "<reading><group script=\"zhong\" phonemes=\"zh ong\"/></reading>"
            "</word>";
        const std::vector<Word> words = convert_pfml(pipeline, src, {});
        check(words.size() == 1 && words[0].readings.size() == 2, "readings: two");
        const std::string back = to_pfml(words);
        check(same_words(words, convert_pfml(pipeline, back, {})),
              "readings: round-trip model equal");
    }

    // 5. bare phoneme run at the fragment level forms one implicit word
    {
        const std::vector<Word> words =
            convert_pfml(pipeline, "<phoneme>AP</phoneme><phoneme>zh</phoneme>", {});
        check(words.size() == 1, "bare: one implicit word");
        check(words[0].readings[0].paths[0][0].phonemes == v({"AP", "zh"}), "bare: phonemes");
        const std::string back = to_pfml(words);
        check(same_words(words, convert_pfml(pipeline, back, {})),
              "bare: round-trip model equal");
    }

    // 6. explicit language on a phoneme becomes "<lang>/<symbol>"
    {
        const std::vector<Word> words = convert_pfml(
            pipeline,
            "<word text=\"x\"><group script=\"g\">"
            "<phoneme language=\"zh\" symbol=\"ong\"/></group></word>",
            {});
        check(words[0].readings[0].paths[0][0].phonemes == v({"zh/ong"}),
              "lang-attr: zh/ong");
    }

    // 7. entity escaping survives both directions
    {
        const std::vector<Word> words = convert_pfml(
            pipeline, "<word text=\"a&amp;b\" language=\"zh\" script=\"s\" phonemes=\"p\"/>", {});
        check(words[0].text == "a&b", "escape: decoded on parse");
        const std::string back = to_pfml(words);
        check(back.find("a&amp;b") != std::string::npos, "escape: re-encoded on write");
        check(same_words(words, convert_pfml(pipeline, back, {})), "escape: round-trip");
    }

    // 8. malformed markup is rejected, not silently dropped
    {
        bool threw = false;
        try {
            convert_pfml(pipeline, "<word text=\"x\"><bogus/></word>", {});
        } catch (const tifa_ggml::InvalidArgument &) {
            threw = true;
        }
        check(threw, "malformed: unknown element throws");
    }

    // 9. group without a script label fills it from the phonemes
    {
        const std::vector<Word> words = convert_pfml(
            pipeline, "<word text=\"x\"><group phonemes=\"a b\"/></word>", {});
        check(words[0].readings[0].paths[0][0].script == "a b", "label fill: script joined");
    }

    // 10. an explicit empty label is a value, not a missing one: the fill rules
    //     must not overwrite it, or the round trip would invent content.
    {
        const std::string src =
            "<word text=\"\" language=\"zh\" script=\"\" phonemes=\"a\"/>";
        const std::vector<Word> words = convert_pfml(pipeline, src, {});
        check(words[0].text.empty(), "explicit empty text is kept");
        check(words[0].readings[0].paths[0][0].script.empty(), "explicit empty script is kept");
        const std::string back = to_pfml(words);
        check(back == src, "explicit empty labels round-trip");
        if (back != src) std::cout << "      got: " << back << "\n";
    }

    // 11. every element the parser accepts must also be detected as PFML
    {
        check(looks_like_pfml(
                  "<reading><path><group><phoneme>a</phoneme></group></path></reading>"),
              "looks_like_pfml: reading/path/group are detected");
        check(looks_like_pfml("<group script=\"a\" phonemes=\"a\"/>"),
              "looks_like_pfml: a bare group is detected");
    }

    // 12. PFML 1.0: comments are ignored, and must not split a text run --
    //     otherwise "重<!-- x -->庆" would become two words instead of one.
    {
        const std::vector<Word> words = convert_pfml(
            pipeline,
            "<!-- lead --><word text=\"x\" language=\"zh\" script=\"s\" phonemes=\"p\"/>", {});
        check(words.size() == 1, "comment: ignored before a word");
        check(words[0].text == "x", "comment: word still parsed");

        const std::vector<Word> two = convert_pfml(
            pipeline,
            "<word text=\"重\" language=\"zh\" script=\"zhong\" phonemes=\"zh ong\"/>"
            "<!-- between -->"
            "<word text=\"庆\" language=\"zh\" script=\"qing\" phonemes=\"q ing\"/>", {});
        check(two.size() == 2, "comment: two words around a comment");
        check(two[0].text == "重" && two[1].text == "庆", "comment: both words kept");
    }

    // 13. PFML 1.0: CDATA is ordinary character data, taken literally (no
    //     entity decoding inside it).
    {
        const std::vector<Word> words = convert_pfml(
            pipeline,
            "<word language=\"zh\" script=\"s\" phonemes=\"p\"><![CDATA[重]]></word>", {});
        check(words.size() == 1 && words[0].text == "重", "cdata: used as the word text");

        const std::vector<Word> literal = convert_pfml(
            pipeline,
            "<word language=\"zh\" script=\"s\" phonemes=\"p\"><![CDATA[a&amp;b]]></word>", {});
        check(literal.size() == 1 && literal[0].text == "a&amp;b",
              "cdata: entities inside are literal");
    }

    // 14. an unknown attribute is an error, not something to ignore
    {
        const auto throws = [&](const std::string & src) {
            try { convert_pfml(pipeline, src, {}); }
            catch (const tifa_ggml::InvalidArgument &) { return true; }
            return false;
        };
        check(throws("<word text=\"x\" language=\"zh\" bogus=\"1\" script=\"s\" phonemes=\"p\"/>"),
              "attributes: unknown attribute on <word> throws");
        check(throws("<word text=\"x\" language=\"zh\" script=\"s\" phonemes=\"p\"/>"
                     "<group script=\"g\" phonemes=\"a\" bogus=\"1\"/>"),
              "attributes: unknown attribute on <group> throws");
        check(!throws("<word text=\"x\" language=\"zh\" script=\"s\" phonemes=\"p\"/>"),
              "attributes: the documented set is still accepted");
    }

    // 15. language-kind: only "any" is defined, it needs an absent language
    //     attribute, and it describes a direct pronunciation only.
    {
        const auto throws = [&](const std::string & src) {
            try { convert_pfml(pipeline, src, {}); }
            catch (const tifa_ggml::InvalidArgument &) { return true; }
            return false;
        };
        check(throws("<word text=\"x\" language-kind=\"bogus\" script=\"s\" phonemes=\"p\"/>"),
              "language-kind: an unknown value throws");
        check(throws("<word text=\"x\" language=\"zh\" language-kind=\"any\""
                     " script=\"s\" phonemes=\"p\"/>"),
              "language-kind: combined with a language attribute throws");
        check(!throws("<word text=\"x\" language-kind=\"any\" script=\"s\" phonemes=\"p\"/>"),
              "language-kind: \"any\" on a direct word is accepted");
        check(throws("<word language-kind=\"any\">plain</word>"),
              "language-kind: on a G2P word throws");
    }

    // 16. PFML 1.0 separates an omitted language (inherit) from an explicit
    //     empty one (clear).  Both must parse; the empty form is what lets a
    //     nested scope drop the inherited language.
    {
        const std::vector<Word> cleared = convert_pfml(
            pipeline,
            "<scope language=\"zh\"><scope language=\"\">"
            "<word text=\"x\" script=\"s\" phonemes=\"p\"/></scope></scope>", {});
        check(cleared.size() == 1 && cleared[0].text == "x",
              "language: an explicit empty scope language parses");
        const std::vector<Word> omitted = convert_pfml(
            pipeline,
            "<scope language=\"zh\"><word text=\"x\" script=\"s\" phonemes=\"p\"/></scope>", {});
        check(omitted.size() == 1 && omitted[0].text == "x",
              "language: an omitted language still parses");
    }

    std::cout << (failures == 0 ? "\nALL PASS\n" : "\nFAILURES\n");
    return failures == 0 ? 0 : 1;
}
