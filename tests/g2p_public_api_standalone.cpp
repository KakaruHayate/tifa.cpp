// Exercises the PUBLIC G2P/PFML API (include/tifa_ggml/g2p.h) exactly the way a
// downstream consumer does: public header only, linked against tifa_ggml_g2p.
// Builds with no ggml, no model and no GGUF.

#include "tifa_ggml/g2p.h"

#include <iostream>
#include <string>
#include <vector>

using namespace tifa_ggml;

static int failures = 0;

static void check(bool ok, const std::string & what) {
    std::cout << (ok ? "ok:   " : "FAIL: ") << what << "\n";
    if (!ok) ++failures;
}

int main() {
    G2PPipeline pipeline = G2PPipeline::from_config("{}", "");

    // --- PFML detection -----------------------------------------------------
    const std::string src =
        "<word text=\"重\" language=\"zh\" script=\"zhong\" phonemes=\"zh ong\"/>";
    check(looks_like_pfml(src), "looks_like_pfml: markup detected");
    check(!looks_like_pfml("plain prose, no tags"), "looks_like_pfml: prose rejected");

    // --- PFML -> model ------------------------------------------------------
    std::vector<G2PWord> words = pipeline.convert_pfml(src, {});
    check(words.size() == 1, "convert_pfml: one word");
    check(words[0].text == "重", "convert_pfml: text");
    check(words[0].language == "zh", "convert_pfml: language");
    check(words[0].readings.size() == 1 && words[0].readings[0].paths.size() == 1 &&
              words[0].readings[0].paths[0].size() == 1,
          "convert_pfml: one reading / path / group");
    check(words[0].readings[0].paths[0][0].phonemes == std::vector<std::string>({"zh", "ong"}),
          "convert_pfml: phonemes");

    // --- model -> PFML, verbatim for the compact form -----------------------
    const std::string back = to_pfml(words);
    check(back == src, "to_pfml: compact round-trip is byte-identical");
    if (back != src) std::cout << "      got: " << back << "\n";

    // --- alternatives force the explicit tree -------------------------------
    G2PWord polyphone;
    polyphone.text     = "重";
    polyphone.language = "zh";
    G2PReading reading;
    reading.paths.push_back(G2PPath{ G2PGroup{ "chong", { "ch", "ong" } } });
    reading.paths.push_back(G2PPath{ G2PGroup{ "zhong", { "zh", "ong" } } });
    polyphone.readings.push_back(reading);

    const std::string tree = to_pfml({ polyphone });
    check(tree.find("<reading>") != std::string::npos, "to_pfml: tree form for alternatives");
    check(tree.find("<path>") != std::string::npos, "to_pfml: paths emitted");
    check(tree.find("symbol=\"ch\"") != std::string::npos, "to_pfml: phonemes emitted");

    const std::vector<G2PWord> round = pipeline.convert_pfml(tree, {});
    check(round.size() == 1 && round[0].readings.size() == 1 &&
              round[0].readings[0].paths.size() == 2,
          "tree round-trip: two paths preserved");
    check(round[0].readings[0].paths[0][0].script == "chong", "tree round-trip: path 0 script");
    check(to_pfml(round) == tree, "tree round-trip: byte-identical");

    // --- candidate projection (what a pronunciation picker lists) -----------
    {
        G2PPipeline passthrough = G2PPipeline::from_config(
            "{\"converters\":[{\"id\":\"passthrough\"}]}", "");
        const std::vector<G2PWordCandidates> listed = candidates(passthrough, "hi", {});
        check(!listed.empty(), "candidates: words projected");
        check(!listed.empty() && listed[0].candidates.size() == 1,
              "candidates: unambiguous word has exactly one candidate");
        check(!listed.empty() && !listed[0].candidates[0].phonemes.empty(),
              "candidates: phonemes captured");
    }

    // --- an inert pipeline refuses text rather than silently returning none --
    {
        bool threw = false;
        try {
            pipeline.convert("text", {});
        } catch (const tifa_ggml::InvalidArgument &) {
            threw = true;
        }
        check(threw, "inert pipeline: text conversion is refused, not silently empty");
    }

    // --- phoneme validation -------------------------------------------------
    const G2PLookup lookup = [](const std::string & symbol) -> int {
        return symbol == "zh/ong" || symbol == "ong" ? 7 : -1;
    };
    check(resolve_phoneme("ong", { "zh" }, lookup).id == 7, "resolve: known phoneme");
    check(resolve_phoneme("zh/ong", { "zh" }, lookup).id == 7, "resolve: language-qualified");
    check(resolve_phoneme("nope", { "zh" }, lookup).id == -1, "resolve: unknown phoneme flagged");

    std::cout << (failures == 0 ? "\nALL PASS\n" : "\nFAILURES\n");
    return failures == 0 ? 0 : 1;
}
