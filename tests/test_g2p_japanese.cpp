// Japanese G2P: the `japanese-mecab` converter and its no-UniDic fallback.
//
// With a MeCab/UniDic dictionary the converter mirrors g2pflow's
// JapaneseMecabConverter: MeCab segments the text, UniDic's `pron` column
// supplies whole-word kana readings, and the kana converter turns those into
// phonemes.  Without a dictionary the kana converter serves the same config
// id, so kana lyrics keep working on the stock model (issue #10).
//
// The MeCab-backed tests need a compiled UniDic dicdir; they are gated on
// TIFA_TEST_UNIDIC_DIR like the FBL/LSTM goldens and skipped in CI.

#include "g2p/g2p.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace tifa_ggml::internal::g2p;

namespace {

// A minimal romaji->phoneme dictionary in the `key\tp1 p2` format of
// dictionaries/japanese_dict_full.txt, covering every sentence used below.
constexpr const char * k_mini_dict =
    "a\ta\ni\ti\nu\tu\ne\te\no\to\ncl\tcl\nn\tN\n"
    "ka\tk a\nki\tk i\nku\tk u\nke\tk e\nko\tk o\n"
    "sa\ts a\nsi\ts i\nsu\ts u\nse\ts e\nso\ts o\n"
    "ta\tt a\nti\tt i\ntu\tt u\nte\tt e\nto\tt o\n"
    "na\tn a\nni\tn i\nnu\tn u\nne\tn e\nno\tn o\n"
    "ha\th a\nhi\th i\nhu\th u\nhe\th e\nho\th o\n"
    "ma\tm a\nmi\tm i\nmu\tm u\nme\tm e\nmo\tm o\n"
    "ya\ty a\nyu\ty u\nyo\ty o\n"
    "ra\tr a\nri\tr i\nru\tr u\nre\tr e\nro\tr o\n"
    "wa\tw a\nwo\tw o\n"
    "ga\tg a\ngi\tg i\ngu\tg u\nge\tg e\ngo\tg o\n"
    "za\tz a\nzi\tz i\nzu\tz u\nze\tz e\nzo\tz o\n"
    "da\td a\ndi\td i\ndu\td u\nde\td e\ndo\td o\n"
    "ba\tb a\nbi\tb i\nbu\tb u\nbe\tb e\nbo\tb o\n"
    "pa\tp a\npi\tp i\npu\tp u\npe\tp e\npo\tp o\n"
    "kya\tky a\nsha\tsh a\nshu\tsh u\nsho\tsh o\n"
    "cha\tch a\nchu\tch u\ncho\tch o\n"
    "nya\tny a\nhyu\thy u\nmya\tmy a\nrya\try a\n"
    "gya\tgy a\nbya\tby a\npya\tpy a\n"
    "tsu\tts u\n";

std::string write_mini_dict() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "tifa_g2p_japanese_test";
    std::filesystem::create_directories(dir);
    const std::filesystem::path file = dir / "japanese_dict_full.txt";
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << k_mini_dict;
    return file.generic_string();
}

// `path` with backslashes swapped for forward slashes so it survives inside
// JSON without escaping.
std::string json_path(const std::string & path) {
    std::string out = path;
    for (char & c : out) {
        if (c == '\\') c = '/';
    }
    return out;
}

Pipeline mecab_pipeline(const std::string & dict_path, const std::string & unidic_dir,
                        bool double_written_sokuon = false) {
    std::string config =
        std::string(R"({"converters":[{"id":"japanese-mecab","kwargs":{"dict_path":")") +
        json_path(dict_path) + "\",\"nbest\":32,\"double_written_sokuon\":" +
        (double_written_sokuon ? "true" : "false");
    if (!unidic_dir.empty()) {
        config += ",\"unidic_dir\":\"" + json_path(unidic_dir) + "\"";
    }
    config += "}}]}";
    return Pipeline::from_config(config, "");
}

std::string first_reading_phonemes(const Word & word) {
    std::string out;
    for (const Path & path : word.readings.front().paths) {
        for (const Group & group : path) {
            for (const std::string & phoneme : group.phonemes) {
                if (!out.empty()) out += ' ';
                out += phoneme;
            }
        }
    }
    return out;
}

std::vector<std::string> word_texts(const std::vector<Word> & words) {
    std::vector<std::string> texts;
    texts.reserve(words.size());
    for (const Word & word : words) texts.push_back(word.text);
    return texts;
}

}  // namespace

// Without a UniDic dictionary the mecab id degrades to the kana converter:
// kana lyrics still align on the stock model (issue #10's fix).
TEST(Japanese, MecabIdServesKanaWithoutUnidic) {
    const Pipeline pipeline = mecab_pipeline(write_mini_dict(), "");
    const std::vector<Word> words = pipeline.convert("かわいい", {});
    const std::vector<std::string> texts = word_texts(words);
    ASSERT_EQ(texts.size(), 4u);
    EXPECT_EQ(texts[0], "か");
    EXPECT_EQ(texts[3], "い");
    EXPECT_EQ(first_reading_phonemes(words[0]), "k a");
    EXPECT_EQ(first_reading_phonemes(words[3]), "i");
}

// With UniDic, MeCab segments the sentence and the morpheme surfaces (kanji
// included) become the word texts.  Readings come from UniDic `pron`; the
// long-vowel mark in カワイー contributes nothing, exactly like upstream.
TEST(Japanese, MecabReadsKanji) {
    const char * unidic = std::getenv("TIFA_TEST_UNIDIC_DIR");
    if (unidic == nullptr || *unidic == '\0') {
        GTEST_SKIP() << "TIFA_TEST_UNIDIC_DIR not set";
    }
    const Pipeline pipeline = mecab_pipeline(write_mini_dict(), unidic);
    const std::vector<Word> words =
        pipeline.convert("可愛いから好きになったなんて", {});
    ASSERT_EQ(word_texts(words),
              (std::vector<std::string>{
                  "可愛い", "から", "好き", "に", "なっ", "た", "なんて" }));
    EXPECT_EQ(first_reading_phonemes(words[0]), "k a w a i");
    EXPECT_EQ(first_reading_phonemes(words[1]), "k a r a");
    EXPECT_EQ(first_reading_phonemes(words[2]), "s u k i");
    EXPECT_EQ(first_reading_phonemes(words[3]), "n i");
    EXPECT_EQ(first_reading_phonemes(words[4]), "n a cl");
    EXPECT_EQ(first_reading_phonemes(words[5]), "t a");
    EXPECT_EQ(first_reading_phonemes(words[6]), "n a N t e");
}

// UniDic can offer several whole-word readings; 可愛い has two single-morpheme
// parses in unidic-lite 1.0.8 and both survive as reading candidates.
TEST(Japanese, MecabKeepsReadingAlternatives) {
    const char * unidic = std::getenv("TIFA_TEST_UNIDIC_DIR");
    if (unidic == nullptr || *unidic == '\0') {
        GTEST_SKIP() << "TIFA_TEST_UNIDIC_DIR not set";
    }
    const Pipeline pipeline = mecab_pipeline(write_mini_dict(), unidic);
    const std::vector<Word> words = pipeline.convert("現在", {});
    ASSERT_EQ(word_texts(words), std::vector<std::string>{"現在"});
    ASSERT_GE(words[0].readings.size(), 1u);
    EXPECT_EQ(first_reading_phonemes(words[0]), "g e N z a i");
}

// double_written_sokuon merges a っ-final morpheme with the next one so the
// geminate stays inside one path choice (upstream's sokuon product).
TEST(Japanese, MecabMergesSokuonWhenAsked) {
    const char * unidic = std::getenv("TIFA_TEST_UNIDIC_DIR");
    if (unidic == nullptr || *unidic == '\0') {
        GTEST_SKIP() << "TIFA_TEST_UNIDIC_DIR not set";
    }
    const Pipeline pipeline =
        mecab_pipeline(write_mini_dict(), unidic, /*double_written_sokuon=*/true);
    const std::vector<Word> words = pipeline.convert("なった", {});
    ASSERT_EQ(word_texts(words), std::vector<std::string>{"なった"});
    EXPECT_EQ(first_reading_phonemes(words[0]), "n a t t a");
}
