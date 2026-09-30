// Pinyin engine: dictionary-driven phrase disambiguation over hanzi input.
//
// Faithful port of the cpp-pinyin engine bundled with openvpi/TIFA:
//   g2p/converters/cpp_pinyin/engine.py       (EngineImpl::queryRaw)
//   g2p/converters/cpp_pinyin/dict_loader.py  (DictUtil file formats)
//   g2p/converters/cpp_pinyin/tones.py        (tone folding, STYLE_NORMAL)
//
// The released converter (g2p/converters/chinese.py) only asks for
// STYLE_NORMAL output, so the public surface here is restricted to that style:
// tone marks are folded to their base letter, trailing TONE3 digits are
// dropped and "ü" is written as "v".

#include "g2p.h"

#include "tifa_ggml/errors.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tifa_ggml::internal::g2p {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

namespace {

constexpr char32_t k_bmp_limit = 0xFFFF;

// One UTF-8 code point starting at s[i]; `i` advances past it.  Malformed or
// truncated bytes yield the offending byte so parsing always makes progress.
char32_t decode_utf8(const std::string & s, std::size_t & i) {
    const unsigned char b0 = static_cast<unsigned char>(s[i]);
    std::size_t extra;
    char32_t    cp;
    if (b0 < 0x80) {
        extra = 0;
        cp    = b0;
    } else if ((b0 & 0xE0) == 0xC0) {
        extra = 1;
        cp    = b0 & 0x1Fu;
    } else if ((b0 & 0xF0) == 0xE0) {
        extra = 2;
        cp    = b0 & 0x0Fu;
    } else if ((b0 & 0xF8) == 0xF0) {
        extra = 3;
        cp    = b0 & 0x07u;
    } else {
        ++i;
        return b0;
    }
    if (i + extra >= s.size()) {              // truncated tail
        ++i;
        return b0;
    }
    for (std::size_t k = 1; k <= extra; ++k) {
        const unsigned char b = static_cast<unsigned char>(s[i + k]);
        if ((b & 0xC0) != 0x80) {             // not a continuation byte
            ++i;
            return b0;
        }
        cp = (cp << 6) | (b & 0x3Fu);
    }
    i += extra + 1;
    return cp;
}

std::vector<char32_t> utf8_to_codepoints(const std::string & s) {
    std::vector<char32_t> out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        out.push_back(decode_utf8(s, i));
    }
    return out;
}

void append_utf8(std::string & out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string to_utf8(char32_t cp) {
    std::string out;
    append_utf8(out, cp);
    return out;
}

// ---------------------------------------------------------------------------
// Tone conversion (tones.py, STYLE_NORMAL only)
// ---------------------------------------------------------------------------

// ManTone.cpp / tones.py tone_map: tone-marked vowel -> base letter.  The tone
// digit is irrelevant for STYLE_NORMAL, only the base letter is kept.
const std::unordered_map<char32_t, const char *> & tone_mark_table() {
    static const std::unordered_map<char32_t, const char *> table = {
        { 0x0101, "a" }, { 0x00E1, "a" }, { 0x01CE, "a" }, { 0x00E0, "a" },
        { 0x0113, "e" }, { 0x00E9, "e" }, { 0x011B, "e" }, { 0x00E8, "e" },
        { 0x012B, "i" }, { 0x00ED, "i" }, { 0x01D0, "i" }, { 0x00EC, "i" },
        { 0x014D, "o" }, { 0x00F3, "o" }, { 0x01D2, "o" }, { 0x00F2, "o" },
        { 0x016B, "u" }, { 0x00FA, "u" }, { 0x01D4, "u" }, { 0x00F9, "u" },
        { 0x01D6, "v" }, { 0x01D8, "v" }, { 0x01DA, "v" }, { 0x01DC, "v" },
        { 0x00FC, "v" },                                        // plain ü
        { 0x0144, "n" }, { 0x0148, "n" }, { 0x01F9, "n" },      // syllabic n
        { 0x1E3F, "m" },                                        // syllabic m
    };
    return table;
}

bool is_ascii_digit(char c) {
    return c >= '0' && c <= '9';
}

// tones.py: apply_tone(pinyin, STYLE_NORMAL).  A trailing digit marks TONE3
// input (user_dict.txt) and is dropped; everything else is mapped through the
// tone table, which also turns "ü" into "v" (v_to_u is false downstream).
std::string to_style_normal(const std::string & pinyin) {
    std::string src = pinyin;
    if (!src.empty() && is_ascii_digit(src.back())) {
        src.pop_back();
    }

    const auto &          table = tone_mark_table();
    std::string           out;
    out.reserve(src.size());
    for (std::size_t i = 0; i < src.size();) {
        const char32_t cp = decode_utf8(src, i);
        const auto     it = table.find(cp);
        if (it != table.end()) {
            out += it->second;
        } else {
            append_utf8(out, cp);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Dictionary loading (dict_loader.py)
// ---------------------------------------------------------------------------

// word.txt:            char   -> readings (tone-marked)
using WordMap = std::unordered_map<char32_t, std::vector<std::string>>;
// phrases_dict.txt + user_dict.txt: encoded phrase key -> one reading sequence
using PhraseMap = std::unordered_map<uint64_t, std::vector<std::string>>;
// trans_word.txt: traditional -> simplified
using TransMap = std::unordered_map<char32_t, char32_t>;

bool is_blank(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// `line.strip()`: dictionary files only ever carry ASCII whitespace around the
// payload, so this matches the Python behaviour on them.
std::string strip_ascii(const std::string & s) {
    std::size_t begin = 0;
    std::size_t end   = s.size();
    while (begin < end && is_blank(s[begin])) ++begin;
    while (end > begin && is_blank(s[end - 1])) --end;
    return s.substr(begin, end - begin);
}

// `line.split(":", 1)`
bool split_key_value(const std::string & line, std::string & key, std::string & value) {
    const std::size_t pos = line.find(':');
    if (pos == std::string::npos) return false;
    key   = line.substr(0, pos);
    value = line.substr(pos + 1);
    return true;
}

// Split on `sep`, stripping each field; empty fields are dropped, matching the
// `[p.strip() for p in s.split(sep) if p.strip()]` idiom.
std::vector<std::string> split_fields(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::size_t              pos = 0;
    while (true) {
        const std::size_t stop = s.find(sep, pos);
        std::string       item =
            strip_ascii(s.substr(pos, stop == std::string::npos ? std::string::npos : stop - pos));
        if (!item.empty()) out.push_back(std::move(item));
        if (stop == std::string::npos) break;
        pos = stop + 1;
    }
    return out;
}

// Whitespace-separated fields (`str.split()`), used by user_dict.txt.
std::vector<std::string> split_whitespace(const std::string & s) {
    std::vector<std::string> out;
    std::size_t              pos = 0;
    while (pos < s.size()) {
        while (pos < s.size() && is_blank(s[pos])) ++pos;
        const std::size_t begin = pos;
        while (pos < s.size() && !is_blank(s[pos])) ++pos;
        if (pos > begin) out.push_back(s.substr(begin, pos - begin));
    }
    return out;
}

// Iterate text-mode lines: python's text reader splits on \n as well as lone
// \r, and empty lines are ignored by every loader below.
template <typename Fn>
void for_each_line(const std::string & text, Fn fn) {
    std::size_t pos = 0;
    while (pos <= text.size()) {
        std::size_t stop = pos;
        while (stop < text.size() && text[stop] != '\n' && text[stop] != '\r') ++stop;
        fn(text.substr(pos, stop - pos));
        if (stop >= text.size()) break;
        pos = stop + 1;
    }
}

// dict_loader.py: encode_phrase_key() -- shift 16, OR.  cpp-pinyin builds the
// key from char16_t units, so only the low 16 bits of each code point taking
// part (every phrase in the shipped dictionaries is BMP).
uint64_t encode_phrase_key(const std::vector<char32_t> & chars) {
    uint64_t key = 0;
    for (char32_t ch : chars) {
        key = (key << 16) | static_cast<uint64_t>(ch & k_bmp_limit);
    }
    return key;
}

// Reads a dictionary file.  `required` files raise an actionable error; the
// optional ones (user_dict.txt) are simply skipped, as in Python.
std::string read_dict_file(const std::string & dir, const char * name, bool required) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/' && path.back() != '\\') path += '/';
    path += name;

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (required) {
            throw InvalidArgument(
                std::string("PinyinEngine: cannot open dictionary file '") + path +
                "' (word.txt, phrases_dict.txt, phrases_map.txt and trans_word.txt are "
                "required; user_dict.txt is optional)");
        }
        return std::string();
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

// `字:拼音1,拼音2` -> char -> list of readings.  A character whose reading list
// comes out empty is kept empty on purpose: the Python lookup treats an empty
// list as "not in the dictionary".
WordMap load_word_dict(const std::string & text) {
    WordMap out;
    for_each_line(text, [&](const std::string & raw) {
        const std::string line = strip_ascii(raw);
        if (line.empty()) return;

        std::string key;
        std::string prons;
        if (!split_key_value(line, key, prons)) return;
        if (key.empty() || prons.empty()) return;

        const std::vector<char32_t> chars = utf8_to_codepoints(key);
        if (chars.size() != 1) return;      // keys are single characters

        out[chars[0]] = split_fields(prons, ',');
    });
    return out;
}

// `词组:拼音1,拼音2,拼音3` -> phrase key -> reading sequence.  Only the first
// sequence stored for a key is ever read back (engine.py uses `it[0]`), so the
// later ones are dropped here.
PhraseMap load_phrase_dict(const std::string & text) {
    PhraseMap out;
    for_each_line(text, [&](const std::string & raw) {
        const std::string line = strip_ascii(raw);
        if (line.empty()) return;

        std::string key;
        std::string prons;
        if (!split_key_value(line, key, prons)) return;
        if (prons.empty()) return;

        const std::vector<char32_t> chars = utf8_to_codepoints(key);
        // query_raw only ever looks up windows of 2-4 characters, and keys
        // built from anything else can never be reproduced (cpp-pinyin only
        // supports 2-4 char phrases), so those entries are unreachable.
        if (chars.size() < 2 || chars.size() > 4) return;
        if (std::any_of(chars.begin(), chars.end(),
                        [](char32_t c) { return c > k_bmp_limit; })) {
            return;
        }

        const std::vector<std::string> readings = split_fields(prons, ',');
        if (readings.size() != chars.size()) return;

        const uint64_t encoded = encode_phrase_key(chars);
        if (out.find(encoded) == out.end()) {
            out.emplace(encoded, readings);
        }
    });
    return out;
}

// `字:digits` -> set of polyphonic characters.  The Python side keeps whole
// strings, which can only ever match single-character lookups.
std::unordered_set<char32_t> load_phrase_map(const std::string & text) {
    std::unordered_set<char32_t> out;
    for_each_line(text, [&](const std::string & raw) {
        const std::string line = strip_ascii(raw);
        if (line.empty()) return;

        std::string key;
        std::string digits;
        if (!split_key_value(line, key, digits)) return;
        if (key.empty()) return;

        const std::vector<char32_t> chars = utf8_to_codepoints(key);
        if (chars.size() != 1) return;
        out.insert(chars[0]);
    });
    return out;
}

// `繁:简` -> traditional -> simplified.  Every entry in the shipped dictionary
// maps to a single character; a multi-character right-hand side cannot be
// represented by simplify()'s char32_t output, so its first character is used.
TransMap load_trans_dict(const std::string & text) {
    TransMap out;
    for_each_line(text, [&](const std::string & raw) {
        const std::string line = strip_ascii(raw);
        if (line.empty()) return;

        std::string key;
        std::string value;
        if (!split_key_value(line, key, value)) return;
        if (key.empty() || value.empty()) return;

        const std::vector<char32_t> lhs = utf8_to_codepoints(key);
        const std::vector<char32_t> rhs = utf8_to_codepoints(value);
        if (lhs.size() != 1 || rhs.empty()) return;

        out[lhs[0]] = rhs[0];
    });
    return out;
}

// `词组:拼音1 拼音2` (space separated, TONE3) -> the same shape as
// phrases_dict.txt.
PhraseMap load_user_dict(const std::string & text) {
    PhraseMap out;
    for_each_line(text, [&](const std::string & raw) {
        const std::string line = strip_ascii(raw);
        if (line.empty()) return;

        std::string key;
        std::string prons;
        if (!split_key_value(line, key, prons)) return;
        if (prons.empty()) return;

        const std::vector<char32_t> chars = utf8_to_codepoints(key);
        if (chars.size() < 2 || chars.size() > 4) return;
        if (std::any_of(chars.begin(), chars.end(),
                        [](char32_t c) { return c > k_bmp_limit; })) {
            return;
        }

        const std::vector<std::string> readings = split_whitespace(prons);
        if (readings.size() != chars.size()) return;

        const uint64_t encoded = encode_phrase_key(chars);
        if (out.find(encoded) == out.end()) {
            out.emplace(encoded, readings);
        }
    });
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// PinyinEngine
// ---------------------------------------------------------------------------

struct PinyinEngine::Impl {
    WordMap                      words;       // word.txt
    PhraseMap                    phrases;     // phrases_dict.txt + user_dict.txt
    std::unordered_set<char32_t> polyphonic;  // phrases_map.txt
    TransMap                     trans;       // trans_word.txt
};

PinyinEngine::PinyinEngine(const std::string & dict_dir)
    : impl_(std::make_shared<Impl>()) {
    impl_->words = load_word_dict(read_dict_file(dict_dir, "word.txt", true));

    impl_->phrases = load_phrase_dict(read_dict_file(dict_dir, "phrases_dict.txt", true));
    // User entries only fill in phrases the built-in dictionary does not know;
    // an existing key keeps its phrases_dict.txt reading (Python's `extend`).
    const PhraseMap user =
        load_user_dict(read_dict_file(dict_dir, "user_dict.txt", false));
    for (const auto & entry : user) {
        impl_->phrases.insert(entry);
    }

    impl_->polyphonic = load_phrase_map(read_dict_file(dict_dir, "phrases_map.txt", true));
    impl_->trans      = load_trans_dict(read_dict_file(dict_dir, "trans_word.txt", true));
}

bool PinyinEngine::ready() const noexcept {
    return impl_ && !impl_->words.empty();
}

std::vector<char32_t> PinyinEngine::simplify(const std::vector<char32_t> & chars) const {
    std::vector<char32_t> out;
    out.reserve(chars.size());
    for (char32_t ch : chars) {
        const auto it = impl_->trans.find(ch);
        out.push_back(it == impl_->trans.end() ? ch : it->second);
    }
    return out;
}

std::vector<std::vector<std::string>> PinyinEngine::query_raw(
    const std::vector<char32_t> & chars) const
{
    const Impl &                          im = *impl_;
    std::vector<std::vector<std::string>> result;
    const long                            n      = static_cast<long>(chars.size());
    long                                  cursor = 0;

    while (cursor < n) {
        const char32_t ch = chars[static_cast<std::size_t>(cursor)];
        const auto     it = im.words.find(ch);

        // ---- not in the dictionary: pass the character through ----
        if (it == im.words.end() || it->second.empty()) {
            result.push_back({ to_utf8(ch) });
            ++cursor;
            continue;
        }

        const std::vector<std::string> & candidates = it->second;
        const std::string                fallback   = to_style_normal(candidates[0]);

        // ---- not polyphonic: fast path (first pronunciation) ----
        if (im.polyphonic.find(ch) == im.polyphonic.end()) {
            result.push_back({ fallback });
            ++cursor;
            continue;
        }

        // ---- polyphonic: sliding-window phrase matching ----
        bool found = false;

        // Mirrors cpp-pinyin's emitPhrase / engine.py's _emit(): window
        // [begin, end) is looked up by its encoded key, its reading is written
        // at `result_idx` (overwriting when that slot already exists, appending
        // otherwise), the cursor advances by `advance` and `pop_last` drops the
        // trailing result first (the 1-backward direction).
        const auto emit = [&](long begin, long end, long result_idx, long advance,
                              bool pop_last) -> bool {
            const std::vector<char32_t> window(chars.begin() + begin, chars.begin() + end);
            const auto                  pit = im.phrases.find(encode_phrase_key(window));
            if (pit == im.phrases.end()) {
                return false;               // no such phrase: try the next direction
            }

            std::vector<std::string> prons;
            prons.reserve(pit->second.size());
            for (const std::string & p : pit->second) {
                prons.push_back(to_style_normal(p));
            }

            if (pop_last && !result.empty()) {
                result.pop_back();
            }
            for (std::size_t i = 0; i < prons.size(); ++i) {
                const long idx  = result_idx + static_cast<long>(i);
                const long size = static_cast<long>(result.size());
                if (idx < size) {
                    long at = idx < 0 ? idx + size : idx;   // python style negative index
                    if (at < 0) at = 0;
                    result[static_cast<std::size_t>(at)] = { prons[i] };
                } else {
                    result.push_back({ prons[i] });
                }
            }
            cursor += advance;
            found = true;
            return true;
        };

        for (long length = 4; length >= 2 && !found; --length) {
            // direction 1: forward -- chars[cursor, cursor+length)
            if (cursor + length <= n) {
                emit(cursor, cursor + length, static_cast<long>(result.size()), length, false);
            }
            // direction 2: 1-backward -- chars[cursor-1, cursor-1+length),
            // clipped by the end of the input (python slice semantics)
            if (!found && cursor >= 1) {
                const long begin = cursor - 1;
                const long end   = std::min(begin + length, n);
                emit(begin, end, static_cast<long>(result.size()) - 1, length - 1, true);
            }
            // direction 3: 1-forward -- chars[cursor+1-length, cursor+1)
            if (!found && cursor + 1 >= length && cursor + 1 <= n) {
                const long begin = cursor + 1 - length;
                emit(begin, cursor + 1, begin, 1, false);
            }
            // direction 4: 2-forward -- chars[cursor+2-length, cursor+2)
            if (!found && cursor + 2 >= length && cursor + 2 <= n) {
                const long begin = cursor + 2 - length;
                emit(begin, cursor + 2, begin, 2, false);
            }
        }

        // ---- no phrase matched: default pronunciation ----
        if (!found) {
            result.push_back({ fallback });
            ++cursor;
        }
    }

    return result;
}

std::vector<std::string> PinyinEngine::readings(char32_t ch) const {
    const auto it = impl_->trans.find(ch);
    const char32_t simplified = it == impl_->trans.end() ? ch : it->second;

    const auto wit = impl_->words.find(simplified);
    if (wit == impl_->words.end() || wit->second.empty()) {
        return { to_utf8(ch) };             // unknown: the character itself
    }

    std::vector<std::string> out;
    out.reserve(wit->second.size());
    for (const std::string & pron : wit->second) {
        out.push_back(to_style_normal(pron));
    }
    return out;
}

}  // namespace tifa_ggml::internal::g2p
