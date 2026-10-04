// G2P pipeline: preprocessors, converters and the JSON-driven factory.
//
// Port of openvpi/TIFA's `g2p` package (`g2p/pipeline.py`,
// `g2p/converters/*.py`, `g2p/preprocessors/simple.py`).  The pipeline routes
// each preprocessed text fragment to the first active converter that claims a
// substring; claimed runs are converted into words with alternative readings.

#include "g2p.h"

#include "json.h"
#ifndef TIFA_G2P_NO_LSTM
#include "lstm_g2p.h"
#endif
#include "mecab_g2p.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tifa_ggml::internal::g2p {

namespace {

// ---------------------------------------------------------------------------
// utf-8 <-> code point helpers
// ---------------------------------------------------------------------------

// Decodes one code point at `i` (which must be < s.size()); `i` advances past
// it.  Malformed bytes decode as U+FFFD, one byte at a time.
char32_t decode_utf8(const std::string & s, std::size_t & i) {
    const auto byte = [&](std::size_t k) {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(s[k]));
    };
    const std::uint32_t b0 = byte(i);
    if (b0 < 0x80) {
        ++i;
        return static_cast<char32_t>(b0);
    }
    std::size_t extra = 0;
    std::uint32_t cp  = 0;
    if ((b0 & 0xE0) == 0xC0) {
        extra = 1;
        cp    = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
        extra = 2;
        cp    = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
        extra = 3;
        cp    = b0 & 0x07;
    } else {
        ++i;
        return static_cast<char32_t>(0xFFFD);
    }
    if (i + extra >= s.size()) {
        ++i;
        return static_cast<char32_t>(0xFFFD);
    }
    for (std::size_t k = 1; k <= extra; ++k) {
        const std::uint32_t bk = byte(i + k);
        if ((bk & 0xC0) != 0x80) {
            ++i;
            return static_cast<char32_t>(0xFFFD);
        }
        cp = (cp << 6) | (bk & 0x3F);
    }
    i += extra + 1;
    return static_cast<char32_t>(cp);
}

void append_utf8(std::string & out, char32_t cp) {
    const std::uint32_t c = static_cast<std::uint32_t>(cp);
    if (c < 0x80) {
        out.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (c >> 6)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (c >> 12)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (c >> 18)));
        out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
}

std::u32string to_u32(const std::string & s) {
    std::u32string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        out.push_back(decode_utf8(s, i));
    }
    return out;
}

std::string to_utf8(const std::u32string & s) {
    std::string out;
    out.reserve(s.size());
    for (char32_t cp : s) {
        append_utf8(out, cp);
    }
    return out;
}

std::string to_utf8(char32_t cp) {
    std::string out;
    append_utf8(out, cp);
    return out;
}

// ---------------------------------------------------------------------------
// character classes
// ---------------------------------------------------------------------------

bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Python's str.isspace() over the code points that appear in real text.
bool is_space32(char32_t c) {
    switch (c) {
        case 0x0009: case 0x000A: case 0x000B: case 0x000C: case 0x000D:
        case 0x001C: case 0x001D: case 0x001E: case 0x001F: case 0x0020:
        case 0x0085: case 0x00A0: case 0x1680:
        case 0x2000: case 0x2001: case 0x2002: case 0x2003: case 0x2004:
        case 0x2005: case 0x2006: case 0x2007: case 0x2008: case 0x2009:
        case 0x200A: case 0x2028: case 0x2029: case 0x202F: case 0x205F:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

bool is_blank(const std::u32string & s) {
    for (char32_t c : s) {
        if (!is_space32(c)) return false;
    }
    return true;
}

bool is_hanzi(char32_t c) {
    return c >= 0x4E00 && c <= 0x9FA5;
}

bool is_kana(char32_t c) {
    return (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
}

// g2p/converters/japanese.py:_is_japanese_char -- the claim range of the
// mecab converter: kana plus the kanji blocks (and 々〆〇).  In particular it
// must never take ASCII romaji/phoneme input from the Japanese dictionary
// converter used by existing training datasets (upstream comment).
bool is_japanese_input(char32_t c) {
    if (is_kana(c)) return true;
    if (c == 0x3005 || c == 0x3006 || c == 0x3007) return true;  // 々〆〇
    return (c >= 0x3400 && c <= 0x4DBF)
        || (c >= 0x4E00 && c <= 0x9FFF)
        || (c >= 0xF900 && c <= 0xFAFF)
        || (c >= 0x20000 && c <= 0x2FA1F)
        || (c >= 0x30000 && c <= 0x323AF);
}

bool is_small_kana(char32_t c) {
    switch (c) {
        case 0x3083: case 0x3085: case 0x3087:                            // ゃゅょ
        case 0x30E3: case 0x30E5: case 0x30E7:                            // ャュョ
        case 0x3041: case 0x3043: case 0x3045: case 0x3047: case 0x3049:  // ぁぃぅぇぉ
        case 0x30A1: case 0x30A3: case 0x30A5: case 0x30A7: case 0x30A9:  // ァィゥェォ
            return true;
        default:
            return false;
    }
}

// g2p/converters/text.py:_is_cjk
bool is_cjk(char32_t c) {
    return (c >= 0x2E80 && c <= 0x2EFF)
        || (c >= 0x2F00 && c <= 0x2FDF)
        || (c >= 0x3000 && c <= 0x303F)
        || is_kana(c)
        || (c >= 0x31F0 && c <= 0x31FF)
        || (c >= 0x3400 && c <= 0x4DBF)
        || (c >= 0x4E00 && c <= 0x9FFF)
        || (c >= 0xAC00 && c <= 0xD7AF)
        || (c >= 0xF900 && c <= 0xFAFF)
        || (c >= 0xFF00 && c <= 0xFFEF);
}

bool is_ascii_letter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// japanese.py:_CONSONANT_LEADING -- an ascii letter that can open a syllable.
bool is_consonant_leading(char c) {
    return is_ascii_letter(c) && std::string("aeiouAEIOU").find(c) == std::string::npos;
}

// ---------------------------------------------------------------------------
// case mapping
// ---------------------------------------------------------------------------

// Python's str.lower() restricted to the scripts that have case (Latin,
// Greek, Cyrillic, fullwidth forms, letterlike symbols); everything else,
// including CJK, is caseless and passes through.
const std::unordered_map<char32_t, char32_t> & lower_table() {
    static const std::unordered_map<char32_t, char32_t> table = {
        {0x00C0, 0x00E0}, {0x00C1, 0x00E1}, {0x00C2, 0x00E2}, {0x00C3, 0x00E3}, {0x00C4, 0x00E4}, {0x00C5, 0x00E5},
        {0x00C6, 0x00E6}, {0x00C7, 0x00E7}, {0x00C8, 0x00E8}, {0x00C9, 0x00E9}, {0x00CA, 0x00EA}, {0x00CB, 0x00EB},
        {0x00CC, 0x00EC}, {0x00CD, 0x00ED}, {0x00CE, 0x00EE}, {0x00CF, 0x00EF}, {0x00D0, 0x00F0}, {0x00D1, 0x00F1},
        {0x00D2, 0x00F2}, {0x00D3, 0x00F3}, {0x00D4, 0x00F4}, {0x00D5, 0x00F5}, {0x00D6, 0x00F6}, {0x00D8, 0x00F8},
        {0x00D9, 0x00F9}, {0x00DA, 0x00FA}, {0x00DB, 0x00FB}, {0x00DC, 0x00FC}, {0x00DD, 0x00FD}, {0x00DE, 0x00FE},
        {0x0100, 0x0101}, {0x0102, 0x0103}, {0x0104, 0x0105}, {0x0106, 0x0107}, {0x0108, 0x0109}, {0x010A, 0x010B},
        {0x010C, 0x010D}, {0x010E, 0x010F}, {0x0110, 0x0111}, {0x0112, 0x0113}, {0x0114, 0x0115}, {0x0116, 0x0117},
        {0x0118, 0x0119}, {0x011A, 0x011B}, {0x011C, 0x011D}, {0x011E, 0x011F}, {0x0120, 0x0121}, {0x0122, 0x0123},
        {0x0124, 0x0125}, {0x0126, 0x0127}, {0x0128, 0x0129}, {0x012A, 0x012B}, {0x012C, 0x012D}, {0x012E, 0x012F},
        {0x0132, 0x0133}, {0x0134, 0x0135}, {0x0136, 0x0137}, {0x0139, 0x013A}, {0x013B, 0x013C}, {0x013D, 0x013E},
        {0x013F, 0x0140}, {0x0141, 0x0142}, {0x0143, 0x0144}, {0x0145, 0x0146}, {0x0147, 0x0148}, {0x014A, 0x014B},
        {0x014C, 0x014D}, {0x014E, 0x014F}, {0x0150, 0x0151}, {0x0152, 0x0153}, {0x0154, 0x0155}, {0x0156, 0x0157},
        {0x0158, 0x0159}, {0x015A, 0x015B}, {0x015C, 0x015D}, {0x015E, 0x015F}, {0x0160, 0x0161}, {0x0162, 0x0163},
        {0x0164, 0x0165}, {0x0166, 0x0167}, {0x0168, 0x0169}, {0x016A, 0x016B}, {0x016C, 0x016D}, {0x016E, 0x016F},
        {0x0170, 0x0171}, {0x0172, 0x0173}, {0x0174, 0x0175}, {0x0176, 0x0177}, {0x0178, 0x00FF}, {0x0179, 0x017A},
        {0x017B, 0x017C}, {0x017D, 0x017E}, {0x0181, 0x0253}, {0x0182, 0x0183}, {0x0184, 0x0185}, {0x0186, 0x0254},
        {0x0187, 0x0188}, {0x0189, 0x0256}, {0x018A, 0x0257}, {0x018B, 0x018C}, {0x018E, 0x01DD}, {0x018F, 0x0259},
        {0x0190, 0x025B}, {0x0191, 0x0192}, {0x0193, 0x0260}, {0x0194, 0x0263}, {0x0196, 0x0269}, {0x0197, 0x0268},
        {0x0198, 0x0199}, {0x019C, 0x026F}, {0x019D, 0x0272}, {0x019F, 0x0275}, {0x01A0, 0x01A1}, {0x01A2, 0x01A3},
        {0x01A4, 0x01A5}, {0x01A6, 0x0280}, {0x01A7, 0x01A8}, {0x01A9, 0x0283}, {0x01AC, 0x01AD}, {0x01AE, 0x0288},
        {0x01AF, 0x01B0}, {0x01B1, 0x028A}, {0x01B2, 0x028B}, {0x01B3, 0x01B4}, {0x01B5, 0x01B6}, {0x01B7, 0x0292},
        {0x01B8, 0x01B9}, {0x01BC, 0x01BD}, {0x01C4, 0x01C6}, {0x01C5, 0x01C6}, {0x01C7, 0x01C9}, {0x01C8, 0x01C9},
        {0x01CA, 0x01CC}, {0x01CB, 0x01CC}, {0x01CD, 0x01CE}, {0x01CF, 0x01D0}, {0x01D1, 0x01D2}, {0x01D3, 0x01D4},
        {0x01D5, 0x01D6}, {0x01D7, 0x01D8}, {0x01D9, 0x01DA}, {0x01DB, 0x01DC}, {0x01DE, 0x01DF}, {0x01E0, 0x01E1},
        {0x01E2, 0x01E3}, {0x01E4, 0x01E5}, {0x01E6, 0x01E7}, {0x01E8, 0x01E9}, {0x01EA, 0x01EB}, {0x01EC, 0x01ED},
        {0x01EE, 0x01EF}, {0x01F1, 0x01F3}, {0x01F2, 0x01F3}, {0x01F4, 0x01F5}, {0x01F6, 0x0195}, {0x01F7, 0x01BF},
        {0x01F8, 0x01F9}, {0x01FA, 0x01FB}, {0x01FC, 0x01FD}, {0x01FE, 0x01FF}, {0x0200, 0x0201}, {0x0202, 0x0203},
        {0x0204, 0x0205}, {0x0206, 0x0207}, {0x0208, 0x0209}, {0x020A, 0x020B}, {0x020C, 0x020D}, {0x020E, 0x020F},
        {0x0210, 0x0211}, {0x0212, 0x0213}, {0x0214, 0x0215}, {0x0216, 0x0217}, {0x0218, 0x0219}, {0x021A, 0x021B},
        {0x021C, 0x021D}, {0x021E, 0x021F}, {0x0220, 0x019E}, {0x0222, 0x0223}, {0x0224, 0x0225}, {0x0226, 0x0227},
        {0x0228, 0x0229}, {0x022A, 0x022B}, {0x022C, 0x022D}, {0x022E, 0x022F}, {0x0230, 0x0231}, {0x0232, 0x0233},
        {0x023A, 0x2C65}, {0x023B, 0x023C}, {0x023D, 0x019A}, {0x023E, 0x2C66}, {0x0241, 0x0242}, {0x0243, 0x0180},
        {0x0244, 0x0289}, {0x0245, 0x028C}, {0x0246, 0x0247}, {0x0248, 0x0249}, {0x024A, 0x024B}, {0x024C, 0x024D},
        {0x024E, 0x024F}, {0x0386, 0x03AC}, {0x0388, 0x03AD}, {0x0389, 0x03AE}, {0x038A, 0x03AF}, {0x038C, 0x03CC},
        {0x038E, 0x03CD}, {0x038F, 0x03CE}, {0x0391, 0x03B1}, {0x0392, 0x03B2}, {0x0393, 0x03B3}, {0x0394, 0x03B4},
        {0x0395, 0x03B5}, {0x0396, 0x03B6}, {0x0397, 0x03B7}, {0x0398, 0x03B8}, {0x0399, 0x03B9}, {0x039A, 0x03BA},
        {0x039B, 0x03BB}, {0x039C, 0x03BC}, {0x039D, 0x03BD}, {0x039E, 0x03BE}, {0x039F, 0x03BF}, {0x03A0, 0x03C0},
        {0x03A1, 0x03C1}, {0x03A3, 0x03C3}, {0x03A4, 0x03C4}, {0x03A5, 0x03C5}, {0x03A6, 0x03C6}, {0x03A7, 0x03C7},
        {0x03A8, 0x03C8}, {0x03A9, 0x03C9}, {0x03AA, 0x03CA}, {0x03AB, 0x03CB}, {0x0400, 0x0450}, {0x0401, 0x0451},
        {0x0402, 0x0452}, {0x0403, 0x0453}, {0x0404, 0x0454}, {0x0405, 0x0455}, {0x0406, 0x0456}, {0x0407, 0x0457},
        {0x0408, 0x0458}, {0x0409, 0x0459}, {0x040A, 0x045A}, {0x040B, 0x045B}, {0x040C, 0x045C}, {0x040D, 0x045D},
        {0x040E, 0x045E}, {0x040F, 0x045F}, {0x0410, 0x0430}, {0x0411, 0x0431}, {0x0412, 0x0432}, {0x0413, 0x0433},
        {0x0414, 0x0434}, {0x0415, 0x0435}, {0x0416, 0x0436}, {0x0417, 0x0437}, {0x0418, 0x0438}, {0x0419, 0x0439},
        {0x041A, 0x043A}, {0x041B, 0x043B}, {0x041C, 0x043C}, {0x041D, 0x043D}, {0x041E, 0x043E}, {0x041F, 0x043F},
        {0x0420, 0x0440}, {0x0421, 0x0441}, {0x0422, 0x0442}, {0x0423, 0x0443}, {0x0424, 0x0444}, {0x0425, 0x0445},
        {0x0426, 0x0446}, {0x0427, 0x0447}, {0x0428, 0x0448}, {0x0429, 0x0449}, {0x042A, 0x044A}, {0x042B, 0x044B},
        {0x042C, 0x044C}, {0x042D, 0x044D}, {0x042E, 0x044E}, {0x042F, 0x044F}, {0x0460, 0x0461}, {0x0462, 0x0463},
        {0x0464, 0x0465}, {0x0466, 0x0467}, {0x0468, 0x0469}, {0x046A, 0x046B}, {0x046C, 0x046D}, {0x046E, 0x046F},
        {0x0470, 0x0471}, {0x0472, 0x0473}, {0x0474, 0x0475}, {0x0476, 0x0477}, {0x0478, 0x0479}, {0x047A, 0x047B},
        {0x047C, 0x047D}, {0x047E, 0x047F}, {0x0480, 0x0481}, {0x048A, 0x048B}, {0x048C, 0x048D}, {0x048E, 0x048F},
        {0x0490, 0x0491}, {0x0492, 0x0493}, {0x0494, 0x0495}, {0x0496, 0x0497}, {0x0498, 0x0499}, {0x049A, 0x049B},
        {0x049C, 0x049D}, {0x049E, 0x049F}, {0x04A0, 0x04A1}, {0x04A2, 0x04A3}, {0x04A4, 0x04A5}, {0x04A6, 0x04A7},
        {0x04A8, 0x04A9}, {0x04AA, 0x04AB}, {0x04AC, 0x04AD}, {0x04AE, 0x04AF}, {0x04B0, 0x04B1}, {0x04B2, 0x04B3},
        {0x04B4, 0x04B5}, {0x04B6, 0x04B7}, {0x04B8, 0x04B9}, {0x04BA, 0x04BB}, {0x04BC, 0x04BD}, {0x04BE, 0x04BF},
        {0x04C0, 0x04CF}, {0x04C1, 0x04C2}, {0x04C3, 0x04C4}, {0x04C5, 0x04C6}, {0x04C7, 0x04C8}, {0x04C9, 0x04CA},
        {0x04CB, 0x04CC}, {0x04CD, 0x04CE}, {0x04D0, 0x04D1}, {0x04D2, 0x04D3}, {0x04D4, 0x04D5}, {0x04D6, 0x04D7},
        {0x04D8, 0x04D9}, {0x04DA, 0x04DB}, {0x04DC, 0x04DD}, {0x04DE, 0x04DF}, {0x04E0, 0x04E1}, {0x04E2, 0x04E3},
        {0x04E4, 0x04E5}, {0x04E6, 0x04E7}, {0x04E8, 0x04E9}, {0x04EA, 0x04EB}, {0x04EC, 0x04ED}, {0x04EE, 0x04EF},
        {0x04F0, 0x04F1}, {0x04F2, 0x04F3}, {0x04F4, 0x04F5}, {0x04F6, 0x04F7}, {0x04F8, 0x04F9}, {0x04FA, 0x04FB},
        {0x04FC, 0x04FD}, {0x04FE, 0x04FF}, {0x0500, 0x0501}, {0x0502, 0x0503}, {0x0504, 0x0505}, {0x0506, 0x0507},
        {0x0508, 0x0509}, {0x050A, 0x050B}, {0x050C, 0x050D}, {0x050E, 0x050F}, {0x0510, 0x0511}, {0x0512, 0x0513},
        {0x0514, 0x0515}, {0x0516, 0x0517}, {0x0518, 0x0519}, {0x051A, 0x051B}, {0x051C, 0x051D}, {0x051E, 0x051F},
        {0x0520, 0x0521}, {0x0522, 0x0523}, {0x0524, 0x0525}, {0x0526, 0x0527}, {0x0528, 0x0529}, {0x052A, 0x052B},
        {0x052C, 0x052D}, {0x052E, 0x052F}, {0x1E00, 0x1E01}, {0x1E02, 0x1E03}, {0x1E04, 0x1E05}, {0x1E06, 0x1E07},
        {0x1E08, 0x1E09}, {0x1E0A, 0x1E0B}, {0x1E0C, 0x1E0D}, {0x1E0E, 0x1E0F}, {0x1E10, 0x1E11}, {0x1E12, 0x1E13},
        {0x1E14, 0x1E15}, {0x1E16, 0x1E17}, {0x1E18, 0x1E19}, {0x1E1A, 0x1E1B}, {0x1E1C, 0x1E1D}, {0x1E1E, 0x1E1F},
        {0x1E20, 0x1E21}, {0x1E22, 0x1E23}, {0x1E24, 0x1E25}, {0x1E26, 0x1E27}, {0x1E28, 0x1E29}, {0x1E2A, 0x1E2B},
        {0x1E2C, 0x1E2D}, {0x1E2E, 0x1E2F}, {0x1E30, 0x1E31}, {0x1E32, 0x1E33}, {0x1E34, 0x1E35}, {0x1E36, 0x1E37},
        {0x1E38, 0x1E39}, {0x1E3A, 0x1E3B}, {0x1E3C, 0x1E3D}, {0x1E3E, 0x1E3F}, {0x1E40, 0x1E41}, {0x1E42, 0x1E43},
        {0x1E44, 0x1E45}, {0x1E46, 0x1E47}, {0x1E48, 0x1E49}, {0x1E4A, 0x1E4B}, {0x1E4C, 0x1E4D}, {0x1E4E, 0x1E4F},
        {0x1E50, 0x1E51}, {0x1E52, 0x1E53}, {0x1E54, 0x1E55}, {0x1E56, 0x1E57}, {0x1E58, 0x1E59}, {0x1E5A, 0x1E5B},
        {0x1E5C, 0x1E5D}, {0x1E5E, 0x1E5F}, {0x1E60, 0x1E61}, {0x1E62, 0x1E63}, {0x1E64, 0x1E65}, {0x1E66, 0x1E67},
        {0x1E68, 0x1E69}, {0x1E6A, 0x1E6B}, {0x1E6C, 0x1E6D}, {0x1E6E, 0x1E6F}, {0x1E70, 0x1E71}, {0x1E72, 0x1E73},
        {0x1E74, 0x1E75}, {0x1E76, 0x1E77}, {0x1E78, 0x1E79}, {0x1E7A, 0x1E7B}, {0x1E7C, 0x1E7D}, {0x1E7E, 0x1E7F},
        {0x1E80, 0x1E81}, {0x1E82, 0x1E83}, {0x1E84, 0x1E85}, {0x1E86, 0x1E87}, {0x1E88, 0x1E89}, {0x1E8A, 0x1E8B},
        {0x1E8C, 0x1E8D}, {0x1E8E, 0x1E8F}, {0x1E90, 0x1E91}, {0x1E92, 0x1E93}, {0x1E94, 0x1E95}, {0x1E9E, 0x00DF},
        {0x1EA0, 0x1EA1}, {0x1EA2, 0x1EA3}, {0x1EA4, 0x1EA5}, {0x1EA6, 0x1EA7}, {0x1EA8, 0x1EA9}, {0x1EAA, 0x1EAB},
        {0x1EAC, 0x1EAD}, {0x1EAE, 0x1EAF}, {0x1EB0, 0x1EB1}, {0x1EB2, 0x1EB3}, {0x1EB4, 0x1EB5}, {0x1EB6, 0x1EB7},
        {0x1EB8, 0x1EB9}, {0x1EBA, 0x1EBB}, {0x1EBC, 0x1EBD}, {0x1EBE, 0x1EBF}, {0x1EC0, 0x1EC1}, {0x1EC2, 0x1EC3},
        {0x1EC4, 0x1EC5}, {0x1EC6, 0x1EC7}, {0x1EC8, 0x1EC9}, {0x1ECA, 0x1ECB}, {0x1ECC, 0x1ECD}, {0x1ECE, 0x1ECF},
        {0x1ED0, 0x1ED1}, {0x1ED2, 0x1ED3}, {0x1ED4, 0x1ED5}, {0x1ED6, 0x1ED7}, {0x1ED8, 0x1ED9}, {0x1EDA, 0x1EDB},
        {0x1EDC, 0x1EDD}, {0x1EDE, 0x1EDF}, {0x1EE0, 0x1EE1}, {0x1EE2, 0x1EE3}, {0x1EE4, 0x1EE5}, {0x1EE6, 0x1EE7},
        {0x1EE8, 0x1EE9}, {0x1EEA, 0x1EEB}, {0x1EEC, 0x1EED}, {0x1EEE, 0x1EEF}, {0x1EF0, 0x1EF1}, {0x1EF2, 0x1EF3},
        {0x1EF4, 0x1EF5}, {0x1EF6, 0x1EF7}, {0x1EF8, 0x1EF9}, {0x1EFA, 0x1EFB}, {0x1EFC, 0x1EFD}, {0x1EFE, 0x1EFF},
        {0x1F08, 0x1F00}, {0x1F09, 0x1F01}, {0x1F0A, 0x1F02}, {0x1F0B, 0x1F03}, {0x1F0C, 0x1F04}, {0x1F0D, 0x1F05},
        {0x1F0E, 0x1F06}, {0x1F0F, 0x1F07}, {0x1F18, 0x1F10}, {0x1F19, 0x1F11}, {0x1F1A, 0x1F12}, {0x1F1B, 0x1F13},
        {0x1F1C, 0x1F14}, {0x1F1D, 0x1F15}, {0x1F28, 0x1F20}, {0x1F29, 0x1F21}, {0x1F2A, 0x1F22}, {0x1F2B, 0x1F23},
        {0x1F2C, 0x1F24}, {0x1F2D, 0x1F25}, {0x1F2E, 0x1F26}, {0x1F2F, 0x1F27}, {0x1F38, 0x1F30}, {0x1F39, 0x1F31},
        {0x1F3A, 0x1F32}, {0x1F3B, 0x1F33}, {0x1F3C, 0x1F34}, {0x1F3D, 0x1F35}, {0x1F3E, 0x1F36}, {0x1F3F, 0x1F37},
        {0x1F48, 0x1F40}, {0x1F49, 0x1F41}, {0x1F4A, 0x1F42}, {0x1F4B, 0x1F43}, {0x1F4C, 0x1F44}, {0x1F4D, 0x1F45},
        {0x1F59, 0x1F51}, {0x1F5B, 0x1F53}, {0x1F5D, 0x1F55}, {0x1F5F, 0x1F57}, {0x1F68, 0x1F60}, {0x1F69, 0x1F61},
        {0x1F6A, 0x1F62}, {0x1F6B, 0x1F63}, {0x1F6C, 0x1F64}, {0x1F6D, 0x1F65}, {0x1F6E, 0x1F66}, {0x1F6F, 0x1F67},
        {0x1F88, 0x1F80}, {0x1F89, 0x1F81}, {0x1F8A, 0x1F82}, {0x1F8B, 0x1F83}, {0x1F8C, 0x1F84}, {0x1F8D, 0x1F85},
        {0x1F8E, 0x1F86}, {0x1F8F, 0x1F87}, {0x1F98, 0x1F90}, {0x1F99, 0x1F91}, {0x1F9A, 0x1F92}, {0x1F9B, 0x1F93},
        {0x1F9C, 0x1F94}, {0x1F9D, 0x1F95}, {0x1F9E, 0x1F96}, {0x1F9F, 0x1F97}, {0x1FA8, 0x1FA0}, {0x1FA9, 0x1FA1},
        {0x1FAA, 0x1FA2}, {0x1FAB, 0x1FA3}, {0x1FAC, 0x1FA4}, {0x1FAD, 0x1FA5}, {0x1FAE, 0x1FA6}, {0x1FAF, 0x1FA7},
        {0x1FB8, 0x1FB0}, {0x1FB9, 0x1FB1}, {0x1FBA, 0x1F70}, {0x1FBB, 0x1F71}, {0x1FBC, 0x1FB3}, {0x1FC8, 0x1F72},
        {0x1FC9, 0x1F73}, {0x1FCA, 0x1F74}, {0x1FCB, 0x1F75}, {0x1FCC, 0x1FC3}, {0x1FD8, 0x1FD0}, {0x1FD9, 0x1FD1},
        {0x1FDA, 0x1F76}, {0x1FDB, 0x1F77}, {0x1FE8, 0x1FE0}, {0x1FE9, 0x1FE1}, {0x1FEA, 0x1F7A}, {0x1FEB, 0x1F7B},
        {0x1FEC, 0x1FE5}, {0x1FF8, 0x1F78}, {0x1FF9, 0x1F79}, {0x1FFA, 0x1F7C}, {0x1FFB, 0x1F7D}, {0x1FFC, 0x1FF3},
        {0x2126, 0x03C9}, {0x212A, 0x006B}, {0x212B, 0x00E5}, {0x2132, 0x214E}, {0x2160, 0x2170}, {0x2161, 0x2171},
        {0x2162, 0x2172}, {0x2163, 0x2173}, {0x2164, 0x2174}, {0x2165, 0x2175}, {0x2166, 0x2176}, {0x2167, 0x2177},
        {0x2168, 0x2178}, {0x2169, 0x2179}, {0x216A, 0x217A}, {0x216B, 0x217B}, {0x216C, 0x217C}, {0x216D, 0x217D},
        {0x216E, 0x217E}, {0x216F, 0x217F}, {0x24B6, 0x24D0}, {0x24B7, 0x24D1}, {0x24B8, 0x24D2}, {0x24B9, 0x24D3},
        {0x24BA, 0x24D4}, {0x24BB, 0x24D5}, {0x24BC, 0x24D6}, {0x24BD, 0x24D7}, {0x24BE, 0x24D8}, {0x24BF, 0x24D9},
        {0x24C0, 0x24DA}, {0x24C1, 0x24DB}, {0x24C2, 0x24DC}, {0x24C3, 0x24DD}, {0x24C4, 0x24DE}, {0x24C5, 0x24DF},
        {0x24C6, 0x24E0}, {0x24C7, 0x24E1}, {0x24C8, 0x24E2}, {0x24C9, 0x24E3}, {0x24CA, 0x24E4}, {0x24CB, 0x24E5},
        {0x24CC, 0x24E6}, {0x24CD, 0x24E7}, {0x24CE, 0x24E8}, {0x24CF, 0x24E9}, {0xFF21, 0xFF41}, {0xFF22, 0xFF42},
        {0xFF23, 0xFF43}, {0xFF24, 0xFF44}, {0xFF25, 0xFF45}, {0xFF26, 0xFF46}, {0xFF27, 0xFF47}, {0xFF28, 0xFF48},
        {0xFF29, 0xFF49}, {0xFF2A, 0xFF4A}, {0xFF2B, 0xFF4B}, {0xFF2C, 0xFF4C}, {0xFF2D, 0xFF4D}, {0xFF2E, 0xFF4E},
        {0xFF2F, 0xFF4F}, {0xFF30, 0xFF50}, {0xFF31, 0xFF51}, {0xFF32, 0xFF52}, {0xFF33, 0xFF53}, {0xFF34, 0xFF54},
        {0xFF35, 0xFF55}, {0xFF36, 0xFF56}, {0xFF37, 0xFF57}, {0xFF38, 0xFF58}, {0xFF39, 0xFF59}, {0xFF3A, 0xFF5A},
    };
    return table;
}

char32_t lower32(char32_t c) {
    if (c >= 'A' && c <= 'Z') return static_cast<char32_t>(c + 32);
    const auto & table = lower_table();
    const auto it = table.find(c);
    return it == table.end() ? c : it->second;
}

// Lowercases a utf-8 string; this feeds the dictionary lookups, so the ascii
// path stays allocation-light.
std::string lower_utf8(const std::string & s) {
    bool ascii = true;
    for (char c : s) {
        if (static_cast<unsigned char>(c) >= 0x80) {
            ascii = false;
            break;
        }
    }
    std::string out;
    out.reserve(s.size());
    if (ascii) {
        for (char c : s) out.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c);
        return out;
    }
    for (std::size_t i = 0; i < s.size();) {
        const char32_t c = decode_utf8(s, i);
        if (c == 0x0130) {
            // 'i' + combining dot above; the only 1:2 mapping of Python's
            // str.lower() in the BMP.
            append_utf8(out, 0x0069);
            append_utf8(out, 0x0307);
            continue;
        }
        append_utf8(out, lower32(c));
    }
    return out;
}

// ---------------------------------------------------------------------------
// text helpers (g2p/converters/text.py)
// ---------------------------------------------------------------------------

// Splits on whitespace and CJK characters, retaining kana digraphs.
std::vector<std::pair<std::size_t, std::size_t>> word_spans(const std::u32string & text) {
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    std::size_t i = 0;
    while (i < text.size()) {
        if (is_space32(text[i])) {
            ++i;
            continue;
        }
        const std::size_t begin = i;
        if (is_kana(text[i]) && i + 1 < text.size() && is_small_kana(text[i + 1])) {
            i += 2;
        } else if (is_cjk(text[i])) {
            i += 1;
        } else {
            i += 1;
            while (i < text.size() && !is_space32(text[i]) && !is_cjk(text[i])) ++i;
        }
        spans.emplace_back(begin, i);
    }
    return spans;
}

std::vector<std::u32string> split_words(const std::u32string & text) {
    std::vector<std::u32string> words;
    for (const auto & span : word_spans(text)) {
        words.push_back(text.substr(span.first, span.second - span.first));
    }
    return words;
}

// Python's str.strip() for utf-8 text: leading and trailing whitespace is
// removed (the byte offsets are found in code point space, so multi-byte
// spaces such as U+3000 are handled too).
std::string strip_whitespace(const std::string & text) {
    std::vector<std::pair<std::size_t, char32_t>> chars;
    for (std::size_t i = 0; i < text.size();) {
        const std::size_t at = i;
        chars.emplace_back(at, decode_utf8(text, i));
    }
    std::size_t first = 0;
    std::size_t last  = chars.size();
    while (first < last && is_space32(chars[first].second)) ++first;
    while (last > first && is_space32(chars[last - 1].second)) --last;
    if (first >= last) return std::string();
    const std::size_t begin = chars[first].first;
    const std::size_t end   = last < chars.size() ? chars[last].first : text.size();
    return text.substr(begin, end - begin);
}

// Finds the first contiguous run of accepted characters.
template <typename Accept>
Match find_run(const std::u32string & text, Accept accepts) {
    for (std::size_t begin = 0; begin < text.size(); ++begin) {
        if (!accepts(text[begin])) continue;
        std::size_t end = begin + 1;
        while (end < text.size() && accepts(text[end])) ++end;
        return Match{ true, begin, end };
    }
    return Match{};
}

// ---------------------------------------------------------------------------
// pronunciation dictionaries (g2p/converters/dictionary.py)
// ---------------------------------------------------------------------------

// Insertion-ordered lookup table: duplicate keys accumulate pronunciations,
// and the key order is the order of first appearance in the file (the
// candidate order of the conversion output depends on it).
struct PronunciationDict {
    std::vector<std::string> keys;
    std::vector<std::vector<std::vector<std::string>>> entries;   // [key][variant][phoneme]
    std::unordered_map<std::string, std::size_t> index;

    const std::vector<std::vector<std::string>> * find(const std::string & key) const {
        const auto it = index.find(key);
        return it == index.end() ? nullptr : &entries[it->second];
    }
};

std::vector<std::string> split_ascii_whitespace(const std::string & s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && is_ascii_space(s[i])) ++i;
        const std::size_t begin = i;
        while (i < s.size() && !is_ascii_space(s[i])) ++i;
        if (i > begin) out.push_back(s.substr(begin, i - begin));
    }
    return out;
}

std::string read_text_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw InvalidArgument("cannot open G2P dictionary '" + path + "'");
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

// `<key>\t<ph1> <ph2> ...`; duplicate keys accumulate pronunciations, and a
// line without a non-empty phoneme field is ignored.
PronunciationDict load_pronunciation_dict(const std::string & path) {
    PronunciationDict dict;
    const std::string text = read_text_file(path);
    std::size_t pos = 0;
    while (pos <= text.size()) {
        std::size_t eol = text.find('\n', pos);
        if (eol == std::string::npos) eol = text.size();
        const std::string line = strip_whitespace(text.substr(pos, eol - pos));
        pos = eol + 1;
        if (line.empty()) continue;
        const std::size_t tab = line.find('\t');
        const std::string key = tab == std::string::npos ? line : line.substr(0, tab);
        const std::string raw = tab == std::string::npos ? std::string() : line.substr(tab + 1);
        if (raw.empty()) continue;
        std::vector<std::string> phonemes = split_ascii_whitespace(raw);
        const auto it = dict.index.find(key);
        if (it == dict.index.end()) {
            dict.index.emplace(key, dict.entries.size());
            dict.keys.push_back(key);
            dict.entries.push_back({ std::move(phonemes) });
        } else {
            dict.entries[it->second].push_back(std::move(phonemes));
        }
    }
    return dict;
}

// Removes a trailing `(N)` variant marker: regex `\s*\(\d+\)$`.
std::string strip_variant_marker(const std::string & key) {
    if (key.size() < 2 || key.back() != ')') return lower_utf8(key);
    std::size_t digits = key.size() - 1;                  // at ')'
    std::size_t begin  = digits;
    while (begin > 0 && key[begin - 1] >= '0' && key[begin - 1] <= '9') --begin;
    if (begin == digits || begin == 0 || key[begin - 1] != '(') return lower_utf8(key);
    std::size_t end = begin - 1;
    while (end > 0 && (key[end - 1] == ' ' || key[end - 1] == '\t')) --end;
    return lower_utf8(key.substr(0, end));
}

// ---------------------------------------------------------------------------
// converter base classes
// ---------------------------------------------------------------------------

// Pipeline-level preprocessor (`g2p/preprocessors/base.py`); a converter-local
// preprocessor chain uses Converter::preprocess instead.
class FragmentPreprocessor {
public:
    virtual ~FragmentPreprocessor() = default;
    virtual std::vector<std::string> process(const std::vector<std::string> & tokens) const = 0;
};

// The reference raises KeyError for a script token that the dictionary does
// not know; str(KeyError) is the repr of its argument, so the text carries an
// extra pair of quotes.  Kept identical here so dumps can be diffed textually.
std::string quoted_key_text(const std::string & message) {
    if (message.find('\'') != std::string::npos && message.find('"') == std::string::npos) {
        return '"' + message + '"';
    }
    return "'" + message + "'";
}

// g2p/converters/base.py:resolve_language -- "" means "any".
std::string resolve_language(const std::vector<std::string> & tags,
                             const std::vector<std::string> & languages) {
    if (tags.empty()) return std::string();
    if (languages.empty()) return tags[0];
    for (const std::string & tag : tags) {
        if (std::find(languages.begin(), languages.end(), tag) != languages.end()) return tag;
    }
    return std::string();
}

// `PronunciationScriptConverter`: text -> pronunciation script ->
// pronunciation paths.  Subclasses implement the two halves.
class ScriptDictionaryConverter : public Converter {
public:
    explicit ScriptDictionaryConverter(const std::string & dict_path)
        : script_dict_(load_pronunciation_dict(dict_path)) {}

    std::vector<Word> convert(const std::u32string & text) const override {
        const std::vector<std::u32string> words = split_words(text);
        const std::vector<std::vector<std::string>> scripts_per_token = text_to_scripts(words);
        if (scripts_per_token.size() != words.size()) {
            throw InvalidArgument("text_to_scripts must preserve word count.");
        }
        std::vector<Word> result;
        result.reserve(words.size());
        for (std::size_t i = 0; i < words.size(); ++i) {
            Word word;
            word.text = to_utf8(words[i]);
            std::unordered_set<std::string> seen_scripts;
            for (const std::string & script : scripts_per_token[i]) {
                if (!seen_scripts.insert(script).second) continue;
                Reading reading;
                std::unordered_set<std::string> seen_paths;
                for (Path & path : script_to_paths(script)) {
                    std::string key;
                    for (const Group & group : path) {
                        key += group.script;
                        key.push_back('\x1D');
                        for (const std::string & phoneme : group.phonemes) {
                            key += phoneme;
                            key.push_back('\x1F');
                        }
                        key.push_back('\x1E');
                    }
                    if (!seen_paths.insert(std::move(key)).second) continue;
                    reading.paths.push_back(std::move(path));
                }
                word.readings.push_back(std::move(reading));
            }
            result.push_back(std::move(word));
        }
        return result;
    }

protected:
    virtual std::vector<std::vector<std::string>> text_to_scripts(
        const std::vector<std::u32string> & words) const = 0;

    virtual std::vector<Path> script_to_paths(const std::string & script) const {
        const std::vector<std::vector<std::string>> * pronunciations = script_dict_.find(script);
        if (pronunciations == nullptr) {
            throw InvalidArgument(
                quoted_key_text("Script token '" + script + "' not found in script-to-phoneme dict."));
        }
        std::vector<Path> paths;
        paths.reserve(pronunciations->size());
        for (const std::vector<std::string> & phonemes : *pronunciations) {
            paths.push_back(phonemes.empty() ? Path{} : Path{ Group{ script, phonemes } });
        }
        return paths;
    }

    PronunciationDict script_dict_;
};

// ---------------------------------------------------------------------------
// converters
// ---------------------------------------------------------------------------

// g2p/converters/simple.py:PassthroughConverter
class PassthroughConverter : public Converter {
public:
    explicit PassthroughConverter(std::vector<std::string> languages)
        : languages_(std::move(languages)) {}

    const std::vector<std::string> & languages() const override { return languages_; }

    Match find(const std::u32string & text) const override {
        const auto spans = word_spans(text);
        if (spans.empty()) return Match{};
        return Match{ true, spans[0].first, spans[0].second };
    }

    std::vector<Word> convert(const std::u32string & text) const override {
        std::vector<Word> result;
        for (const std::u32string & token : split_words(text)) {
            const std::string word_text = to_utf8(token);
            Word word;
            word.text     = word_text;
            word.readings = { Reading{ { Path{ Group{ word_text, { word_text } } } } } };
            result.push_back(std::move(word));
        }
        return result;
    }

private:
    std::vector<std::string> languages_;                 // empty = Language.ANY
};

// g2p/converters/simple.py:CharPhonemeConverter
class CharactersConverter : public Converter {
public:
    CharactersConverter(std::vector<std::string> languages,
                        std::unordered_map<char32_t, std::vector<std::string>> mapping)
        : languages_(std::move(languages)), mapping_(std::move(mapping)) {}

    const std::vector<std::string> & languages() const override { return languages_; }

    Match find(const std::u32string & text) const override {
        for (const auto & span : word_spans(text)) {
            bool mapped = true;
            for (std::size_t i = span.first; i < span.second && mapped; ++i) {
                mapped = mapping_.find(text[i]) != mapping_.end();
            }
            if (mapped) return Match{ true, span.first, span.second };
        }
        return Match{};
    }

    std::vector<Word> convert(const std::u32string & text) const override {
        std::vector<Word> result;
        for (const std::u32string & token : split_words(text)) {
            std::vector<std::string> phonemes;
            for (char32_t c : token) {
                const auto it = mapping_.find(c);
                if (it == mapping_.end()) {
                    throw InvalidArgument("CharactersConverter: no mapping for '" + to_utf8(c) + "'.");
                }
                phonemes.insert(phonemes.end(), it->second.begin(), it->second.end());
            }
            Word word;
            word.text     = to_utf8(token);
            word.readings = { Reading{ { phonemes.empty()
                                             ? Path{}
                                             : Path{ Group{ word.text, phonemes } } } } };
            result.push_back(std::move(word));
        }
        return result;
    }

private:
    std::vector<std::string>                              languages_;
    std::unordered_map<char32_t, std::vector<std::string>> mapping_;
};

// g2p/converters/dictionary.py:DictionaryConverter
class DictionaryConverter : public Converter {
public:
    DictionaryConverter(std::vector<std::string> languages, const std::string & dict_path)
        : languages_(std::move(languages)) {
        const PronunciationDict raw = load_pronunciation_dict(dict_path);
        // `word(N)` / `word (N)` are variant spellings of the same key.
        for (std::size_t i = 0; i < raw.keys.size(); ++i) {
            const std::string key = strip_variant_marker(raw.keys[i]);
            const auto it = dict_.index.find(key);
            if (it == dict_.index.end()) {
                dict_.index.emplace(key, dict_.entries.size());
                dict_.keys.push_back(key);
                dict_.entries.push_back(raw.entries[i]);
            } else {
                std::vector<std::vector<std::string>> & target = dict_.entries[it->second];
                target.insert(target.end(), raw.entries[i].begin(), raw.entries[i].end());
            }
        }
    }

    const std::vector<std::string> & languages() const override { return languages_; }

    Match find(const std::u32string & text) const override {
        for (const auto & span : word_spans(text)) {
            const std::string word =
                lower_utf8(to_utf8(text.substr(span.first, span.second - span.first)));
            if (dict_.find(word) != nullptr) return Match{ true, span.first, span.second };
        }
        return Match{};
    }

    std::vector<Word> convert(const std::u32string & text) const override {
        std::vector<Word> result;
        for (const std::u32string & token : split_words(text)) {
            const std::string token_text = to_utf8(token);
            const std::vector<std::vector<std::string>> * pronunciations =
                dict_.find(lower_utf8(token_text));
            if (pronunciations == nullptr) {
                throw InvalidArgument("DictionaryConverter: token '" + token_text +
                                      "' not in dictionary. find should have filtered it.");
            }
            Reading reading;
            reading.paths.reserve(pronunciations->size());
            for (const std::vector<std::string> & phonemes : *pronunciations) {
                reading.paths.push_back(phonemes.empty() ? Path{}
                                                         : Path{ Group{ token_text, phonemes } });
            }
            Word word;
            word.text     = token_text;
            word.readings = { std::move(reading) };
            result.push_back(std::move(word));
        }
        return result;
    }

private:
    std::vector<std::string> languages_;
    PronunciationDict        dict_;
};

// g2p/converters/lstm.py: LSTMConverter — a LexiconConverter whose OOV words
// are inferred by a small char→phoneme LSTM (2-layer bi-LSTM encoder feeding a
// beam-searched attention decoder).  The reference reads a directory of ONNX
// files; this port consumes the single GGUF produced offline by
// scripts/convert_lstm_g2p_to_gguf.py and runs on ggml graphs through the
// GGML_OP_LSTM sweep op (lstm_g2p.h; CPU backend — the beam loop gives a GPU
// nothing to do).
//
// This is the only converter that needs ggml, so a build without the model
// (TIFA_G2P_NO_LSTM, used by the standalone tifa_ggml_g2p target) omits it and
// leaves English OOV words to the dictionary.
#ifndef TIFA_G2P_NO_LSTM
class LstmConverter : public Converter {
public:
    LstmConverter(std::vector<std::string> languages,
                  const std::string & dict_path,
                  const std::string & model_path,
                  int beam_size)
        : languages_(std::move(languages)), beam_size_(beam_size) {
        if (!dict_path.empty()) {
            const PronunciationDict raw = load_pronunciation_dict(dict_path);
            // `word(N)` / `word (N)` are variant spellings of the same key
            // (same merge rules as DictionaryConverter).
            for (std::size_t i = 0; i < raw.keys.size(); ++i) {
                const std::string key = strip_variant_marker(raw.keys[i]);
                const auto it = dict_.index.find(key);
                if (it == dict_.index.end()) {
                    dict_.index.emplace(key, dict_.entries.size());
                    dict_.keys.push_back(key);
                    dict_.entries.push_back(raw.entries[i]);
                } else {
                    std::vector<std::vector<std::string>> & target = dict_.entries[it->second];
                    target.insert(target.end(), raw.entries[i].begin(), raw.entries[i].end());
                }
            }
        }
        lstm_.reset(new LstmG2p(LstmG2p::from_file(resolve_gguf(model_path))));
    }

    const std::vector<std::string> & languages() const override { return languages_; }

    Match find(const std::u32string & text) const override {
        // Port of LSTMConverter.find: a word is claimed when it is in the
        // dictionary *or* every (lower-cased) character is inside the char
        // vocabulary, i.e. the LSTM can infer it.
        for (const auto & span : word_spans(text)) {
            const std::string word =
                lower_utf8(to_utf8(text.substr(span.first, span.second - span.first)));
            if (dict_.find(word) != nullptr) return Match{ true, span.first, span.second };
            if (lstm_->can_encode(word)) return Match{ true, span.first, span.second };
        }
        return Match{};
    }

    std::vector<Word> convert(const std::u32string & text) const override {
        std::vector<Word> result;
        for (const std::u32string & token : split_words(text)) {
            const std::string token_text = to_utf8(token);
            const std::vector<std::vector<std::string>> * pronunciations =
                dict_.find(lower_utf8(token_text));
            Reading reading;
            if (pronunciations != nullptr) {
                reading.paths.reserve(pronunciations->size());
                for (const std::vector<std::string> & phonemes : *pronunciations) {
                    reading.paths.push_back(phonemes.empty()
                                                ? Path{}
                                                : Path{ Group{ token_text, phonemes } });
                }
            } else {
                // Port of LexiconConverter.infer_oov → LSTMConverter._predict:
                // ranked, de-duplicated pronunciations, best first.
                for (const std::vector<std::string> & phonemes :
                     lstm_->predict(token_text, beam_size_)) {
                    reading.paths.push_back(phonemes.empty()
                                                ? Path{}
                                                : Path{ Group{ token_text, phonemes } });
                }
            }
            Word word;
            word.text     = token_text;
            word.readings = { std::move(reading) };
            result.push_back(std::move(word));
        }
        return result;
    }

private:
    // `model_path` in the reference config is the ONNX directory; the C++
    // build consumes a converted GGUF, looked up in the layouts that occur in
    // practice: an explicit .gguf path, the GGUF parked next to the directory
    // (models/assets/LstmG2p-Eng.gguf), or a rename inside it.
    static std::string resolve_gguf(const std::string & model_path) {
        auto exists = [](const std::string & path) {
            std::ifstream in(path, std::ios::binary);
            return static_cast<bool>(in);
        };
        if (model_path.size() >= 5 &&
            model_path.compare(model_path.size() - 5, 5, ".gguf") == 0) {
            return model_path;
        }
        const std::string aside = model_path + ".gguf";
        if (exists(aside)) return aside;
        std::string out = model_path;
        if (!out.empty() && out.back() != '/' && out.back() != '\\') out.push_back('/');
        return out + "lstm-g2p.gguf";
    }

    std::vector<std::string> languages_;
    PronunciationDict        dict_;
    std::unique_ptr<LstmG2p> lstm_;
    int                      beam_size_ = 0;   // 0 = the value recorded in the GGUF
};
#endif  // TIFA_G2P_NO_LSTM

// g2p/converters/chinese.py:_ChineseScriptConverter
class ChineseConverter : public ScriptDictionaryConverter {
public:
    ChineseConverter(std::vector<std::string> languages, const std::string & dict_path,
                     const std::string & engine_dict_dir)
        : ScriptDictionaryConverter(dict_path),
          languages_(std::move(languages)),
          engine_(engine_dict_dir) {}

    const std::vector<std::string> & languages() const override { return languages_; }

    Match find(const std::u32string & text) const override {
        return find_run(text, [](char32_t c) { return is_hanzi(c); });
    }

protected:
    std::vector<std::vector<std::string>> text_to_scripts(
        const std::vector<std::u32string> & words) const override
    {
        // The engine disambiguates phrases over the whole run, so the
        // characters are queried together and sliced back per word.
        std::vector<char32_t> chars;
        for (const std::u32string & word : words) {
            chars.insert(chars.end(), word.begin(), word.end());
        }
        chars = engine_.simplify(chars);
        const std::vector<std::vector<std::string>> best = engine_.query_raw(chars);

        std::vector<std::vector<std::string>> result;
        std::size_t index = 0;
        for (const std::u32string & word : words) {
            for (std::size_t k = 0; k < word.size(); ++k, ++index) {
                const std::vector<std::string> readings = engine_.readings(chars[index]);
                const std::string primary =
                    index < best.size() && !best[index].empty()
                        ? best[index][0]
                        : (readings.empty() ? to_utf8(chars[index]) : readings[0]);
                std::vector<std::string> scripts{ primary };
                for (const std::string & reading : readings) {
                    if (reading != primary) scripts.push_back(reading);
                }
                std::vector<std::string> supported;
                for (const std::string & script : scripts) {
                    if (script_dict_.find(script) != nullptr) supported.push_back(script);
                }
                // Preserve the lookup error when none of the readings is supported.
                result.push_back(supported.empty() ? scripts : supported);
            }
        }
        return result;
    }

private:
    std::vector<std::string> languages_;
    PinyinEngine             engine_;
};

// Katakana folds to hiragana for a unified lookup (both converters share it).
std::u32string kata_to_hira(const std::u32string & text) {
    constexpr char32_t k_katakana_start = 0x30A1;
    constexpr char32_t k_hiragana_start = 0x3041;
    constexpr char32_t k_span           = 0x5E;
    std::u32string result;
    result.reserve(text.size());
    for (char32_t c : text) {
        result.push_back(c >= k_katakana_start && c < k_katakana_start + k_span
                             ? static_cast<char32_t>(c - k_katakana_start + k_hiragana_start)
                             : c);
    }
    return result;
}

std::u32string utf8_to_u32(const std::string & text) {
    std::u32string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        out.push_back(decode_utf8(text, i));
    }
    return out;
}

// g2p/converters/japanese.py:JapaneseKanaConverter (gemination is off unless
// the config asks for `double_written_sokuon: true`)
class JapaneseKanaConverter : public ScriptDictionaryConverter {
public:
    // The mecab converter reuses the romaji table to re-join kana digraphs.
    friend class JapaneseMecabConverter;

    JapaneseKanaConverter(std::vector<std::string> languages, const std::string & dict_path,
                          bool double_written_sokuon)
        : ScriptDictionaryConverter(dict_path),
          languages_(std::move(languages)),
          double_written_sokuon_(double_written_sokuon) {}

    const std::vector<std::string> & languages() const override { return languages_; }

    Match find(const std::u32string & text) const override {
        return find_run(text, [](char32_t c) { return is_kana(c); });
    }

protected:
    std::vector<Path> script_to_paths(const std::string & script) const override {
        if (script.empty()) return { Path{} };
        if (script_dict_.find(script) != nullptr) {
            return ScriptDictionaryConverter::script_to_paths(script);
        }
        // Single consonants from gemination pass through directly.
        if (script.size() == 1 && is_consonant_leading(script[0])) {
            return { Path{ Group{ script, { script } } } };
        }
        throw InvalidArgument(
            quoted_key_text("Script token '" + script + "' not found in script-to-phoneme dict."));
    }

    std::vector<std::vector<std::string>> text_to_scripts(
        const std::vector<std::u32string> & words) const override
    {
        const auto & table = kana_to_romaji();
        std::vector<std::string> romaji_list;
        romaji_list.reserve(words.size());
        for (const std::u32string & word : words) {
            // Katakana is folded to hiragana for a unified lookup; the long
            // vowel mark U+30FC folds onto U+309C and is emptied below.
            const std::string token = to_utf8(kata_to_hira(word));
            if (token == "ー" || token == "゜") {
                romaji_list.push_back(std::string());
                continue;
            }
            const auto it = table.find(token);
            romaji_list.push_back(it == table.end() ? token : it->second);
        }
        if (double_written_sokuon_) romaji_list = apply_sokuon(romaji_list);

        std::vector<std::vector<std::string>> result;
        result.reserve(romaji_list.size());
        for (std::string & romaji : romaji_list) {
            result.push_back({ std::move(romaji) });
        }
        return result;
    }

private:
    // Hiragana-only romaji table, derived from cpp-kana's kanaToRomajiMap.
    static const std::unordered_map<std::string, std::string> & kana_to_romaji() {
        static const std::unordered_map<std::string, std::string> table = {
            { "っ", "cl" },
            { "あ", "a" }, { "い", "i" }, { "う", "u" }, { "え", "e" }, { "お", "o" },
            { "か", "ka" }, { "き", "ki" }, { "く", "ku" }, { "け", "ke" }, { "こ", "ko" },
            { "さ", "sa" }, { "し", "shi" }, { "す", "su" }, { "せ", "se" }, { "そ", "so" },
            { "た", "ta" }, { "ち", "chi" }, { "つ", "tsu" }, { "て", "te" }, { "と", "to" },
            { "な", "na" }, { "に", "ni" }, { "ぬ", "nu" }, { "ね", "ne" }, { "の", "no" },
            { "は", "ha" }, { "ひ", "hi" }, { "ふ", "fu" }, { "へ", "he" }, { "ほ", "ho" },
            { "ま", "ma" }, { "み", "mi" }, { "む", "mu" }, { "め", "me" }, { "も", "mo" },
            { "や", "ya" }, { "ゆ", "yu" }, { "よ", "yo" },
            { "ら", "ra" }, { "り", "ri" }, { "る", "ru" }, { "れ", "re" }, { "ろ", "ro" },
            { "わ", "wa" }, { "ゐ", "wi" }, { "ゑ", "we" },
            { "ん", "n" },
            { "が", "ga" }, { "ぎ", "gi" }, { "ぐ", "gu" }, { "げ", "ge" }, { "ご", "go" },
            { "ざ", "za" }, { "じ", "ji" }, { "ず", "zu" }, { "ぜ", "ze" }, { "ぞ", "zo" },
            { "だ", "da" }, { "ぢ", "ji" }, { "づ", "zu" }, { "で", "de" }, { "ど", "do" },
            { "ば", "ba" }, { "び", "bi" }, { "ぶ", "bu" }, { "べ", "be" }, { "ぼ", "bo" },
            { "ぱ", "pa" }, { "ぴ", "pi" }, { "ぷ", "pu" }, { "ぺ", "pe" }, { "ぽ", "po" },
            { "きゃ", "kya" }, { "きゅ", "kyu" }, { "きょ", "kyo" }, { "きぇ", "kye" },
            { "ぎゃ", "gya" }, { "ぎゅ", "gyu" }, { "ぎょ", "gyo" }, { "ぎぇ", "gye" },
            { "しゃ", "sha" }, { "しゅ", "shu" }, { "しょ", "sho" }, { "しぇ", "she" },
            { "じゃ", "ja" }, { "じゅ", "ju" }, { "じょ", "jo" }, { "じぇ", "je" },
            { "ちゃ", "cha" }, { "ちゅ", "chu" }, { "ちょ", "cho" }, { "ちぇ", "che" },
            { "にゃ", "nya" }, { "にゅ", "nyu" }, { "にょ", "nyo" }, { "にぇ", "nye" },
            { "ひゃ", "hya" }, { "ひゅ", "hyu" }, { "ひょ", "hyo" }, { "ひぇ", "hye" },
            { "びゃ", "bya" }, { "びゅ", "byu" }, { "びょ", "byo" }, { "びぇ", "bye" },
            { "ぴゃ", "pya" }, { "ぴゅ", "pyu" }, { "ぴょ", "pyo" }, { "ぴぇ", "pye" },
            { "みゃ", "mya" }, { "みゅ", "myu" }, { "みょ", "myo" }, { "みぇ", "mye" },
            { "りゃ", "rya" }, { "りゅ", "ryu" }, { "りょ", "ryo" }, { "りぇ", "rye" },
            { "いぇ", "ye" },
            { "うぁ", "wa" }, { "うぃ", "wi" }, { "うぇ", "we" }, { "うぉ", "wo" },
            { "くぁ", "kwa" }, { "くぃ", "kwi" }, { "くぇ", "kwe" }, { "くぉ", "kwo" },
            { "ぐぁ", "gwa" }, { "ぐぃ", "gwi" }, { "ぐぇ", "gwe" }, { "ぐぉ", "gwo" },
            { "すぁ", "swa" }, { "すぃ", "swi" }, { "すぇ", "swe" }, { "すぉ", "swo" },
            { "ずぁ", "zwa" }, { "ずぃ", "zwi" }, { "ずぇ", "zwe" }, { "ずぉ", "zwo" },
            { "つぁ", "tsa" }, { "つぃ", "tsi" }, { "つぇ", "tse" }, { "つぉ", "tso" },
            { "てぃ", "ti" }, { "てゅ", "tyu" },
            { "でぃ", "di" }, { "でゅ", "dyu" },
            { "とぅ", "tu" },
            { "どぅ", "du" },
            { "ふぁ", "fa" }, { "ふぃ", "fi" }, { "ふぇ", "fe" }, { "ふぉ", "fo" },
            { "ぶぁ", "bwa" }, { "ぶぃ", "bwi" }, { "ぶぇ", "bwe" }, { "ぶぉ", "bwo" },
            { "ぷぁ", "pwa" }, { "ぷぃ", "pwi" }, { "ぷぇ", "pwe" }, { "ぷぉ", "pwo" },
            { "ぬぁ", "nwa" }, { "ぬぃ", "nwi" }, { "ぬぇ", "nwe" }, { "ぬぉ", "nwo" },
            { "むぁ", "mwa" }, { "むぃ", "mwi" }, { "むぇ", "mwe" }, { "むぉ", "mwo" },
            { "るぁ", "rwa" }, { "るぃ", "rwi" }, { "るぇ", "rwe" }, { "るぉ", "rwo" },
            { "ゔ", "vu" },
            { "ゔぁ", "va" }, { "ゔぃ", "vi" }, { "ゔぇ", "ve" }, { "ゔぉ", "vo" },
            { "を", "o" },
        };
        return table;
    }

    // Gemination: 'cl' takes the leading consonant of the next non-empty token.
    static std::vector<std::string> apply_sokuon(const std::vector<std::string> & romaji_list) {
        std::vector<std::string> result;
        result.reserve(romaji_list.size());
        std::size_t i = 0;
        while (i < romaji_list.size()) {
            if (romaji_list[i] == "cl") {
                std::size_t j = i + 1;
                while (j < romaji_list.size() && romaji_list[j].empty()) ++j;
                if (j < romaji_list.size() && is_consonant_leading(romaji_list[j][0])) {
                    result.push_back(romaji_list[j].substr(0, 1));
                    ++i;
                    continue;
                }
            }
            result.push_back(romaji_list[i]);
            ++i;
        }
        return result;
    }

    std::vector<std::string> languages_;
    bool                     double_written_sokuon_;
};

// g2p/converters/japanese.py:JapaneseMecabConverter -- segment full Japanese
// word forms with MeCab and enumerate whole-word readings from UniDic's
// `pron` field (katakana).  Romaji, phoneme groups, dictionary alternatives
// and long vowels stay in JapaneseKanaConverter, exactly as upstream: MeCab
// supplies only the kana reading of each surface form.
#ifndef TIFA_G2P_NO_MECAB
class JapaneseMecabConverter : public Converter {
public:
    JapaneseMecabConverter(std::vector<std::string> languages,
                           std::unique_ptr<MecabTagger> tagger,
                           std::unique_ptr<JapaneseKanaConverter> kana,
                           bool double_written_sokuon, int nbest)
        : languages_(std::move(languages)),
          tagger_(std::move(tagger)),
          kana_(std::move(kana)),
          double_written_sokuon_(double_written_sokuon),
          nbest_(nbest) {}

    const std::vector<std::string> & languages() const override { return languages_; }

    Match find(const std::u32string & text) const override {
        return find_run(text, is_japanese_input);
    }

    std::vector<Word> convert(const std::u32string & text) const override {
        // Snapshot the surfaces before the N-best calls below (MeCab's lattice
        // is reused), re-joining kana digraphs MeCab split off: a surface that
        // starts with a small kana merges back when the two together form a
        // kana unit with a romaji spelling (upstream does the same).
        std::vector<std::u32string> surfaces;
        for (const MecabTagger::Morph & morph : tagger_->segment(to_utf8(text))) {
            const std::u32string surface = utf8_to_u32(morph.surface);
            if (!surfaces.empty() && !surface.empty() && is_small_kana(surface[0])
                && all_kana(surfaces.back() + surface)
                && JapaneseKanaConverter::kana_to_romaji().count(to_utf8(kata_to_hira(
                       std::u32string(1, surfaces.back().back()) + surface.substr(0, 1))))
                       > 0) {
                surfaces.back() += surface;
            } else {
                surfaces.push_back(surface);
            }
        }

        // Per-surface pronunciations.  A kanji surface without any UniDic
        // reading is a hard error upstream (G2PConversionError); kana falls
        // through to the per-unit dictionary path.
        struct Item {
            std::u32string           text;
            std::vector<std::string> pronunciations;
        };
        std::vector<Item> items;
        for (const std::u32string & surface : surfaces) {
            const std::string surface_utf8 = to_utf8(surface);
            std::vector<std::string> prons =
                tagger_->pronunciations(surface_utf8, nbest_);
            if (prons.empty() && !all_kana(surface)) {
                throw InvalidArgument(
                    "cannot read the Japanese word '" + surface_utf8 +
                    "': MeCab/UniDic has no pronunciation for it");
            }
            if (!items.empty() && double_written_sokuon_) {
                const std::vector<std::string> previous =
                    items.back().pronunciations.empty()
                        ? std::vector<std::string>{ to_utf8(items.back().text) }
                        : items.back().pronunciations;
                const bool previous_is_sokuon =
                    std::any_of(previous.begin(), previous.end(),
                                [](const std::string & candidate) {
                                    return ends_with_sokuon(candidate);
                                });
                if (previous_is_sokuon) {
                    // Keep a combined sokuon dictionary key within one path
                    // choice: the previous item absorbs this surface, its
                    // reading candidates becoming every left+right pair
                    // (upstream's product).
                    const std::vector<std::string> right =
                        prons.empty() ? std::vector<std::string>{ surface_utf8 } : prons;
                    std::vector<std::string>        merged;
                    std::unordered_set<std::string> seen;
                    for (const std::string & l : previous) {
                        for (const std::string & r : right) {
                            if (seen.insert(l + r).second) merged.push_back(l + r);
                        }
                    }
                    items.back().text += surface;
                    items.back().pronunciations = std::move(merged);
                    continue;
                }
            }
            items.push_back(Item{ surface, std::move(prons) });
        }

        std::vector<Word> result;
        for (const Item & item : items) {
            if (!item.pronunciations.empty()) {
                Word word;
                word.text = to_utf8(item.text);
                for (const std::string & pron : item.pronunciations) {
                    word.readings.push_back(reading_from_kana(pron));
                }
                result.push_back(std::move(word));
            } else {
                // Pure kana without a UniDic pron: the kana converter's own
                // per-unit dictionary alternatives (upstream `_kana._convert`).
                std::vector<Word> words = kana_->convert(item.text);
                result.insert(result.end(), std::make_move_iterator(words.begin()),
                              std::make_move_iterator(words.end()));
            }
        }
        return result;
    }

private:
    static bool all_kana(const std::u32string & text) {
        if (text.empty()) return false;
        for (char32_t c : text) {
            if (!is_kana(c)) return false;
        }
        return true;
    }

    // True when the kana reading ends in っ -- the sokuon that
    // double_written_sokuon merges into the following syllable.  Trailing
    // long-vowel marks and handakuten are stripped first (upstream rstrip).
    static bool ends_with_sokuon(const std::string & kana_utf8) {
        std::u32string hira = kata_to_hira(utf8_to_u32(kana_utf8));
        while (!hira.empty() && (hira.back() == 0x30FC || hira.back() == 0x309C)) {
            hira.pop_back();
        }
        return !hira.empty() && hira.back() == 0x3063;  // っ
    }

    // One Reading per pronunciation; per-kana dictionary alternatives combine
    // within this reading, not across readings (upstream `_reading`): the
    // cross product of every kana unit's alternatives forms the path set.
    Reading reading_from_kana(const std::string & pron) const {
        Reading out;
        const std::vector<Word> words = kana_->convert(utf8_to_u32(pron));
        std::vector<std::vector<Path>> alternatives;
        alternatives.reserve(words.size());
        for (const Word & word : words) {
            std::vector<Path> options;
            for (const Reading & reading : word.readings) {
                for (const Path & path : reading.paths) options.push_back(path);
            }
            if (options.empty()) return out;  // product over an empty set
            alternatives.push_back(std::move(options));
        }
        std::unordered_set<std::string> seen;
        std::vector<std::size_t>        index(alternatives.size(), 0);
        while (true) {
            Path        path;
            std::string key;
            for (std::size_t i = 0; i < alternatives.size(); ++i) {
                for (const Group & group : alternatives[i][index[i]]) {
                    path.push_back(group);
                    key += group.script;
                    key.push_back('\x1D');
                    for (const std::string & phoneme : group.phonemes) {
                        key += phoneme;
                        key.push_back('\x1F');
                    }
                    key.push_back('\x1E');
                }
            }
            if (seen.insert(std::move(key)).second) out.paths.push_back(std::move(path));
            std::size_t pos = index.size();
            bool        carried = false;
            while (pos > 0) {
                --pos;
                if (++index[pos] < alternatives[pos].size()) {
                    carried = true;
                    break;
                }
                index[pos] = 0;
            }
            if (!carried) break;
        }
        return out;
    }

    std::vector<std::string>               languages_;
    std::unique_ptr<MecabTagger>           tagger_;
    std::unique_ptr<JapaneseKanaConverter> kana_;
    bool                                   double_written_sokuon_;
    int                                    nbest_;
};
#endif  // TIFA_G2P_NO_MECAB

// ---------------------------------------------------------------------------
// preprocessors (g2p/preprocessors/simple.py)
// ---------------------------------------------------------------------------

// Splits tokens on punctuation and discards the punctuation characters.
// Apostrophes and hyphens are kept, as they are common in phonetic
// transcriptions.
class FilterPunctuation : public FragmentPreprocessor {
public:
    std::vector<std::string> process(const std::vector<std::string> & tokens) const override {
        const std::unordered_set<char32_t> & punctuation = punctuation_set();
        std::vector<std::string> result;
        for (const std::string & token : tokens) {
            std::string part;
            for (std::size_t i = 0; i < token.size();) {
                const char32_t c = decode_utf8(token, i);
                if (punctuation.count(c) != 0) {
                    if (!part.empty()) {
                        result.push_back(part);
                        part.clear();
                    }
                } else {
                    append_utf8(part, c);
                }
            }
            if (!part.empty()) result.push_back(part);
        }
        return result;
    }

private:
    static const std::unordered_set<char32_t> & punctuation_set() {
        static const std::unordered_set<char32_t> set = [] {
            std::unordered_set<char32_t> out;
            // string.punctuation, minus "'" and "-"
            const char * ascii = "!\"#$%&()*+,./:;<=>?@[\\]^_`{|}~";
            for (const char * p = ascii; *p != '\0'; ++p) {
                out.insert(static_cast<char32_t>(*p));
            }
            for (char32_t c : to_u32("，。；：“”‘’（）【】《》…—～、·！？")) {
                out.insert(c);
            }
            return out;
        }();
        return set;
    }
};

class Lowercase : public FragmentPreprocessor {
public:
    std::vector<std::string> process(const std::vector<std::string> & tokens) const override {
        std::vector<std::string> result;
        result.reserve(tokens.size());
        for (const std::string & token : tokens) result.push_back(lower_utf8(token));
        return result;
    }
};

class StripWhitespace : public FragmentPreprocessor {
public:
    std::vector<std::string> process(const std::vector<std::string> & tokens) const override {
        std::vector<std::string> result;
        result.reserve(tokens.size());
        for (const std::string & token : tokens) {
            std::string stripped = strip_whitespace(token);
            if (!stripped.empty()) result.push_back(std::move(stripped));
        }
        return result;
    }
};

// Strips combining marks the way `unicodedata.normalize("NFKD", t)` followed
// by a combining-mark filter does.  Without ICU there is no full Unicode
// decomposition, so this covers the precomposed Latin-1 / Latin Extended-A
// letters (cafe, resume, naive, ...), the kana voiced signs, the fullwidth
// forms and the ideographic space.  Characters outside those families keep
// their compatibility decomposition (Hangul syllables, halfwidth katakana,
// CJK compatibility ideographs, ...).
class RemoveAccents : public FragmentPreprocessor {
public:
    std::vector<std::string> process(const std::vector<std::string> & tokens) const override {
        const std::unordered_map<char32_t, char32_t> & table = decomposition_table();
        std::vector<std::string> result;
        result.reserve(tokens.size());
        for (const std::string & token : tokens) {
            std::string out;
            out.reserve(token.size());
            for (std::size_t i = 0; i < token.size();) {
                const char32_t c = decode_utf8(token, i);
                // NFKD folds the fullwidth forms onto ascii.
                if (c >= 0xFF01 && c <= 0xFF5E) {
                    append_utf8(out, static_cast<char32_t>(c - 0xFEE0));
                    continue;
                }
                if (c == 0x3000) {
                    append_utf8(out, 0x0020);
                    continue;
                }
                const auto it = table.find(c);
                append_utf8(out, it == table.end() ? c : it->second);
            }
            result.push_back(std::move(out));
        }
        return result;
    }

private:
    static const std::unordered_map<char32_t, char32_t> & decomposition_table() {
        static const std::unordered_map<char32_t, char32_t> table = {
            {0x00C0, 0x0041}, {0x00C1, 0x0041}, {0x00C2, 0x0041}, {0x00C3, 0x0041}, {0x00C4, 0x0041}, {0x00C5, 0x0041},
            {0x00C7, 0x0043}, {0x00C8, 0x0045}, {0x00C9, 0x0045}, {0x00CA, 0x0045}, {0x00CB, 0x0045}, {0x00CC, 0x0049},
            {0x00CD, 0x0049}, {0x00CE, 0x0049}, {0x00CF, 0x0049}, {0x00D1, 0x004E}, {0x00D2, 0x004F}, {0x00D3, 0x004F},
            {0x00D4, 0x004F}, {0x00D5, 0x004F}, {0x00D6, 0x004F}, {0x00D9, 0x0055}, {0x00DA, 0x0055}, {0x00DB, 0x0055},
            {0x00DC, 0x0055}, {0x00DD, 0x0059}, {0x00E0, 0x0061}, {0x00E1, 0x0061}, {0x00E2, 0x0061}, {0x00E3, 0x0061},
            {0x00E4, 0x0061}, {0x00E5, 0x0061}, {0x00E7, 0x0063}, {0x00E8, 0x0065}, {0x00E9, 0x0065}, {0x00EA, 0x0065},
            {0x00EB, 0x0065}, {0x00EC, 0x0069}, {0x00ED, 0x0069}, {0x00EE, 0x0069}, {0x00EF, 0x0069}, {0x00F1, 0x006E},
            {0x00F2, 0x006F}, {0x00F3, 0x006F}, {0x00F4, 0x006F}, {0x00F5, 0x006F}, {0x00F6, 0x006F}, {0x00F9, 0x0075},
            {0x00FA, 0x0075}, {0x00FB, 0x0075}, {0x00FC, 0x0075}, {0x00FD, 0x0079}, {0x00FF, 0x0079}, {0x0100, 0x0041},
            {0x0101, 0x0061}, {0x0102, 0x0041}, {0x0103, 0x0061}, {0x0104, 0x0041}, {0x0105, 0x0061}, {0x0106, 0x0043},
            {0x0107, 0x0063}, {0x0108, 0x0043}, {0x0109, 0x0063}, {0x010A, 0x0043}, {0x010B, 0x0063}, {0x010C, 0x0043},
            {0x010D, 0x0063}, {0x010E, 0x0044}, {0x010F, 0x0064}, {0x0112, 0x0045}, {0x0113, 0x0065}, {0x0114, 0x0045},
            {0x0115, 0x0065}, {0x0116, 0x0045}, {0x0117, 0x0065}, {0x0118, 0x0045}, {0x0119, 0x0065}, {0x011A, 0x0045},
            {0x011B, 0x0065}, {0x011C, 0x0047}, {0x011D, 0x0067}, {0x011E, 0x0047}, {0x011F, 0x0067}, {0x0120, 0x0047},
            {0x0121, 0x0067}, {0x0122, 0x0047}, {0x0123, 0x0067}, {0x0124, 0x0048}, {0x0125, 0x0068}, {0x0128, 0x0049},
            {0x0129, 0x0069}, {0x012A, 0x0049}, {0x012B, 0x0069}, {0x012C, 0x0049}, {0x012D, 0x0069}, {0x012E, 0x0049},
            {0x012F, 0x0069}, {0x0130, 0x0049}, {0x0134, 0x004A}, {0x0135, 0x006A}, {0x0136, 0x004B}, {0x0137, 0x006B},
            {0x0139, 0x004C}, {0x013A, 0x006C}, {0x013B, 0x004C}, {0x013C, 0x006C}, {0x013D, 0x004C}, {0x013E, 0x006C},
            {0x0143, 0x004E}, {0x0144, 0x006E}, {0x0145, 0x004E}, {0x0146, 0x006E}, {0x0147, 0x004E}, {0x0148, 0x006E},
            {0x014C, 0x004F}, {0x014D, 0x006F}, {0x014E, 0x004F}, {0x014F, 0x006F}, {0x0150, 0x004F}, {0x0151, 0x006F},
            {0x0154, 0x0052}, {0x0155, 0x0072}, {0x0156, 0x0052}, {0x0157, 0x0072}, {0x0158, 0x0052}, {0x0159, 0x0072},
            {0x015A, 0x0053}, {0x015B, 0x0073}, {0x015C, 0x0053}, {0x015D, 0x0073}, {0x015E, 0x0053}, {0x015F, 0x0073},
            {0x0160, 0x0053}, {0x0161, 0x0073}, {0x0162, 0x0054}, {0x0163, 0x0074}, {0x0164, 0x0054}, {0x0165, 0x0074},
            {0x0168, 0x0055}, {0x0169, 0x0075}, {0x016A, 0x0055}, {0x016B, 0x0075}, {0x016C, 0x0055}, {0x016D, 0x0075},
            {0x016E, 0x0055}, {0x016F, 0x0075}, {0x0170, 0x0055}, {0x0171, 0x0075}, {0x0172, 0x0055}, {0x0173, 0x0075},
            {0x0174, 0x0057}, {0x0175, 0x0077}, {0x0176, 0x0059}, {0x0177, 0x0079}, {0x0178, 0x0059}, {0x0179, 0x005A},
            {0x017A, 0x007A}, {0x017B, 0x005A}, {0x017C, 0x007A}, {0x017D, 0x005A}, {0x017E, 0x007A}, {0x017F, 0x0073},
            // kana: the voiced/semi-voiced signs are combining marks after
            // NFKD, so they are stripped as well (matching the reference).
            {0x304C, 0x304B}, {0x304E, 0x304D}, {0x3050, 0x304F}, {0x3052, 0x3051}, {0x3054, 0x3053}, {0x3056, 0x3055},
            {0x3058, 0x3057}, {0x305A, 0x3059}, {0x305C, 0x305B}, {0x305E, 0x305D}, {0x3060, 0x305F}, {0x3062, 0x3061},
            {0x3065, 0x3064}, {0x3067, 0x3066}, {0x3069, 0x3068}, {0x3070, 0x306F}, {0x3071, 0x306F}, {0x3073, 0x3072},
            {0x3074, 0x3072}, {0x3076, 0x3075}, {0x3077, 0x3075}, {0x3079, 0x3078}, {0x307A, 0x3078}, {0x307C, 0x307B},
            {0x307D, 0x307B}, {0x3094, 0x3046}, {0x309B, 0x0020}, {0x309C, 0x0020}, {0x309E, 0x309D}, {0x30AC, 0x30AB},
            {0x30AE, 0x30AD}, {0x30B0, 0x30AF}, {0x30B2, 0x30B1}, {0x30B4, 0x30B3}, {0x30B6, 0x30B5}, {0x30B8, 0x30B7},
            {0x30BA, 0x30B9}, {0x30BC, 0x30BB}, {0x30BE, 0x30BD}, {0x30C0, 0x30BF}, {0x30C2, 0x30C1}, {0x30C5, 0x30C4},
            {0x30C7, 0x30C6}, {0x30C9, 0x30C8}, {0x30D0, 0x30CF}, {0x30D1, 0x30CF}, {0x30D3, 0x30D2}, {0x30D4, 0x30D2},
            {0x30D6, 0x30D5}, {0x30D7, 0x30D5}, {0x30D9, 0x30D8}, {0x30DA, 0x30D8}, {0x30DC, 0x30DB}, {0x30DD, 0x30DB},
            {0x30F4, 0x30A6}, {0x30F7, 0x30EF}, {0x30F8, 0x30F0}, {0x30F9, 0x30F1}, {0x30FA, 0x30F2}, {0x30FE, 0x30FD},
        };
        return table;
    }
};

// ---------------------------------------------------------------------------
// config helpers
// ---------------------------------------------------------------------------

void warn(const std::string & message) {
    std::fprintf(stderr, "[g2p] %s\n", message.c_str());
}

std::string join_path(const std::string & dir, const std::string & rel) {
    if (dir.empty()) return rel;
    std::string out = dir;
    if (out.back() != '/' && out.back() != '\\') out.push_back('/');
    return out + rel;
}

bool file_exists(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

// "@dictionaries/x" resolves against the directory holding the dictionaries
// (the model directory); any other path is taken as given.
std::string resolve_path_ref(const std::string & path, const std::string & dict_dir) {
    if (!path.empty() && path[0] == '@') return join_path(dict_dir, path.substr(1));
    return path;
}

const json::Value * json_object(const json::Value & parent, const char * key) {
    if (!parent.is_object()) return nullptr;
    const json::Value * value = parent.find(key);
    return value != nullptr && value->is_object() ? value : nullptr;
}

std::string json_string(const json::Value & parent, const char * key,
                        const std::string & fallback = std::string()) {
    if (!parent.is_object()) return fallback;
    const json::Value * value = parent.find(key);
    return value != nullptr && value->is_string() ? value->str : fallback;
}

std::string converter_id(const json::Value & config) {
    return json_string(config, "id");
}

// `zh,zho,cmn` -> {"zh", "zho", "cmn"} (g2p/registry.py:parse_language)
std::vector<std::string> parse_language(const std::string & language) {
    std::vector<std::string> tags;
    std::size_t begin = 0;
    while (true) {
        const std::size_t comma = language.find(',', begin);
        std::string tag = comma == std::string::npos ? language.substr(begin)
                                                     : language.substr(begin, comma - begin);
        const std::size_t first = tag.find_first_not_of(" \t\r\n");
        const std::size_t last  = tag.find_last_not_of(" \t\r\n");
        tags.push_back(first == std::string::npos ? std::string()
                                                  : tag.substr(first, last - first + 1));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return tags;
}

// The cpp-pinyin engine dictionaries (word.txt, phrases_dict.txt, ...) are
// data files that live next to the model's `dictionaries` folder rather than
// inside it.  They are looked up in a few conventional locations and can
// always be pinned with the `engine_dict_path` converter kwarg.
std::string find_engine_dict_dir(const std::string & language,
                                 const std::string & dict_dir,
                                 const std::string & explicit_path) {
    std::vector<std::string> candidates;
    if (!explicit_path.empty()) candidates.push_back(explicit_path);
    if (!dict_dir.empty()) {
        candidates.push_back(join_path(dict_dir, "cpp_pinyin/" + language));
        candidates.push_back(join_path(dict_dir, "cpp_pinyin/dicts/" + language));
        candidates.push_back(join_path(dict_dir, language));
    }
    for (const std::string & candidate : candidates) {
        if (file_exists(join_path(candidate, "word.txt"))) return candidate;
    }
    return candidates.empty() ? std::string() : candidates.front();
}

// Same conventional-location probing as above, for the MeCab/UniDic dicdir of
// the `japanese-mecab` converter: a candidate directory must hold sys.dic.
// The release ships the dictionary as a separate asset to extract into
// <model dir>/unidic.
std::string find_unidic_dir(const std::string & dict_dir,
                            const std::string & explicit_path) {
    std::vector<std::string> candidates;
    if (!explicit_path.empty()) candidates.push_back(explicit_path);
    if (!dict_dir.empty()) {
        candidates.push_back(join_path(dict_dir, "unidic"));
        candidates.push_back(join_path(dict_dir, "mecab/unidic"));
        candidates.push_back(join_path(dict_dir, "dictionaries/unidic"));
    }
    for (const std::string & candidate : candidates) {
        if (file_exists(join_path(candidate, "sys.dic"))) return candidate;
    }
    return std::string();
}

}  // namespace

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

struct Pipeline::Impl {
    std::vector<std::unique_ptr<FragmentPreprocessor>> preprocessors;
    std::vector<std::unique_ptr<Converter>>            converters;
};

Pipeline::Pipeline() : impl_(new Impl()) {}
Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline &&) noexcept = default;
Pipeline & Pipeline::operator=(Pipeline &&) noexcept = default;

Pipeline Pipeline::from_config(const std::string & g2p_json, const std::string & dict_dir) {
    Pipeline pipeline;
    Impl & impl = *pipeline.impl_;

    // An empty config (a model without `inference.g2p`) builds a pipeline with
    // no preprocessors and no converters.
    const bool empty_config = g2p_json.find_first_not_of(" \t\r\n") == std::string::npos;
    const json::Value doc = json::parse(empty_config ? std::string("{}") : g2p_json);
    if (!doc.is_object()) {
        throw InvalidArgument("g2p config must be a JSON object");
    }

    if (const json::Value * preprocessors = doc.find("preprocessors");
        preprocessors != nullptr && preprocessors->is_array()) {
        for (const json::Value & config : preprocessors->arr) {
            const std::string id = converter_id(config);
            if (id == "filter-punctuation") {
                impl.preprocessors.push_back(std::make_unique<FilterPunctuation>());
            } else if (id == "lowercase") {
                impl.preprocessors.push_back(std::make_unique<Lowercase>());
            } else if (id == "strip-whitespace") {
                impl.preprocessors.push_back(std::make_unique<StripWhitespace>());
            } else if (id == "remove-accents") {
                impl.preprocessors.push_back(std::make_unique<RemoveAccents>());
            } else {
                warn("unknown preprocessor '" + id + "', skipping");
            }
        }
    }

    if (const json::Value * converters = doc.find("converters");
        converters != nullptr && converters->is_array()) {
        for (const json::Value & config : converters->arr) {
            const std::string id = converter_id(config);
            const json::Value * kwargs = json_object(config, "kwargs");

            const auto kwarg = [&](const char * key) {
                return kwargs == nullptr ? std::string() : json_string(*kwargs, key);
            };
            const auto kwarg_path = [&](const char * key) {
                return resolve_path_ref(kwarg(key), dict_dir);
            };
            // The stock model's GGUF embeds the kwargs of tifa.vocab.json, whose
            // values are strings ('nbest': '32', 'double_written_sokuon':
            // 'False'); accept native JSON scalars as well.
            const auto kwarg_int = [&](const char * key, int fallback) {
                if (kwargs != nullptr) {
                    if (const json::Value * value = kwargs->find(key)) {
                        if (value->is_number()) return static_cast<int>(value->number);
                        if (value->is_string()) {
                            const std::string & text = value->str;
                            std::size_t         at   = 0;
                            if (at < text.size() && (text[at] == '-' || text[at] == '+')) ++at;
                            if (at < text.size() && text.find_first_not_of("0123456789", at)
                                                       == std::string::npos) {
                                return std::atoi(text.c_str());
                            }
                        }
                    }
                }
                return fallback;
            };
            const auto kwarg_bool = [&](const char * key, bool fallback) {
                if (kwargs != nullptr) {
                    if (const json::Value * value = kwargs->find(key)) {
                        if (value->is_bool()) return value->boolean;
                        if (value->is_string()) {
                            std::string text;
                            for (char c : value->str) {
                                text.push_back(
                                    static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
                            }
                            if (text == "true" || text == "1" || text == "yes") return true;
                            if (text == "false" || text == "0" || text == "no") return false;
                        }
                    }
                }
                return fallback;
            };
            const auto languages_of = [&](std::vector<std::string> defaults) {
                const json::Value * language = config.is_object() ? config.find("language") : nullptr;
                if (language != nullptr && language->is_string()) {
                    return parse_language(language->str);
                }
                return defaults;
            };

            // Supported ids mirror the reference registry; anything else is
            // skipped so that configs carrying MeCab or the English LSTM still
            // load in this build.
            if (id == "chinese-pinyin" || id == "yue-jyutping") {
                const bool cantonese = id == "yue-jyutping";
                const std::string dict_path = kwarg_path("dict_path");
                if (dict_path.empty()) {
                    throw InvalidArgument("converter '" + id + "' requires a 'dict_path' kwarg");
                }
                const std::string engine_dir = find_engine_dict_dir(
                    cantonese ? "cantonese" : "mandarin", dict_dir, kwarg_path("engine_dict_path"));
                if (!file_exists(join_path(engine_dir, "word.txt"))) {
                    warn("cpp-pinyin dictionaries for '" + id + "' not found in '" + engine_dir +
                         "', skipping the converter (set the 'engine_dict_path' kwarg)");
                    continue;
                }
                // The registry defaults: pinyin serves Mandarin, jyutping Yue.
                std::vector<std::string> tags{ "zh", "zho", "cmn" };
                if (cantonese) tags = { "yue" };
                impl.converters.push_back(std::make_unique<ChineseConverter>(
                    languages_of(std::move(tags)), dict_path, engine_dir));
            } else if (id == "japanese-kana" || id == "japanese-mecab") {
                // The stock TIFA config declares `japanese-mecab`: MeCab
                // segments the lyrics and UniDic supplies the kana reading
                // (`pron`), which then flows through the kana converter.  The
                // kana converter shares this entry's dict_path and
                // double_written_sokuon kwargs and serves the id whenever no
                // usable UniDic dictionary is around -- otherwise nothing
                // would register a Japanese converter at all and `-l ja`
                // would fail on kana as well as kanji.
                const std::string dict_path = kwarg_path("dict_path");
                if (dict_path.empty()) {
                    throw InvalidArgument("converter '" + id + "' requires a 'dict_path' kwarg");
                }
                const bool sokuon = kwarg_bool("double_written_sokuon", false);
                const int  nbest  = kwarg_int("nbest", 32);

                std::unique_ptr<MecabTagger> tagger;
                std::string                  mecab_unavailable;
#ifndef TIFA_G2P_NO_MECAB
                if (id == "japanese-mecab") {
                    const std::string unidic_dir =
                        find_unidic_dir(dict_dir, kwarg_path("unidic_dir"));
                    if (!unidic_dir.empty()) {
                        try {
                            tagger = MecabTagger::open(unidic_dir);
                        } catch (const std::exception & e) {
                            mecab_unavailable = std::string("the UniDic dictionary at '") +
                                                unidic_dir + "' is unusable: " + e.what();
                        }
                    } else {
                        mecab_unavailable =
                            "no UniDic dictionary was found under '" + dict_dir + "'";
                    }
                }
#else
                if (id == "japanese-mecab") {
                    mecab_unavailable = "this build was compiled without MeCab";
                }
#endif
                if (!mecab_unavailable.empty()) {
                    warn("converter 'japanese-mecab': " + mecab_unavailable +
                         "; serving kana with the kana converter. For kanji lyrics, "
                         "extract the 'unidic-lite-dicdir' release asset to "
                         "<model dir>/unidic (or set the converter's 'unidic_dir' kwarg)");
                }
                if (tagger != nullptr) {
                    impl.converters.push_back(std::make_unique<JapaneseMecabConverter>(
                        languages_of({ "ja", "jpn" }), std::move(tagger),
                        std::make_unique<JapaneseKanaConverter>(std::vector<std::string>{},
                                                                dict_path, sokuon),
                        sokuon, nbest));
                } else {
                    impl.converters.push_back(std::make_unique<JapaneseKanaConverter>(
                        languages_of({ "ja", "jpn" }), dict_path, sokuon));
                }
            } else if (id == "dictionary") {
                const std::string dict_path = kwarg_path("dict_path");
                if (dict_path.empty()) {
                    throw InvalidArgument("converter 'dictionary' requires a 'dict_path' kwarg");
                }
                impl.converters.push_back(std::make_unique<DictionaryConverter>(
                    languages_of({}), dict_path));
            } else if (id == "passthrough") {
                impl.converters.push_back(
                    std::make_unique<PassthroughConverter>(languages_of({})));

            } else if (id == "characters") {
                std::unordered_map<char32_t, std::vector<std::string>> mapping;
                if (kwargs != nullptr) {
                    if (const json::Value * entries = kwargs->find("mapping");
                        entries != nullptr && entries->is_object()) {
                        for (const auto & entry : entries->obj) {
                            if (!entry.second.is_array()) continue;
                            std::size_t i = 0;
                            const char32_t c =
                                entry.first.empty() ? char32_t{ 0 } : decode_utf8(entry.first, i);
                            std::vector<std::string> phonemes;
                            for (const json::Value & value : entry.second.arr) {
                                if (value.is_string()) phonemes.push_back(value.str);
                            }
                            mapping.emplace(c, std::move(phonemes));
                        }
                    }
                }
                impl.converters.push_back(std::make_unique<CharactersConverter>(
                    languages_of({}), std::move(mapping)));
            } else if (id == "lstm") {
                // g2p/converters/lstm.py: dictionary-backed converter with a
                // char-LSTM for OOV words.  `model_path` is the reference's
                // ONNX directory; here it points at the converted GGUF (a
                // directory is accepted too — <dir>/lstm-g2p.gguf is used).
#ifdef TIFA_G2P_NO_LSTM
                warn("converter 'lstm' needs the ggml model backend, skipping");
#else
                const std::string model_path = kwarg_path("model_path");
                if (model_path.empty()) {
                    throw InvalidArgument("converter 'lstm' requires a 'model_path' kwarg");
                }
                int beam_size = 0;   // 0 = the value recorded in the GGUF
                if (kwargs != nullptr) {
                    if (const json::Value * value = kwargs->find("beam_size");
                        value != nullptr && value->is_number()) {
                        beam_size = static_cast<int>(value->number);
                    }
                }
                impl.converters.push_back(std::make_unique<LstmConverter>(
                    languages_of({}), kwarg_path("dict_path"), model_path, beam_size));
#endif
            } else {
                warn("unsupported converter '" + id + "', skipping");
            }
        }
    }

    return pipeline;
}

std::vector<Word> Pipeline::convert(const std::string & text,
                                    const std::vector<std::string> & languages) const {
    std::vector<const Converter *> active;
    for (const std::unique_ptr<Converter> & converter : impl_->converters) {
        const std::vector<std::string> & tags = converter->languages();
        if (languages.empty() || tags.empty()) {
            active.push_back(converter.get());
            continue;
        }
        for (const std::string & tag : tags) {
            if (std::find(languages.begin(), languages.end(), tag) != languages.end()) {
                active.push_back(converter.get());
                break;
            }
        }
    }
    if (active.empty()) {
        throw InvalidArgument("No converter matches the requested languages.");
    }

    std::vector<std::string> fragments{ text };
    for (const std::unique_ptr<FragmentPreprocessor> & processor : impl_->preprocessors) {
        fragments = processor->process(fragments);
    }

    // Route every fragment to the first converter that claims a run.  The
    // pending stack is lifo, so the left remainder is pushed after the right
    // one and therefore processed first.
    std::vector<std::pair<std::size_t, std::u32string>> claimed;
    std::vector<std::u32string> unrecognized;
    for (const std::string & fragment_text : fragments) {
        const std::u32string fragment = to_u32(fragment_text);

        std::vector<std::array<std::size_t, 3>> ranges;      // begin, end, converter
        std::vector<std::array<std::size_t, 3>> pending;
        pending.push_back({ 0, fragment.size(), 0 });
        while (!pending.empty()) {
            const std::array<std::size_t, 3> current = pending.back();
            pending.pop_back();
            const std::size_t begin    = current[0];
            const std::size_t end      = current[1];
            const std::size_t priority = current[2];
            if (begin == end) continue;

            const std::u32string part = fragment.substr(begin, end - begin);
            bool matched = false;
            for (std::size_t index = priority; index < active.size(); ++index) {
                const Match match = active[index]->find(part);
                if (!match.ok) continue;
                if (!(match.begin < match.end && match.end <= part.size())) {
                    throw InvalidArgument("converter find() returned an invalid range");
                }
                ranges.push_back({ begin + match.begin, begin + match.end, index });
                pending.push_back({ begin + match.end, end, index });
                pending.push_back({ begin, begin + match.begin, index + 1 });
                matched = true;
                break;
            }
            if (!matched && !is_blank(part)) unrecognized.push_back(part);
        }

        std::sort(ranges.begin(), ranges.end());
        for (const std::array<std::size_t, 3> & range : ranges) {
            claimed.emplace_back(range[2], fragment.substr(range[0], range[1] - range[0]));
        }
    }

    if (!unrecognized.empty()) {
        // Mirrors the reference's list repr, ['x', "y'z"], including its
        // quote selection, so the error text can be diffed against it.
        std::string message =
            "The following tokens could not be converted by any converter in the chain: [";
        for (std::size_t i = 0; i < unrecognized.size(); ++i) {
            const std::string token = to_utf8(unrecognized[i]);
            if (i != 0) message += ", ";
            if (token.find('\'') != std::string::npos &&
                token.find('"') == std::string::npos) {
                message += '"' + token + '"';
                continue;
            }
            message += '\'';
            for (char c : token) {
                if (c == '\'') message += '\\';
                message += c;
            }
            message += '\'';
        }
        message += "]";
        throw InvalidArgument(message);
    }

    std::vector<Word> result;
    for (const auto & claim : claimed) {
        const Converter * converter = active[claim.first];
        const std::string language  = resolve_language(converter->languages(), languages);
        const std::string run       = to_utf8(claim.second);
        // Converter-local preprocessors run on the claimed run, before convert().
        for (const std::string & part : converter->preprocess(run)) {
            if (part.empty()) continue;
            std::vector<Word> words = converter->convert(to_u32(part));
            for (Word & word : words) word.language = language;
            result.insert(result.end(), std::make_move_iterator(words.begin()),
                          std::make_move_iterator(words.end()));
        }
    }
    return result;
}

}  // namespace tifa_ggml::internal::g2p
