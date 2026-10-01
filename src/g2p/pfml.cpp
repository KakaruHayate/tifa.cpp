// PFML parser + converter (see pfml.h).
//
// A small tolerant XML-fragment scanner: elements are <scope>, <word> and
// <phoneme>; attributes are name="value" (single quotes accepted); entities
// &amp; &lt; &gt; &quot; &apos; and numeric &#NN; / &#xNN; are decoded.
// Self-closing <word .../> and <phoneme/> are accepted.  Anything else —
// unknown tags, unclosed elements, stray '<' — raises InvalidArgument with
// the byte offset, since silently dropping transcript text would mis-align.

#include "pfml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace tifa_ggml::internal::g2p {

namespace {

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::string trim(const std::string & s) {
    std::size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) ++b;
    while (e > b && is_space(s[e - 1])) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> split_ascii_whitespace(const std::string & s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && is_space(s[i])) ++i;
        const std::size_t b = i;
        while (i < s.size() && !is_space(s[i])) ++i;
        if (i > b) out.push_back(s.substr(b, i - b));
    }
    return out;
}

// "zh, zho cmn" -> {"zh", "zho", "cmn"} (first tag wins for routing)
std::vector<std::string> split_language_tags(const std::string & text) {
    std::vector<std::string> tags;
    std::string cur;
    for (const char c : text) {
        if (c == ',' || is_space(c)) {
            if (!cur.empty()) { tags.push_back(cur); cur.clear(); }
            continue;
        }
        cur.push_back(c);
    }
    if (!cur.empty()) tags.push_back(cur);
    return tags;
}

std::string decode_entities(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '&') { out.push_back(s[i]); continue; }
        const std::size_t semi = s.find(';', i + 1);
        if (semi == std::string::npos || semi - i > 12) { out.push_back('&'); continue; }
        const std::string ent = s.substr(i + 1, semi - i - 1);
        if (ent == "amp")       { out.push_back('&'); }
        else if (ent == "lt")   { out.push_back('<'); }
        else if (ent == "gt")   { out.push_back('>'); }
        else if (ent == "quot") { out.push_back('"'); }
        else if (ent == "apos") { out.push_back('\''); }
        else if (!ent.empty() && ent[0] == '#') {
            const bool hex = ent.size() > 1 && (ent[1] == 'x' || ent[1] == 'X');
            const char * digits = ent.c_str() + (hex ? 2 : 1);
            char * end = nullptr;
            const long cp = std::strtol(digits, &end, hex ? 16 : 10);
            if (end == digits || cp <= 0 || cp > 0x10FFFF) { out.push_back('&'); continue; }
            // encode as UTF-8
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
        } else {
            out.push_back('&');
            continue;   // keep scanning from '&' as literal text
        }
        i = semi;
    }
    return out;
}

struct Attr {
    std::string name;
    std::string value;
};

struct Node {
    enum class Kind { Text, Scope, Word, Phoneme };
    Kind kind = Kind::Text;
    std::string text;          // Text: the run; Word/Phoneme: inner text
    std::string language;      // scope / word attribute (raw)
    std::string script;        // word attribute
    std::vector<std::string> phonemes;   // word attribute (split) or <phoneme> children
    bool has_phonemes = false;           // a `phonemes` attribute was present
    std::vector<Node> children;
};

[[noreturn]] void parse_error(const std::string & what, std::size_t at) {
    throw InvalidArgument("PFML: " + what + " at byte " + std::to_string(at));
}

class Parser {
public:
    explicit Parser(const std::string & text) : s_(text) {}

    std::vector<Node> parse() {
        std::vector<Node> nodes = children(std::string());
        if (i_ < s_.size()) parse_error("unexpected content", i_);
        return nodes;
    }

private:
    const std::string & s_;
    std::size_t i_ = 0;

    bool starts_with(const char * prefix) const {
        const std::size_t n = std::strlen(prefix);
        return s_.compare(i_, n, prefix) == 0;
    }

    // Returns the element name when s_[i_] starts a tag; advances past the
    // '<' and the name.  Empty name = not a tag.
    std::string open_tag_name() {
        if (i_ >= s_.size() || s_[i_] != '<') return std::string();
        std::size_t j = i_ + 1;
        if (j < s_.size() && (s_[j] == '/' || s_[j] == '!' || s_[j] == '?')) return std::string();
        const std::size_t b = j;
        while (j < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[j])) ||
                                 s_[j] == '-' || s_[j] == '_')) {
            ++j;
        }
        if (j == b) return std::string();
        if (j >= s_.size() || (s_[j] != '>' && s_[j] != '/' && !is_space(s_[j]))) {
            return std::string();
        }
        const std::string name = s_.substr(b, j - b);
        i_ = j;
        return name;
    }

    std::vector<Attr> attributes() {
        std::vector<Attr> attrs;
        while (i_ < s_.size()) {
            while (i_ < s_.size() && is_space(s_[i_])) ++i_;
            if (i_ >= s_.size()) parse_error("unterminated tag", i_);
            if (s_[i_] == '>' || s_[i_] == '/') return attrs;
            const std::size_t b = i_;
            while (i_ < s_.size() && s_[i_] != '=' && !is_space(s_[i_]) &&
                   s_[i_] != '>' && s_[i_] != '/') {
                ++i_;
            }
            const std::string name = s_.substr(b, i_ - b);
            while (i_ < s_.size() && is_space(s_[i_])) ++i_;
            if (i_ >= s_.size() || s_[i_] != '=') parse_error("attribute without a value", b);
            ++i_;
            while (i_ < s_.size() && is_space(s_[i_])) ++i_;
            if (i_ >= s_.size() || (s_[i_] != '"' && s_[i_] != '\'')) {
                parse_error("attribute value must be quoted", i_);
            }
            const char quote = s_[i_++];
            const std::size_t vb = i_;
            while (i_ < s_.size() && s_[i_] != quote) ++i_;
            if (i_ >= s_.size()) parse_error("unterminated attribute value", vb);
            attrs.push_back({ name, decode_entities(s_.substr(vb, i_ - vb)) });
            ++i_;
        }
        return attrs;
    }

    Node element(const std::string & name, std::size_t offset) {
        Node node;
        if (name == "scope")      node.kind = Node::Kind::Scope;
        else if (name == "word")  node.kind = Node::Kind::Word;
        else if (name == "phoneme") node.kind = Node::Kind::Phoneme;
        else parse_error("unknown element <" + name + ">", offset);

        // re-parse the attributes (open_tag_name left us right after the name)
        const std::vector<Attr> attrs = attributes();
        for (const Attr & a : attrs) {
            if (a.name == "language")      node.language = a.value;
            else if (a.name == "script")   node.script = a.value;
            else if (a.name == "phonemes") {
                node.phonemes = split_ascii_whitespace(a.value);
                node.has_phonemes = true;
            }
            // unknown attributes are ignored (forward compatibility)
        }
        const bool self_closing = [&] {
            if (s_[i_] == '/') {
                ++i_;
                if (i_ >= s_.size() || s_[i_] != '>') parse_error("malformed self-closing tag", i_);
                ++i_;
                return true;
            }
            if (s_[i_] != '>') parse_error("malformed tag", i_);
            ++i_;
            return false;
        }();

        if (self_closing) {
            if (node.kind == Node::Kind::Scope) parse_error("<scope/> cannot be self-closing", offset);
            return node;
        }
        node.children = children(name);
        return node;
    }

    // Parse siblings until `closing` (empty = end of input).  Text runs
    // between tags become Text nodes; whitespace-only runs between tags are
    // kept (they separate words).
    std::vector<Node> children(const std::string & closing) {
        std::vector<Node> nodes;
        std::string text;
        const auto flush = [&] {
            if (!text.empty()) {
                Node n;
                n.kind = Node::Kind::Text;
                n.text = decode_entities(text);
                nodes.push_back(std::move(n));
                text.clear();
            }
        };
        while (i_ < s_.size()) {
            if (s_[i_] == '<') {
                if (starts_with("</")) {
                    const std::size_t close_at = i_;
                    i_ += 2;
                    const std::size_t b = i_;
                    while (i_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[i_])) ||
                                              s_[i_] == '-' || s_[i_] == '_')) {
                        ++i_;
                    }
                    const std::string name = s_.substr(b, i_ - b);
                    while (i_ < s_.size() && is_space(s_[i_])) ++i_;
                    if (i_ >= s_.size() || s_[i_] != '>') parse_error("malformed closing tag", close_at);
                    ++i_;
                    if (closing.empty()) parse_error("unexpected </" + name + ">", close_at);
                    if (name != closing) {
                        parse_error("</" + name + "> closes <" + closing + ">", close_at);
                    }
                    flush();
                    return nodes;
                }
                const std::size_t tag_at = i_;
                const std::string name = open_tag_name();
                if (name.empty()) parse_error("stray '<' (use &lt; for literal text)", tag_at);
                flush();
                nodes.push_back(element(name, tag_at));
                continue;
            }
            text.push_back(s_[i_++]);
        }
        if (!closing.empty()) parse_error("unclosed <" + closing + ">", s_.size());
        flush();
        return nodes;
    }
};

std::vector<std::string> languages_of(const std::string & language,
                                      const std::vector<std::string> & fallback) {
    const std::vector<std::string> tags = split_language_tags(language);
    return tags.empty() ? fallback : tags;
}

// One word from direct final phonemes: a single reading / path / group.  The
// group script label falls back to the word text (or the joined phonemes)
// when no `script` attribute is given.
Word direct_word(const std::string & text, const std::string & language,
                 const std::string & script, std::vector<std::string> phonemes) {
    Word word;
    if (!text.empty()) {
        word.text = text;
    } else {
        for (const std::string & p : phonemes) {
            if (!word.text.empty()) word.text.push_back(' ');
            word.text += p;
        }
    }
    const std::vector<std::string> tags = split_language_tags(language);
    word.language = tags.empty() ? std::string() : tags.front();
    Group group;
    group.script = script.empty() ? word.text : script;
    group.phonemes = std::move(phonemes);
    word.readings = { Reading{ { Path{ std::move(group) } } } };
    return word;
}

void convert_children(const Pipeline & pipeline, const std::vector<Node> & nodes,
                      const std::vector<std::string> & languages,
                      std::vector<Word> & out) {
    // Consecutive <phoneme> siblings group into one word (the containers
    // around a phoneme run may be omitted).
    std::vector<std::string> pending_phonemes;

    const auto flush_phonemes = [&] {
        if (pending_phonemes.empty()) return;
        out.push_back(direct_word(std::string(), std::string(), std::string(),
                                  std::move(pending_phonemes)));
        pending_phonemes.clear();
    };

    for (const Node & node : nodes) {
        switch (node.kind) {
            case Node::Kind::Text: {
                const std::string run = node.text;
                if (trim(run).empty()) continue;   // inter-tag whitespace
                flush_phonemes();
                std::vector<Word> words = pipeline.convert(run, languages);
                out.insert(out.end(), std::make_move_iterator(words.begin()),
                           std::make_move_iterator(words.end()));
                break;
            }
            case Node::Kind::Scope: {
                flush_phonemes();
                convert_children(pipeline, node.children,
                                 languages_of(node.language, languages), out);
                break;
            }
            case Node::Kind::Word: {
                flush_phonemes();
                // <phoneme> children of a <word> are its phonemes
                std::vector<std::string> phonemes = node.phonemes;
                for (const Node & child : node.children) {
                    if (child.kind != Node::Kind::Phoneme) continue;
                    if (!child.text.empty()) phonemes.push_back(child.text);
                    phonemes.insert(phonemes.end(), child.phonemes.begin(),
                                    child.phonemes.end());
                }
                const std::string inner = trim([&] {
                    std::string joined;
                    for (const Node & child : node.children) {
                        if (child.kind == Node::Kind::Text) joined += child.text;
                    }
                    return joined;
                }());

                if (node.has_phonemes || !phonemes.empty()) {
                    out.push_back(direct_word(inner, node.language, node.script,
                                              std::move(phonemes)));
                    break;
                }
                if (inner.empty()) break;   // <word language="zh"></word>
                std::vector<Word> words = pipeline.convert(
                    inner, languages_of(node.language, languages));
                out.insert(out.end(), std::make_move_iterator(words.begin()),
                           std::make_move_iterator(words.end()));
                break;
            }
            case Node::Kind::Phoneme: {
                // inner text is the phoneme name (children hold it)
                std::string text = node.text;
                for (const Node & child : node.children) {
                    if (child.kind == Node::Kind::Text) text += child.text;
                }
                text = trim(text);
                if (!text.empty()) pending_phonemes.push_back(text);
                pending_phonemes.insert(pending_phonemes.end(),
                                        node.phonemes.begin(), node.phonemes.end());
                break;
            }
        }
    }
    flush_phonemes();
}

}  // namespace

bool looks_like_pfml(const std::string & text) {
    static const char * kNames[] = { "scope", "word", "phoneme", nullptr };
    for (std::size_t i = 0; i + 1 < text.size(); ++i) {
        if (text[i] != '<') continue;
        std::size_t j = i + 1;
        if (text[j] == '/') ++j;
        for (const char ** name = kNames; *name != nullptr; ++name) {
            const std::size_t n = std::strlen(*name);
            if (text.compare(j, n, *name) != 0) continue;
            const char after = j + n < text.size() ? text[j + n] : '\0';
            if (after != '>' && after != '/' && !is_space(after) && after != '\0') continue;
            // require a well-formed tag end to avoid treating "a < word > b"
            // prose as markup only when it actually closes
            if (text.find('>', j) == std::string::npos) continue;
            return true;
        }
    }
    return false;
}

std::vector<Word> convert_pfml(const Pipeline & pipeline, const std::string & text,
                               const std::vector<std::string> & languages) {
    Parser parser(text);
    const std::vector<Node> nodes = parser.parse();
    std::vector<Word> words;
    convert_children(pipeline, nodes, languages, words);
    return words;
}

}  // namespace tifa_ggml::internal::g2p
