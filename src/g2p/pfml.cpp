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
    enum class Kind { Text, Scope, Word, Reading, Path, Group, Phoneme };
    Kind kind = Kind::Text;
    std::string text;          // Text: the run; Word/Phoneme: inner text
    std::string language;      // scope / word / phoneme attribute (raw)
    std::string script;        // word / group attribute
    std::string symbol;        // phoneme attribute
    std::vector<std::string> phonemes;   // word / group attribute (split)
    bool has_phonemes = false;           // a `phonemes` attribute was present
    bool has_text_attr = false;          // a `text` attribute was present
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
        if (name == "scope")        node.kind = Node::Kind::Scope;
        else if (name == "word")    node.kind = Node::Kind::Word;
        else if (name == "reading") node.kind = Node::Kind::Reading;
        else if (name == "path")    node.kind = Node::Kind::Path;
        else if (name == "group")   node.kind = Node::Kind::Group;
        else if (name == "phoneme") node.kind = Node::Kind::Phoneme;
        else parse_error("unknown element <" + name + ">", offset);

        // re-parse the attributes (open_tag_name left us right after the name)
        const std::vector<Attr> attrs = attributes();
        for (const Attr & a : attrs) {
            if (a.name == "language")      node.language = a.value;
            else if (a.name == "script")   node.script = a.value;
            else if (a.name == "symbol")   node.symbol = a.value;
            else if (a.name == "text")   { node.text = a.value; node.has_text_attr = true; }
            else if (a.name == "phonemes") {
                node.phonemes = split_ascii_whitespace(a.value);
                node.has_phonemes = true;
            }
            // `language-kind` is accepted and ignored: this model has no
            // separate "ANY" tag, an empty language already means any.
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
            if (node.kind == Node::Kind::Scope)   parse_error("<scope/> cannot be self-closing", offset);
            if (node.kind == Node::Kind::Reading) parse_error("<reading/> cannot be self-closing", offset);
            if (node.kind == Node::Kind::Path)    parse_error("<path/> cannot be self-closing", offset);
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

std::string join_with_spaces(const std::vector<std::string> & parts) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) out.push_back(' ');
        out += parts[i];
    }
    return out;
}

std::string first_language_tag(const std::string & language) {
    const std::vector<std::string> tags = split_language_tags(language);
    return tags.empty() ? std::string() : tags.front();
}

std::string text_children(const Node & node) {
    std::string joined;
    for (const Node & child : node.children) {
        if (child.kind == Node::Kind::Text) joined += child.text;
    }
    return joined;
}

// One <phoneme> -> its final symbol.  An explicit `language` attribute is
// prefixed as "<language>/<symbol>"; the symbol itself is never split, so
// symbol="zh/ong" and language="zh" symbol="ong" both yield "zh/ong".
std::string phoneme_of(const Node & node) {
    std::string symbol = node.symbol;
    if (symbol.empty()) symbol = trim(node.text + text_children(node));
    if (symbol.empty()) return symbol;
    return node.language.empty() ? symbol : node.language + "/" + symbol;
}

// One <group> -> Group.  A missing `script` falls back to the phonemes joined
// with single spaces (PFML 1.0 label filling).
Group group_of(const Node & node) {
    Group group;
    group.phonemes = node.phonemes;
    for (const Node & child : node.children) {
        if (child.kind != Node::Kind::Phoneme) continue;
        const std::string phoneme = phoneme_of(child);
        if (!phoneme.empty()) group.phonemes.push_back(phoneme);
    }
    group.script = node.script.empty() ? join_with_spaces(group.phonemes) : node.script;
    return group;
}

// The children of a <path> (or bare <group>/<phoneme> siblings) -> one Path.
// Consecutive bare phonemes form a single implicit group.
Path path_of(const std::vector<Node> & nodes) {
    Path path;
    std::vector<std::string> pending;
    const auto flush = [&] {
        if (pending.empty()) return;
        Group group;
        group.script   = join_with_spaces(pending);
        group.phonemes = pending;
        path.push_back(std::move(group));
        pending.clear();
    };
    for (const Node & node : nodes) {
        if (node.kind == Node::Kind::Group) {
            flush();
            path.push_back(group_of(node));
        } else if (node.kind == Node::Kind::Phoneme) {
            const std::string phoneme = phoneme_of(node);
            if (!phoneme.empty()) pending.push_back(phoneme);
        }
    }
    flush();
    return path;
}

// The children of a <reading> (or bare <path>/... siblings) -> its Paths.
std::vector<Path> paths_of(const std::vector<Node> & nodes) {
    std::vector<Path> paths;
    for (const Node & node : nodes) {
        if (node.kind == Node::Kind::Path) paths.push_back(path_of(node.children));
    }
    if (paths.empty()) paths.push_back(path_of(nodes));
    return paths;
}

// The pronunciation children of a <word> (or of the fragment) -> Readings.
// Containers may be omitted at any level; siblings must share one level.
std::vector<Reading> readings_of(const std::vector<Node> & nodes) {
    std::vector<Reading> readings;
    for (const Node & node : nodes) {
        if (node.kind == Node::Kind::Reading) readings.push_back(Reading{ paths_of(node.children) });
    }
    if (readings.empty()) readings.push_back(Reading{ paths_of(nodes) });
    return readings;
}

// Word text fallback when neither a `text` attribute nor character data is
// present: the phonemes of the first path, joined with spaces.
std::string phonemes_as_text(const std::vector<Reading> & readings) {
    if (readings.empty() || readings.front().paths.empty()) return std::string();
    std::vector<std::string> phonemes;
    for (const Group & group : readings.front().paths.front()) {
        phonemes.insert(phonemes.end(), group.phonemes.begin(), group.phonemes.end());
    }
    return join_with_spaces(phonemes);
}

// A compact `<word ... script="x" phonemes="a b">` is one reading / one path /
// one group; the script label falls back to the word text (the pre-existing
// behaviour for this form).
Word compact_word(const std::string & text, const std::string & language,
                  const std::string & script, std::vector<std::string> phonemes) {
    Word word;
    word.text = text.empty() ? join_with_spaces(phonemes) : text;
    word.language = first_language_tag(language);
    Group group;
    group.script   = script.empty() ? word.text : script;
    group.phonemes = std::move(phonemes);
    word.readings = { Reading{ { Path{ std::move(group) } } } };
    return word;
}

void convert_children(const Pipeline & pipeline, const std::vector<Node> & nodes,
                      const std::vector<std::string> & languages,
                      std::vector<Word> & out) {
    // Consecutive pronunciation elements with no <word> around them group into
    // one word (the containers are optional at the fragment level too).
    std::vector<Node> pending;

    const auto flush_pending = [&] {
        if (pending.empty()) return;
        Word word;
        word.readings = readings_of(pending);
        word.text     = phonemes_as_text(word.readings);
        out.push_back(std::move(word));
        pending.clear();
    };

    for (const Node & node : nodes) {
        switch (node.kind) {
            case Node::Kind::Text: {
                const std::string run = node.text;
                if (trim(run).empty()) continue;   // inter-tag whitespace
                flush_pending();
                std::vector<Word> words = pipeline.convert(run, languages);
                out.insert(out.end(), std::make_move_iterator(words.begin()),
                           std::make_move_iterator(words.end()));
                break;
            }
            case Node::Kind::Scope: {
                flush_pending();
                convert_children(pipeline, node.children,
                                 languages_of(node.language, languages), out);
                break;
            }
            case Node::Kind::Word: {
                flush_pending();
                const std::string inner = node.has_text_attr ? node.text
                                                             : trim(text_children(node));
                if (node.has_text_attr && !trim(text_children(node)).empty()) {
                    parse_error("<word> has both a text attribute and character data", 0);
                }
                bool has_pronunciation = node.has_phonemes;
                for (const Node & child : node.children) {
                    if (child.kind != Node::Kind::Text) { has_pronunciation = true; break; }
                }

                if (has_pronunciation) {
                    if (node.has_phonemes) {
                        std::vector<std::string> phonemes = node.phonemes;
                        for (const Node & child : node.children) {
                            if (child.kind != Node::Kind::Phoneme) continue;
                            const std::string phoneme = phoneme_of(child);
                            if (!phoneme.empty()) phonemes.push_back(phoneme);
                        }
                        out.push_back(compact_word(inner, node.language, node.script,
                                                   std::move(phonemes)));
                        break;
                    }
                    Word word;
                    word.language = first_language_tag(node.language);
                    word.readings = readings_of(node.children);
                    word.text     = inner.empty() ? phonemes_as_text(word.readings) : inner;
                    out.push_back(std::move(word));
                    break;
                }
                if (inner.empty()) break;   // <word language="zh"></word>
                std::vector<Word> words = pipeline.convert(
                    inner, languages_of(node.language, languages));
                out.insert(out.end(), std::make_move_iterator(words.begin()),
                           std::make_move_iterator(words.end()));
                break;
            }
            case Node::Kind::Reading:
            case Node::Kind::Path:
            case Node::Kind::Group:
            case Node::Kind::Phoneme: {
                pending.push_back(node);
                break;
            }
        }
    }
    flush_pending();
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

namespace {

void append_escaped(std::string & out, const std::string & value) {
    for (const char c : value) {
        switch (c) {
            case '&':  out += "&amp;";  break;
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '"':  out += "&quot;"; break;
            default:   out.push_back(c); break;
        }
    }
}

}  // namespace

std::string to_pfml(const std::vector<Word> & words) {
    std::string out;
    for (const Word & word : words) {
        out += "<word text=\"";
        append_escaped(out, word.text);
        out += "\" language=\"";
        append_escaped(out, word.language);
        out += "\"";

        // A word with a single reading / path / group serializes to the
        // compact form the documentation uses:
        //   <word text="重" language="zh" script="zhong" phonemes="zh ong"/>
        // Everything else needs the explicit container tree, because the
        // compact attributes cannot express alternatives.
        if (word.readings.size() == 1 && word.readings[0].paths.size() == 1 &&
            word.readings[0].paths[0].size() == 1) {
            const Group & group = word.readings[0].paths[0][0];
            out += " script=\"";
            append_escaped(out, group.script);
            out += "\" phonemes=\"";
            append_escaped(out, join_with_spaces(group.phonemes));
            out += "\"/>";
            continue;
        }

        out += ">";
        for (const Reading & reading : word.readings) {
            out += "<reading>";
            for (const Path & path : reading.paths) {
                out += "<path>";
                for (const Group & group : path) {
                    out += "<group script=\"";
                    append_escaped(out, group.script);
                    out += "\">";
                    for (const std::string & phoneme : group.phonemes) {
                        // Verbatim: symbol="zh/ong" and language="zh"
                        // symbol="ong" are equivalent, so no split is needed.
                        out += "<phoneme symbol=\"";
                        append_escaped(out, phoneme);
                        out += "\"/>";
                    }
                    out += "</group>";
                }
                out += "</path>";
            }
            out += "</reading>";
        }
        out += "</word>";
    }
    return out;
}

}  // namespace tifa_ggml::internal::g2p
