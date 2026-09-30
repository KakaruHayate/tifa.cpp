#include "json.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

namespace tifa_ggml::internal::json {

// ---------------------------------------------------------------------------
// Value accessors
// ---------------------------------------------------------------------------

bool Value::is_null()   const noexcept { return type == Type::Null; }
bool Value::is_bool()   const noexcept { return type == Type::Bool; }
bool Value::is_number() const noexcept { return type == Type::Number; }
bool Value::is_string() const noexcept { return type == Type::String; }
bool Value::is_array()  const noexcept { return type == Type::Array; }
bool Value::is_object() const noexcept { return type == Type::Object; }

const Value * Value::find(const std::string & key) const {
    if (type != Type::Object) return nullptr;
    // Duplicate keys are kept in `obj`; the last one wins, so scan backwards.
    for (auto it = obj.rbegin(); it != obj.rend(); ++it) {
        if (it->first == key) return &it->second;
    }
    return nullptr;
}

bool Value::as_bool(bool def) const noexcept {
    return type == Type::Bool ? boolean : def;
}

double Value::as_number(double def) const noexcept {
    return type == Type::Number ? number : def;
}

const std::string & Value::as_string(const std::string & def) const {
    return type == Type::String ? str : def;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

namespace {

// Nesting cap: recursive descent would otherwise overflow the stack on
// adversarial input long before the file size became a problem.
constexpr int k_max_depth = 256;

void append_utf8(std::string & out, uint32_t cp) {
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

class Parser {
public:
    explicit Parser(const std::string & text) : s_(text) {}

    Value parse_document() {
        skip_ws();
        Value v = parse_value(0);
        skip_ws();
        if (!eof()) fail("trailing garbage after JSON value");
        return v;
    }

private:
    const std::string & s_;
    size_t              i_ = 0;

    bool eof() const { return i_ >= s_.size(); }
    char peek() const { return eof() ? '\0' : s_[i_]; }

    [[noreturn]] void fail(const std::string & msg) const {
        throw InvalidArgument("bad JSON at offset " + std::to_string(i_) + ": " + msg);
    }

    void skip_ws() {
        while (!eof() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) {
            ++i_;
        }
    }

    // -- values --

    Value parse_value(int depth) {
        skip_ws();
        if (eof()) fail("unexpected end of input while parsing a value");
        switch (peek()) {
            case '{': return parse_object(depth);
            case '[': return parse_array(depth);
            case '"': {
                Value v;
                v.type = Value::Type::String;
                v.str  = parse_string();
                return v;
            }
            case 't': {
                expect_literal("true");
                Value v;
                v.type    = Value::Type::Bool;
                v.boolean = true;
                return v;
            }
            case 'f': {
                expect_literal("false");
                Value v;
                v.type    = Value::Type::Bool;
                v.boolean = false;
                return v;
            }
            case 'n':
                expect_literal("null");
                return Value{};
            default:
                if (peek() == '-' || (peek() >= '0' && peek() <= '9')) {
                    Value v;
                    v.type   = Value::Type::Number;
                    v.number = parse_number();
                    return v;
                }
                fail(std::string("unexpected character '") + peek() + "' while parsing a value");
        }
    }

    Value parse_object(int depth) {
        if (depth >= k_max_depth) fail("maximum nesting depth (256) exceeded");
        Value v;
        v.type = Value::Type::Object;
        ++i_;  // consume '{'
        skip_ws();
        if (peek() == '}') {
            ++i_;
            return v;
        }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("expected a string key in object");
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') fail("expected ':' after object key");
            ++i_;
            // Duplicate keys are kept as-is; `find()` lets the last one win.
            v.obj.emplace_back(std::move(key), parse_value(depth + 1));
            skip_ws();
            if (eof()) fail("unexpected end of input inside object");
            const char c = s_[i_++];
            if (c == ',') continue;
            if (c == '}') break;
            fail(std::string("expected ',' or '}' in object, got '") + c + "'");
        }
        return v;
    }

    Value parse_array(int depth) {
        if (depth >= k_max_depth) fail("maximum nesting depth (256) exceeded");
        Value v;
        v.type = Value::Type::Array;
        ++i_;  // consume '['
        skip_ws();
        if (peek() == ']') {
            ++i_;
            return v;
        }
        while (true) {
            v.arr.push_back(parse_value(depth + 1));
            skip_ws();
            if (eof()) fail("unexpected end of input inside array");
            const char c = s_[i_++];
            if (c == ',') continue;
            if (c == ']') break;
            fail(std::string("expected ',' or ']' in array, got '") + c + "'");
        }
        return v;
    }

    // -- strings --

    std::string parse_string() {
        ++i_;  // consume opening '"'
        std::string out;
        while (true) {
            if (eof()) fail("unterminated string");
            const char c = s_[i_++];
            if (c == '"') return out;
            if (c == '\\') {
                parse_escape(out);
                continue;
            }
            if (static_cast<unsigned char>(c) < 0x20) fail("unescaped control character in string");
            out.push_back(c);  // other bytes, including UTF-8, pass through
        }
    }

    void parse_escape(std::string & out) {
        if (eof()) fail("unterminated escape sequence");
        const char e = s_[i_++];
        switch (e) {
            case '"':  out.push_back('"');  break;
            case '\\': out.push_back('\\'); break;
            case '/':  out.push_back('/');  break;
            case 'b':  out.push_back('\b'); break;
            case 'f':  out.push_back('\f'); break;
            case 'n':  out.push_back('\n'); break;
            case 'r':  out.push_back('\r'); break;
            case 't':  out.push_back('\t'); break;
            case 'u':  parse_unicode_escape(out); break;
            default:   fail(std::string("invalid escape '\\") + e + "'");
        }
    }

    void parse_unicode_escape(std::string & out) {
        uint32_t cp = parse_hex4();
        if (cp >= 0xD800 && cp <= 0xDBFF) {
            // High surrogate: a matching low surrogate must follow.
            if (i_ + 1 >= s_.size() || s_[i_] != '\\' || s_[i_ + 1] != 'u') {
                fail("high surrogate not followed by a \\u escape");
            }
            i_ += 2;
            const uint32_t lo = parse_hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid low surrogate in \\u escape");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("unpaired low surrogate in \\u escape");
        }
        append_utf8(out, cp);
    }

    uint32_t parse_hex4() {
        if (i_ + 4 > s_.size()) fail("truncated \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            v = (v << 4) | hex_digit(s_[i_ + k]);
        }
        i_ += 4;
        return v;
    }

    uint32_t hex_digit(char c) const {
        if (c >= '0' && c <= '9') return static_cast<uint32_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<uint32_t>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<uint32_t>(c - 'A' + 10);
        fail("invalid hex digit in \\u escape");
    }

    // -- numbers --

    double parse_number() {
        const size_t start = i_;
        if (peek() == '-') ++i_;

        if (eof()) fail("unexpected end of input in number");
        if (s_[i_] == '0') {
            ++i_;  // no leading zeros allowed
        } else if (s_[i_] >= '1' && s_[i_] <= '9') {
            while (!eof() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        } else {
            fail("expected a digit");
        }

        if (!eof() && s_[i_] == '.') {
            ++i_;
            if (eof() || s_[i_] < '0' || s_[i_] > '9') fail("expected a digit after '.'");
            while (!eof() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }

        if (!eof() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (!eof() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            if (eof() || s_[i_] < '0' || s_[i_] > '9') fail("expected a digit in exponent");
            while (!eof() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }

        const std::string tok = s_.substr(start, i_ - start);
        return std::strtod(tok.c_str(), nullptr);
    }

    void expect_literal(const char * lit) {
        for (const char * p = lit; *p != '\0'; ++p) {
            if (eof() || s_[i_] != *p) {
                fail(std::string("invalid literal, expected \"") + lit + '"');
            }
            ++i_;
        }
    }
};

}  // namespace

Value parse(const std::string & text) {
    Parser p(text);
    return p.parse_document();
}

// ---------------------------------------------------------------------------
// Serializer
// ---------------------------------------------------------------------------

namespace {

void escape_string(const std::string & s, std::string & out) {
    static const char hex[] = "0123456789abcdef";
    out.push_back('"');
    for (size_t k = 0; k < s.size(); ++k) {
        const unsigned char c = static_cast<unsigned char>(s[k]);
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 0x0F]);
                } else {
                    out.push_back(static_cast<char>(c));  // UTF-8 passes through
                }
        }
    }
    out.push_back('"');
}

void append_number(double v, std::string & out) {
    if (!std::isfinite(v)) {
        out += "null";  // JSON cannot represent inf/nan
        return;
    }
    char buf[40];
    if (v == std::floor(v) && std::fabs(v) < 1e15) {
        std::snprintf(buf, sizeof(buf), "%.0f", v);
        out += buf;
        return;
    }
    // Shortest representation that still parses back to the same double.
    for (int prec = 15; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    out += buf;
}

class Dumper {
public:
    explicit Dumper(int indent) : indent_(indent) {}

    std::string run(const Value & v) {
        write(v, 0);
        return std::move(out_);
    }

private:
    std::string out_;
    int         indent_;

    void newline(int depth) {
        if (indent_ < 0) return;
        out_.push_back('\n');
        out_.append(static_cast<size_t>(indent_) * static_cast<size_t>(depth), ' ');
    }

    void write(const Value & v, int depth) {
        switch (v.type) {
            case Value::Type::Null:
                out_ += "null";
                break;
            case Value::Type::Bool:
                out_ += v.boolean ? "true" : "false";
                break;
            case Value::Type::Number:
                append_number(v.number, out_);
                break;
            case Value::Type::String:
                escape_string(v.str, out_);
                break;
            case Value::Type::Array:
                if (v.arr.empty()) {
                    out_ += "[]";
                    break;
                }
                out_.push_back('[');
                for (size_t k = 0; k < v.arr.size(); ++k) {
                    if (k > 0) out_.push_back(',');
                    newline(depth + 1);
                    write(v.arr[k], depth + 1);
                }
                newline(depth);
                out_.push_back(']');
                break;
            case Value::Type::Object:
                if (v.obj.empty()) {
                    out_ += "{}";
                    break;
                }
                out_.push_back('{');
                for (size_t k = 0; k < v.obj.size(); ++k) {
                    if (k > 0) out_.push_back(',');
                    newline(depth + 1);
                    escape_string(v.obj[k].first, out_);
                    out_.push_back(':');
                    if (indent_ >= 0) out_.push_back(' ');
                    write(v.obj[k].second, depth + 1);
                }
                newline(depth);
                out_.push_back('}');
                break;
        }
    }
};

}  // namespace

std::string dump(const Value & v, int indent) {
    Dumper d(indent);
    return d.run(v);
}

}  // namespace tifa_ggml::internal::json
