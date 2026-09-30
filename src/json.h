#pragma once

// Internal JSON parser / serializer.  Not part of the public API.
//
// Minimal, dependency-free recursive-descent implementation of the JSON
// grammar (RFC 8259).  Typical use:
//
//     auto doc = tifa_ggml::internal::json::parse(text);   // throws InvalidArgument
//     if (const auto * syms = doc.find("symbols")) {
//         for (const auto & kv : syms->obj) {
//             const double id = kv.second.as_number(-1.0);
//             ...
//         }
//     }
//     const std::string pretty = tifa_ggml::internal::json::dump(doc, 2);
//
// Notes:
//  - Objects keep insertion order and retain duplicate keys; `find()` returns
//    the *last* entry with the key, matching common JSON semantics.
//  - `parse` rejects malformed input, trailing garbage and nesting deeper than
//    256 levels by throwing `InvalidArgument`; the message carries the byte
//    offset of the failure.
//  - `dump` escapes only what JSON requires (`\" \\ \n \r \t \b \f` and other
//    control characters as `\u00XX`); non-ASCII bytes pass through unchanged,
//    so UTF-8 keys round-trip.  Non-finite numbers are emitted as `null`
//    (JSON has no representation for inf/nan).

#include "tifa_ggml/errors.h"

#include <string>
#include <utility>
#include <vector>

namespace tifa_ggml::internal::json {

// A parsed JSON value.  A single tagged struct is used instead of a variant so
// that callers can also build values by hand before dumping them.
struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;

    bool        boolean = false;
    double      number  = 0.0;
    std::string str;
    std::vector<Value>                         arr;
    std::vector<std::pair<std::string, Value>> obj;  // insertion-ordered

    bool is_null()   const noexcept;
    bool is_bool()   const noexcept;
    bool is_number() const noexcept;
    bool is_string() const noexcept;
    bool is_array()  const noexcept;
    bool is_object() const noexcept;

    // Object lookup; returns nullptr when absent or not an object.
    const Value * find(const std::string & key) const;

    // Convenience accessors with fallbacks (never throw).  `as_string` returns
    // a reference, so the fallback must outlive the call (pass a named string,
    // not a temporary).
    bool               as_bool  (bool def = false) const noexcept;
    double             as_number(double def = 0.0) const noexcept;
    const std::string &as_string(const std::string & def) const;
};

// Parse a complete JSON document.  Throws `InvalidArgument` (message includes
// the byte offset) on malformed input or trailing garbage.
Value parse(const std::string & text);

// Serialize `v`.  `indent < 0` produces a compact single line; `indent >= 0`
// pretty-prints with that many spaces per nesting level.
std::string dump(const Value & v, int indent = -1);

}  // namespace tifa_ggml::internal::json
