#include "gguf_io.h"

#include "json.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <utility>

namespace tifa_ggml::internal {

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

GgufFile GgufFile::open(const std::string & path) {
    gguf_init_params params{};
    params.no_alloc = true;    // don't allocate the tensor data region
    params.ctx      = nullptr; // we don't want ggml tensors materialised here

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        throw GgufError("failed to open GGUF file: " + path);
    }
    GgufFile f;
    f.ctx_  = ctx;
    f.path_ = path;
    return f;
}

GgufFile::~GgufFile() {
    if (ctx_) gguf_free(ctx_);
}

GgufFile::GgufFile(GgufFile && other) noexcept
    : ctx_(other.ctx_), path_(std::move(other.path_)) {
    other.ctx_ = nullptr;
}

GgufFile & GgufFile::operator=(GgufFile && other) noexcept {
    if (this != &other) {
        if (ctx_) gguf_free(ctx_);
        ctx_  = other.ctx_;
        path_ = std::move(other.path_);
        other.ctx_ = nullptr;
    }
    return *this;
}

bool GgufFile::has(const std::string & key) const {
    return gguf_find_key(ctx_, key.c_str()) >= 0;
}

// -- typed getters --

namespace {
    [[noreturn]] void wrong_type(const std::string & key, const char * wanted, const char * got) {
        throw GgufError("GGUF key '" + key + "' has wrong type: wanted " + wanted + ", got " + got);
    }
    [[noreturn]] void missing(const std::string & key) {
        throw GgufError("GGUF key '" + key + "' is missing");
    }
}

std::string GgufFile::get_string(const std::string & key) const {
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) missing(key);
    const auto t = gguf_get_kv_type(ctx_, id);
    if (t != GGUF_TYPE_STRING) wrong_type(key, "STRING", gguf_type_name(t));
    return std::string(gguf_get_val_str(ctx_, id));
}

int64_t GgufFile::get_int(const std::string & key) const {
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) missing(key);
    const auto t = gguf_get_kv_type(ctx_, id);
    switch (t) {
        case GGUF_TYPE_INT8:   return gguf_get_val_i8  (ctx_, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16 (ctx_, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32 (ctx_, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64 (ctx_, id);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8  (ctx_, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16 (ctx_, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32 (ctx_, id);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx_, id));
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(ctx_, id) ? 1 : 0;
        default: wrong_type(key, "INT*", gguf_type_name(t));
    }
}

float GgufFile::get_float(const std::string & key) const {
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) missing(key);
    const auto t = gguf_get_kv_type(ctx_, id);
    switch (t) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx_, id);
        case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(ctx_, id));
        case GGUF_TYPE_INT32:   return static_cast<float>(gguf_get_val_i32(ctx_, id));
        case GGUF_TYPE_INT64:   return static_cast<float>(gguf_get_val_i64(ctx_, id));
        default: wrong_type(key, "FLOAT*", gguf_type_name(t));
    }
}

bool GgufFile::get_bool(const std::string & key) const {
    const int64_t id = gguf_find_key(ctx_, key.c_str());
    if (id < 0) missing(key);
    const auto t = gguf_get_kv_type(ctx_, id);
    if (t != GGUF_TYPE_BOOL) wrong_type(key, "BOOL", gguf_type_name(t));
    return gguf_get_val_bool(ctx_, id);
}

std::optional<std::string> GgufFile::get_string_opt(const std::string & key) const {
    return has(key) ? std::optional<std::string>(get_string(key)) : std::nullopt;
}
std::optional<int64_t> GgufFile::get_int_opt(const std::string & key) const {
    return has(key) ? std::optional<int64_t>(get_int(key)) : std::nullopt;
}
std::optional<float> GgufFile::get_float_opt(const std::string & key) const {
    return has(key) ? std::optional<float>(get_float(key)) : std::nullopt;
}
std::optional<bool> GgufFile::get_bool_opt(const std::string & key) const {
    return has(key) ? std::optional<bool>(get_bool(key)) : std::nullopt;
}

// -- tensor table --

std::vector<GgufFile::TensorInfo> GgufFile::list_tensors() const {
    const int64_t n = gguf_get_n_tensors(ctx_);
    std::vector<TensorInfo> out;
    out.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        TensorInfo info;
        info.name = gguf_get_tensor_name(ctx_, i);
        info.type = static_cast<int32_t>(gguf_get_tensor_type(ctx_, i));
        info.size_bytes = gguf_get_tensor_size(ctx_, i);
        info.offset     = gguf_get_tensor_offset(ctx_, i);
        // GGUF tensors store ne[] inline; we query it via the ggml_tensor
        // structure created by gguf_init_from_file when params.ctx is null,
        // so instead we recover the shape from the public helper by
        // re-opening with a context.  As a lightweight alternative that
        // avoids re-opening, we walk the gguf internal struct via the
        // n_dims / ne[] read by the library during init — but the public
        // C API exposes these only through the ggml_tensor path.  For the
        // inspector we can still render useful information using
        // size_bytes + type, and shape is filled in the richer loader in
        // later tasks.  Leave `shape` empty here for now.
        out.push_back(std::move(info));
    }
    return out;
}

std::vector<std::string> GgufFile::list_keys() const {
    const int64_t n = gguf_get_n_kv(ctx_);
    std::vector<std::string> out;
    out.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_key(ctx_, i));
    }
    return out;
}

size_t GgufFile::total_params() const {
    // Accurate param count requires reading shape arrays; the simplest path
    // that works for all ggml types is to re-open with a ggml_context and
    // count via ggml_nelements().  We prefer not to pay that cost in the
    // inspector, so estimate from tensor sizes assuming FP32 (true for this
    // project's v1 schema).  If mixed types appear later we'll switch to
    // the exact path here.
    const int64_t n = gguf_get_n_tensors(ctx_);
    size_t params = 0;
    for (int64_t i = 0; i < n; ++i) {
        const auto t    = gguf_get_tensor_type(ctx_, i);
        const size_t sz = gguf_get_tensor_size(ctx_, i);
        // For FP32 assume 4 bytes/element.  For other types we'd need the
        // per-type block_size / type_size from ggml_type_traits.
        if (t == GGML_TYPE_F32) {
            params += sz / 4;
        } else {
            params += sz;
        }
    }
    return params;
}

// ---------------------------------------------------------------------------
// JSON parsing for flat {"str":int} objects
// ---------------------------------------------------------------------------

namespace {

struct JsonCursor {
    const std::string & s;
    size_t i = 0;

    bool eof() const { return i >= s.size(); }
    char peek() const { return eof() ? '\0' : s[i]; }
    char next() { return eof() ? '\0' : s[i++]; }

    void skip_ws() {
        while (!eof() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }

    [[noreturn]] void fail(const std::string & msg) const {
        throw InvalidArgument("bad JSON at offset " + std::to_string(i) + ": " + msg);
    }

    void expect(char c) {
        skip_ws();
        if (next() != c) fail(std::string("expected '") + c + "'");
    }

    std::string parse_string() {
        skip_ws();
        if (next() != '"') fail("expected string");
        std::string out;
        while (!eof()) {
            char c = next();
            if (c == '"') return out;
            if (c == '\\') {
                char e = next();
                switch (e) {
                    case '"': out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/');  break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    default:  fail("unsupported escape");
                }
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }

    int parse_int() {
        skip_ws();
        std::string tok;
        if (peek() == '-' || peek() == '+') tok.push_back(next());
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) tok.push_back(next());
        if (tok.empty() || tok == "-" || tok == "+") fail("expected integer");
        try {
            return std::stoi(tok);
        } catch (...) {
            fail("integer out of range");
        }
    }
};

}  // namespace

std::map<std::string, int> parse_flat_int_object(const std::string & json) {
    JsonCursor c{json};
    c.expect('{');
    std::map<std::string, int> out;
    c.skip_ws();
    if (c.peek() == '}') { c.next(); return out; }

    while (true) {
        auto key = c.parse_string();
        c.expect(':');
        int value = c.parse_int();
        out.emplace(std::move(key), value);
        c.skip_ws();
        char nxt = c.next();
        if (nxt == ',') continue;
        if (nxt == '}') break;
        c.fail("expected ',' or '}'");
    }
    c.skip_ws();
    if (!c.eof()) c.fail("trailing content");
    return out;
}

// ---------------------------------------------------------------------------
// Config loader
// ---------------------------------------------------------------------------

namespace {

// Backbone (JEBFBackbone) hyper-parameters, written flat as `tifa.backbone.*`.
void fill_backbone(const GgufFile & f, BackboneConfig & b) {
    const std::string p = "tifa.backbone.";
    b.cls                = f.get_string_opt(p + "cls").value_or("");
    b.dim                = static_cast<int>(f.get_int(p + "dim"));
    b.num_layers         = static_cast<int>(f.get_int(p + "num_layers"));
    b.num_heads          = static_cast<int>(f.get_int(p + "num_heads"));
    b.head_dim           = static_cast<int>(f.get_int(p + "head_dim"));
    b.attn_type          = f.get_string_opt(p + "attn_type").value_or("joint");
    b.ffn_type           = f.get_string_opt(p + "ffn_type").value_or("glu");
    b.qk_norm            = f.get_bool_opt(p + "qk_norm").value_or(true);
    b.use_rope           = f.get_bool_opt(p + "use_rope").value_or(true);
    b.use_ls             = f.get_bool_opt(p + "use_ls").value_or(true);
    b.use_out_norm       = f.get_bool_opt(p + "use_out_norm").value_or(true);
    b.skip_first_ffn     = f.get_bool_opt(p + "skip_first_ffn").value_or(false);
    b.skip_out_ffn       = f.get_bool_opt(p + "skip_out_ffn").value_or(false);
    b.theta              = f.get_float_opt(p + "theta").value_or(10000.0f);
    b.c_kernel_size_token = static_cast<int>(f.get_int_opt(p + "c_kernel_size_token").value_or(7));
    b.m_kernel_size_token = static_cast<int>(f.get_int_opt(p + "m_kernel_size_token").value_or(5));
    b.c_kernel_size_x     = static_cast<int>(f.get_int_opt(p + "c_kernel_size_x").value_or(31));
    b.m_kernel_size_x     = static_cast<int>(f.get_int_opt(p + "m_kernel_size_x").value_or(31));
}

std::vector<std::string> json_string_list(const json::Value * v) {
    std::vector<std::string> out;
    if (!v || !v->is_array()) return out;
    for (const auto & item : v->arr) {
        if (item.is_string()) out.push_back(item.str);
    }
    return out;
}

// Parse `tifa.vocab.json` — the model's vocabulary.json plus the G2P pipeline
// config, packed by the converter into a single metadata string:
//   {"symbols": {"zh/a": 158, ...}, "global_symbols": [...],
//    "stop_symbols": [...], "merged_groups": [["AP","br"], ...], "g2p": {...}}
void fill_vocabulary(const GgufFile & f, TifaModelConfig & c) {
    const auto raw = f.get_string_opt("tifa.vocab.json");
    if (!raw.has_value() || raw->empty()) {
        throw GgufError("missing GGUF metadata 'tifa.vocab.json'");
    }
    const json::Value doc = json::parse(*raw);

    const json::Value * symbols = doc.find("symbols");
    if (!symbols || !symbols->is_object()) {
        throw GgufError("'tifa.vocab.json' has no 'symbols' object");
    }
    int max_id = -1;
    for (const auto & kv : symbols->obj) {
        if (!kv.second.is_number()) {
            throw GgufError("vocabulary entry '" + kv.first + "' is not a number");
        }
        const int id = static_cast<int>(kv.second.number);
        c.symbol_to_id[kv.first] = id;
        if (id > max_id) max_id = id;
    }
    c.id_to_symbol.assign(static_cast<std::size_t>(max_id) + 1, "<unused>");
    for (const auto & kv : symbols->obj) {
        c.id_to_symbol[static_cast<std::size_t>(c.symbol_to_id[kv.first])] = kv.first;
    }

    c.global_symbols = json_string_list(doc.find("global_symbols"));
    c.stop_symbols   = json_string_list(doc.find("stop_symbols"));
    if (const json::Value * groups = doc.find("merged_groups"); groups && groups->is_array()) {
        for (const auto & g : groups->arr) {
            c.merged_groups.push_back(json_string_list(&g));
        }
    }
    if (const json::Value * g2p = doc.find("g2p"); g2p && !g2p->is_null()) {
        c.g2p_json = json::dump(*g2p);
    }
}

}  // namespace

TifaModelConfig load_config(const GgufFile & f) {
    TifaModelConfig c{};
    c.architecture = f.get_string("general.architecture");
    if (c.architecture != "tifa-fa") {
        throw GgufError("unsupported GGUF architecture '" + c.architecture +
                        "' (expected 'tifa-fa')");
    }
    c.name    = f.get_string_opt("general.name").value_or("");
    c.version = f.get_string_opt("general.version").value_or("");

    c.arch          = f.get_string_opt("tifa.model.arch").value_or("ForcedAlignmentModel");
    c.max_vocab_size = static_cast<int>(f.get_int("tifa.model.max_vocab_size"));
    c.in_dim        = static_cast<int>(f.get_int("tifa.model.in_dim"));
    c.embedding_dim = static_cast<int>(f.get_int("tifa.model.embedding_dim"));
    c.out_dim       = static_cast<int>(f.get_int("tifa.model.out_dim"));

    fill_backbone(f, c.backbone);

    auto & feat = c.features;
    feat.audio_sample_rate = static_cast<int>(f.get_int("tifa.features.audio_sample_rate"));
    feat.hop_size          = static_cast<int>(f.get_int("tifa.features.hop_size"));
    feat.fft_size          = static_cast<int>(f.get_int("tifa.features.fft_size"));
    feat.win_size          = static_cast<int>(f.get_int_opt("tifa.features.win_size").value_or(feat.fft_size));
    feat.num_bins          = static_cast<int>(f.get_int("tifa.features.num_bins"));
    feat.fmin              = f.get_float("tifa.features.fmin");
    feat.fmax              = f.get_float("tifa.features.fmax");
    feat.clip_val          = f.get_float_opt("tifa.features.clip_val").value_or(1e-5f);

    c.skip_penalty = f.get_float_opt("tifa.inference.skip_penalty").value_or(0.5f);
    c.score_unit   = f.get_string_opt("tifa.inference.score_unit").value_or("levenshtein");

    fill_vocabulary(f, c);
    return c;
}

}  // namespace tifa_ggml::internal
