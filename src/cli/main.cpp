// tifa_ggml_cli — command line front-end for the TIFA ggml aligner.

#include "phone_input.h"

#include "align_decode.h"
#include "audio_io.h"
#include "breath/breath.h"
#include "g2p/g2p.h"
#include "g2p/pfml.h"
#include "g2p/select.h"
#include "json.h"
#include "model_impl.h"

#include "tifa_ggml/errors.h"
#include "tifa_ggml/model.h"
#include "tifa_ggml/textgrid.h"
#include "tifa_ggml/types.h"
#include "tifa_ggml/version.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <fcntl.h>
#  include <io.h>
#endif

namespace fs = std::filesystem;
using namespace tifa_ggml;

namespace {

void print_version() {
    std::printf("tifa_ggml_cli %d.%d.%d\n", TIFA_GGML_VERSION_MAJOR,
                TIFA_GGML_VERSION_MINOR, TIFA_GGML_VERSION_PATCH);
}

void print_usage() {
    std::printf(R"(tifa_ggml_cli — TIFA (Token-Imputing Forced Aligner) on ggml

Usage:
  tifa_ggml_cli --version | --help
  tifa_ggml_cli inspect <model.gguf>
  tifa_ggml_cli align <audio-or-dir> -m <model.gguf> [options]
  tifa_ggml_cli breathe <audio-or-dir> -m <breath.gguf> [options]

Breathe options:
  -m, --model PATH            breath/AP detector GGUF (required)
  -o, --output-dir DIR        output directory (default: input directory)
      --output-formats LIST   textgrid,json           (default: textgrid,json)
      --backend NAME          auto | cpu | vulkan | cuda | metal
      --merge PATH|DIR        fold the detected AP/SP into the phones tier of
                              the first-pass alignment (<name>.TextGrid) and
                              write the enriched TextGrid for the second pass
                              (dataset 2PASS workflow: align -> breathe --merge
                              -> align --textgrid)
      --phones-tier NAME      tier to merge into       (default: phones)
      --min-insert-ms MS      shortest AP/SP inserted (default: 50)
      --ep                    detect and annotate EP (exhale) events (BreathLab v6)
  -q, --quiet                 only report errors

Align options:
  -m, --model PATH            TIFA GGUF model (required)
  -o, --output-dir DIR        output directory (default: input directory)
  -l, --language LANG         default language; its prefix is dropped in labels
      --phones "AP zh e n"    inline phone sequence
      --phones-file PATH      whitespace-separated phone list
      --textgrid PATH|DIR     read the phones tier of a TextGrid (file, or a
                              directory scanned as <name>.TextGrid)
      --phones-tier NAME      TextGrid tier name (default: phones)
      --transcriptions-csv P  DiffSinger transcriptions.csv with ph_seq rows
      --text STR              inline transcript (G2P) for every input file
      --text-file PATH        transcript file (G2P); default: <name>.txt/.lab
      --dict-dir DIR          G2P dictionary root (default: <model dir>/dicts)
      --oov-handling MODE     raise | discard | force         (default: discard)
      --key ID                row identifier in the csv (default: file stem)
      --skip-handling MODE    discard | omit | preserve   (default: omit)
      --skip-penalty F        raw cosine cost per skipped phone (default: 0.5)
      --output-formats LIST   textgrid,json           (default: textgrid,json)
      --backend NAME          auto | cpu | vulkan | cuda | metal
      --max-frames N          refuse inputs longer than N mel frames
  -q, --quiet                 only report errors
)");
}

// ---------------------------------------------------------------------------
// Small argument helper
// ---------------------------------------------------------------------------

struct Args {
    std::vector<std::string> values;
    std::size_t i = 0;

    const char * next() {
        return i < values.size() ? values[i++].c_str() : nullptr;
    }
    bool has() const { return i < values.size(); }
};

std::string basename_no_ext(const fs::path & p) {
    return p.stem().string();
}

std::vector<fs::path> collect_audio(const fs::path & input) {
    std::vector<fs::path> files;
    if (fs::is_directory(input)) {
        for (const auto & entry : fs::recursive_directory_iterator(input)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".wav" || ext == ".flac" || ext == ".mp3") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(input);
    }
    return files;
}

struct AlignOptions {
    std::string model;
    std::string output_dir;
    std::string language;
    std::string phones;
    std::string phones_file;
    std::string textgrid;
    std::string phones_tier = "phones";
    std::string csv;
    std::string key;
    std::string skip_handling = "omit";
    // JSON on by default: the diagnosis file (confidence, monotonicity,
    // per-phone scores) is how the dataset workflow triages which clips
    // need proof-reading.  Pass `--output-formats textgrid` to opt out.
    std::string output_formats = "textgrid,json";
    std::string backend = "auto";
    std::string dump_dir;
    std::string text;             // inline transcript (every file)
    std::string text_file;        // transcript file
    std::string dict_dir;         // G2P dictionary root (default: model dir/dicts)
    std::string oov_handling = "discard";
    float       skip_penalty = 0.5f;
    long        max_frames = 6000;
    bool        quiet = false;
};

int cmd_inspect(const std::string & path) {
    Model model = Model::load(path);
    const auto & cfg = model.config();
    std::printf("architecture : %s\n", cfg.architecture.c_str());
    std::printf("name/version : %s / %s\n", cfg.name.c_str(), cfg.version.c_str());
    std::printf("arch         : %s\n", cfg.arch.c_str());
    std::printf("vocab size   : %d (symbols: %zu)\n", cfg.max_vocab_size, cfg.symbol_to_id.size());
    std::printf("features     : %d Hz, hop %d, fft %d, win %d, %d mels, %.0f–%.0f Hz\n",
                cfg.features.audio_sample_rate, cfg.features.hop_size,
                cfg.features.fft_size, cfg.features.win_size,
                cfg.features.num_bins, cfg.features.fmin, cfg.features.fmax);
    std::printf("timestep     : %.4f s\n", cfg.features.timestep());
    std::printf("backbone     : dim %d, %d layers, %d heads x %d, %s/%s, qk_norm %d, rope %d\n",
                cfg.backbone.dim, cfg.backbone.num_layers, cfg.backbone.num_heads,
                cfg.backbone.head_dim, cfg.backbone.attn_type.c_str(),
                cfg.backbone.ffn_type.c_str(), cfg.backbone.qk_norm ? 1 : 0,
                cfg.backbone.use_rope ? 1 : 0);
    std::printf("global syms  : ");
    for (const auto & s : cfg.global_symbols) std::printf("%s ", s.c_str());
    std::printf("\nstop syms    : ");
    for (const auto & s : cfg.stop_symbols) std::printf("%s ", s.c_str());
    std::printf("\n");
    return 0;
}

void write_diagnosis_json(const fs::path & path, const AlignResult & r,
                          const std::string & identifier)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) throw Error("cannot write " + path.string());
    out << "{\n";
    out << "  \"identifier\": \"" << identifier << "\",\n";
    out << "  \"num_frames\": " << r.num_frames << ",\n";
    out << "  \"num_tokens\": " << r.labels.size() << ",\n";
    out << "  \"agreement\": " << r.agreement << ",\n";
    out << "  \"confidence\": " << r.confidence << ",\n";
    out << "  \"determinacy\": " << r.determinacy << ",\n";
    out << "  \"monotonicity\": " << r.monotonicity << "\n";
    out << "}\n";
}

// ---- golden dumps (debug aid for scripts/compare_golden.py) ----------------

void write_f32(const fs::path & p, const std::vector<float> & v) {
    std::ofstream o(p, std::ios::binary);
    o.write(reinterpret_cast<const char *>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
}

void dump_forward(const fs::path & dir, const std::string & identifier,
                  const Model & model, const Model::Forward & fwd,
                  const std::vector<std::int32_t> & tokens,
                  const std::vector<std::pair<int, int>> & spans)
{
    fs::create_directories(dir);
    const auto & cfg = model.config();
    write_f32(dir / "mel.bin", fwd.mel);
    write_f32(dir / "frame_features.bin", fwd.frame_features);
    write_f32(dir / "token_features.bin", fwd.token_features);
    write_f32(dir / "frame_logits.bin", fwd.frame_logits);
    write_f32(dir / "token_logits.bin", fwd.token_logits);
    write_f32(dir / "similarity.bin", fwd.similarity);
    write_f32(dir / "tap0_ffn1_token.bin", fwd.tap_ffn1_token);
    write_f32(dir / "tap0_ffn1_x.bin", fwd.tap_ffn1_x);
    write_f32(dir / "tap0_attn_token.bin", fwd.tap_attn_token);
    write_f32(dir / "tap0_attn_x.bin", fwd.tap_attn_x);
    write_f32(dir / "tap0_pjac_token.bin", fwd.tap_pjac_token);
    write_f32(dir / "tap0_pjac_x.bin", fwd.tap_pjac_x);
    write_f32(dir / "tap0_ffn1_x_norm.bin", fwd.tap_ffn1_x_norm);
    write_f32(dir / "tap0_ffn1_x_glu.bin", fwd.tap_ffn1_x_glu);
    write_f32(dir / "tap0_q_token.bin", fwd.tap_q_token);
    write_f32(dir / "tap0_k_token.bin", fwd.tap_k_token);
    write_f32(dir / "tap0_v_token.bin", fwd.tap_v_token);
    write_f32(dir / "tap0_q_x.bin", fwd.tap_q_x);
    write_f32(dir / "tap0_k_x.bin", fwd.tap_k_x);
    write_f32(dir / "tap0_v_x.bin", fwd.tap_v_x);
    write_f32(dir / "tap0_qkv_token.bin", fwd.tap_qkv_token);
    write_f32(dir / "tap0_qkv_x.bin", fwd.tap_qkv_x);
    write_f32(dir / "tap0_attn_norm_token.bin", fwd.tap_attn_norm_token);
    write_f32(dir / "tap0_attn_norm_x.bin", fwd.tap_attn_norm_x);
    write_f32(dir / "tap0_q_pre.bin", fwd.tap_q_pre);
    write_f32(dir / "tap0_q_normed.bin", fwd.tap_q_normed);
    for (std::size_t i = 0; i < fwd.layer_x.size(); ++i) {
        write_f32(dir / ("layer" + std::to_string(i) + "_x.bin"), fwd.layer_x[i]);
        write_f32(dir / ("layer" + std::to_string(i) + "_token.bin"), fwd.layer_token[i]);
    }

    {
        std::ofstream o(dir / "spans.bin", std::ios::binary);
        for (const auto & s : spans) {
            const std::int32_t pair[2] = {s.first, s.second};
            o.write(reinterpret_cast<const char *>(pair), sizeof(pair));
        }
        o.close();
        std::ofstream t(dir / "tokens.bin", std::ios::binary);
        t.write(reinterpret_cast<const char *>(tokens.data()),
                static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
    }
    std::ofstream meta(dir / "meta.json", std::ios::binary);
    meta << "{\n";
    meta << "  \"identifier\": \"" << identifier << "\",\n";
    meta << "  \"num_frames\": " << fwd.T << ",\n";
    meta << "  \"num_tokens\": " << fwd.N << ",\n";
    meta << "  \"in_dim\": " << cfg.in_dim << ",\n";
    meta << "  \"out_dim\": " << cfg.out_dim << ",\n";
    meta << "  \"vocab_size\": " << cfg.max_vocab_size << ",\n";
    meta << "  \"dim\": " << cfg.backbone.dim << ",\n";
    meta << "  \"num_layers\": " << cfg.backbone.num_layers << ",\n";
    meta << "  \"layer_dumps\": " << fwd.layer_x.size() << "\n";
    meta << "}\n";
}


// ---- text (G2P) mode -------------------------------------------------------

// Resolve the transcript for one input: --text, --text-file, or a sidecar
// <name>.txt / <name>.lab next to the audio (TIFA's convention).
std::string resolve_transcript(const AlignOptions & opt, const fs::path & audio,
                               bool & found)
{
    found = false;
    if (!opt.text.empty()) { found = true; return opt.text; }
    if (!opt.text_file.empty()) {
        std::ifstream in(opt.text_file, std::ios::binary);
        if (!in) throw Error("cannot open transcript: " + opt.text_file);
        std::ostringstream buf; buf << in.rdbuf();
        found = true;
        return buf.str();
    }
    for (const char * ext : { ".txt", ".lab" }) {
        fs::path candidate = audio;
        candidate.replace_extension(ext);
        if (fs::exists(candidate) && !fs::is_directory(candidate)) {
            std::ifstream in(candidate, std::ios::binary);
            if (!in) continue;
            std::ostringstream buf; buf << in.rdbuf();
            found = true;
            return buf.str();
        }
    }
    return {};
}

std::string trim_text(std::string s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// G2P -> candidate grid -> first choices -> flat token sequence.
AlignTokenRequest build_text_request(const Model & model,
                                     tifa_ggml::internal::g2p::Pipeline & pipeline,
                                     const std::string & text,
                                     const std::string & language,
                                     const std::string & oov_handling)
{
    using namespace tifa_ggml::internal::g2p;
    const auto & cfg = model.config();

    std::vector<std::string> languages;
    if (!language.empty()) languages.push_back(language);

    // Upstream requires PFML for G2P input; a fragment uses direct phonemes /
    // language scopes, plain text goes through the converters unchanged.
    const std::vector<Word> words = looks_like_pfml(text)
        ? convert_pfml(pipeline, text, languages)
        : pipeline.convert(text, languages);
    const auto lookup = [&](const std::string & symbol) {
        return model.resolve_symbol(symbol, languages);
    };
    const CandidateGrid grid = encode_paths(words, lookup, cfg.global_symbols,
                                            cfg.stop_symbols, languages,
                                            oov_handling, 3);
    const std::vector<int> choices = first_choices(grid);
    const MaterializedPaths mat = materialize_paths(grid, choices);

    AlignTokenRequest req;
    req.tokens.resize(static_cast<std::size_t>(mat.count));
    req.labels.reserve(static_cast<std::size_t>(mat.count));
    req.words.resize(static_cast<std::size_t>(mat.count));
    req.groups.resize(static_cast<std::size_t>(mat.count));
    for (int i = 0; i < mat.count; ++i) {
        req.tokens[static_cast<std::size_t>(i)] = mat.tokens[static_cast<std::size_t>(i)];
        req.words[static_cast<std::size_t>(i)]  = mat.words[static_cast<std::size_t>(i)];
        req.groups[static_cast<std::size_t>(i)] = mat.groups[static_cast<std::size_t>(i)];
    }

    req.word_texts = grid.texts;
    std::vector<std::string> group_scripts;
    for (int w = 0; w < grid.W; ++w) {
        const int choice = choices[static_cast<std::size_t>(w)];
        if (choice <= 0) continue;
        const auto & cand = grid.lexicon[static_cast<std::size_t>(w)]
                                       [static_cast<std::size_t>(choice - 1)];
        for (const auto & ph : cand.phonemes) req.labels.push_back(ph);
        for (const auto & sc : cand.scripts)  group_scripts.push_back(sc);
    }
    if (req.labels.size() != req.tokens.size()) {
        throw Error("G2P grid is inconsistent (labels " +
                    std::to_string(req.labels.size()) + " vs tokens " +
                    std::to_string(req.tokens.size()) + ")");
    }
    req.group_scripts = group_scripts;
    req.language = language;
    return req;
}

int cmd_align(const std::string & input, const AlignOptions & opt) {
    if (opt.model.empty()) throw InvalidArgument("--model is required");
    const fs::path in_path(input);
    if (!fs::exists(in_path)) throw InvalidArgument("input not found: " + input);

    std::vector<fs::path> files = collect_audio(in_path);
    if (files.empty()) throw InvalidArgument("no audio files found in " + input);

    fs::path out_dir = opt.output_dir.empty()
        ? (fs::is_directory(in_path) ? in_path : in_path.parent_path())
        : fs::path(opt.output_dir);
    fs::create_directories(out_dir);

    const bool want_textgrid =
        opt.output_formats.find("textgrid") != std::string::npos;
    const bool want_json = opt.output_formats.find("json") != std::string::npos;

    // --backend routes through the library's env override (read at load time).
    if (!opt.backend.empty() && opt.backend != "auto") {
#if defined(_WIN32)
        _putenv_s("TIFA_GGML_BACKEND", opt.backend.c_str());
#else
        setenv("TIFA_GGML_BACKEND", opt.backend.c_str(), 1);
#endif
    }

    Model model = Model::load(opt.model);
    const SkipHandling skip = parse_skip_handling(opt.skip_handling);

    // ---- text (G2P) mode plumbing ----------------------------------------
    const bool have_phone_source =
        !opt.phones.empty() || !opt.phones_file.empty() ||
        !opt.textgrid.empty() || !opt.csv.empty();
    const bool text_mode = !have_phone_source;
    std::unique_ptr<tifa_ggml::internal::g2p::Pipeline> g2p;
    if (text_mode) {
        if (model.config().g2p_json.empty()) {
            throw InvalidArgument("the model carries no G2P configuration; use "
                                  "--phones/--textgrid/--transcriptions-csv");
        }
        std::string dict_dir = opt.dict_dir;
        if (dict_dir.empty()) {
            const fs::path model_dir = fs::path(opt.model).parent_path();
            dict_dir = fs::exists(model_dir / "dicts")
                ? (model_dir / "dicts").string()
                : model_dir.string();
        }
        g2p = std::make_unique<tifa_ggml::internal::g2p::Pipeline>(
            tifa_ggml::internal::g2p::Pipeline::from_config(
                model.config().g2p_json, dict_dir));
        if (!opt.quiet) {
            std::fprintf(stderr, "g2p: dictionaries from %s\n", dict_dir.c_str());
        }
    }

    // Optional shared phone source loaded up front.
    tifa_cli::PhoneSequence inline_phones;
    if (!opt.phones.empty())      inline_phones = tifa_cli::parse_phone_list(opt.phones);
    if (!opt.phones_file.empty()) inline_phones = tifa_cli::read_phones_file(opt.phones_file);

    int ok = 0, skipped = 0;
    for (const fs::path & audio : files) {
        const std::string identifier = basename_no_ext(audio);
        try {
            tifa_cli::PhoneSequence seq;
            if (text_mode) {
                // tokens come from G2P below; nothing to resolve here
            } else if (!inline_phones.phones.empty()) {
                seq = inline_phones;
            } else if (!opt.textgrid.empty()) {
                fs::path tg = opt.textgrid;
                if (fs::is_directory(tg)) tg /= (identifier + ".TextGrid");
                seq = tifa_cli::read_textgrid_tier(tg.string(), opt.phones_tier);
            } else if (!opt.csv.empty()) {
                const std::string key = opt.key.empty() ? identifier : opt.key;
                seq = tifa_cli::read_transcriptions_csv(opt.csv, key);
            } else {
                throw InvalidArgument("no phone source: pass --phones/--phones-file/"
                                      "--textgrid/--transcriptions-csv");
            }

            internal::AudioBuffer audio_buf = internal::load_audio_file(audio.string());

            const auto t0 = std::chrono::steady_clock::now();
            AlignResult result;
            if (text_mode) {
                bool found = false;
                const std::string text = trim_text(resolve_transcript(opt, audio, found));
                if (!found || text.empty()) {
                    throw InvalidArgument("no transcript: pass --text/--text-file "
                                          "or put <name>.txt beside the audio");
                }
                AlignTokenRequest req = build_text_request(model, *g2p, text,
                                                           opt.language,
                                                           opt.oov_handling);
                req.skip_penalty  = opt.skip_penalty;
                req.skip_handling = opt.skip_handling;
                result = model.align_tokens(audio_buf.samples.data(),
                                            audio_buf.samples.size(),
                                            audio_buf.sample_rate, req);
            } else {
                AlignRequest req;
                req.phones       = seq.phones;
                req.groups       = seq.groups;
                req.language     = opt.language;
                req.skip_penalty = opt.skip_penalty;
                result = model.align(audio_buf.samples.data(), audio_buf.samples.size(),
                                     audio_buf.sample_rate, req);
            }

            if (!opt.dump_dir.empty()) {
                const auto toks = model.internals().encode_phones(seq.phones, opt.language);
                Model::Forward fwd = model.forward(audio_buf.samples.data(),
                                                   audio_buf.samples.size(),
                                                   audio_buf.sample_rate, toks);
                std::vector<std::int32_t> groups(static_cast<std::size_t>(fwd.N));
                for (int i = 0; i < fwd.N; ++i) groups[static_cast<std::size_t>(i)] = i + 1;
                const auto spans = tifa_ggml::internal::decode_alignment_flat(
                    fwd.similarity.data(), fwd.T, fwd.N, groups.data(), opt.skip_penalty);
                dump_forward(fs::path(opt.dump_dir), identifier, model, fwd, toks, spans);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (opt.max_frames > 0 && result.num_frames > opt.max_frames) {
                throw InvalidArgument("input exceeds --max-frames");
            }

            std::vector<TextGridTier> tiers;
            double xmax = 0.0;
            const bool keep = build_alignment_tiers(
                result, skip,
                static_cast<double>(result.num_frames) * model.config().features.timestep(),
                tiers, xmax);
            if (!keep) {
                ++skipped;
                if (!opt.quiet) std::fprintf(stderr, "[skip] %s (zero-width spans)\n", identifier.c_str());
                continue;
            }
            if (want_textgrid) {
                write_textgrid_file((out_dir / (identifier + ".TextGrid")).string(), tiers, xmax);
            }
            if (want_json) {
                write_diagnosis_json(out_dir / (identifier + ".diagnosis.json"), result, identifier);
            }
            ++ok;
            if (!opt.quiet) {
                std::printf("%-40s %5d frames  %4zu phones  %7.1f ms  agreement %.3f\n",
                            identifier.c_str(), result.num_frames, result.labels.size(), ms,
                            result.agreement);
            }
        } catch (const std::exception & e) {
            ++skipped;
            std::fprintf(stderr, "[error] %s: %s\n", identifier.c_str(), e.what());
        }
    }
    std::fprintf(stderr, "done: %d ok, %d skipped/failed\n", ok, skipped);
    return skipped == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// breathe — BreathLab AP/SP detection (breath.h)
// ---------------------------------------------------------------------------

struct BreatheOptions {
    std::string model;
    std::string output_dir;
    // JSON on by default: the diagnosis file (confidence, monotonicity,
    // per-phone scores) is how the dataset workflow triages which clips
    // need proof-reading.  Pass `--output-formats textgrid` to opt out.
    std::string output_formats = "textgrid,json";
    std::string backend = "auto";
    // 2PASS support: fold the detected AP/SP segments into a phones tier
    // (an alignment produced by `align`), writing the enriched TextGrid that
    // the second `align --textgrid` pass consumes.
    std::string merge;              // alignment TextGrid file or directory
    std::string phones_tier = "phones";
    double      min_insert_ms = 50.0;
    bool        ep = false;
    bool        quiet = false;
};

// ---- 2PASS merge: fold AP/SP breath segments into a phones tier -----------

std::string unquote_textgrid_text(std::string s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        s = s.substr(1, s.size() - 2);
    }
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '"' && i + 1 < s.size() && s[i + 1] == '"') {  // "" -> "
            out.push_back('"');
            ++i;
            continue;
        }
        if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == '"') { // \" -> "
            out.push_back('"');
            ++i;
            continue;
        }
        out.push_back(s[i]);
    }
    return out;
}

// Minimal reader for the "long" ooTextFile format TIFA writes (and the Python
// `textgrid` package round-trips).  Returns the labelled intervals of the
// first tier called `tier_name`.
std::vector<tifa_cli::TimedInterval> read_interval_tier(const std::string & path,
                                              const std::string & tier_name) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw InvalidArgument("cannot open TextGrid '" + path + "'");

    std::vector<tifa_cli::TimedInterval> intervals;
    bool in_target = false;
    std::string line;
    while (std::getline(in, line)) {
        const auto trim = [](const std::string & s) {
            const std::size_t b = s.find_first_not_of(" \t\r");
            if (b == std::string::npos) return std::string();
            const std::size_t e = s.find_last_not_of(" \t\r");
            return s.substr(b, e - b + 1);
        };
        const std::string t = trim(line);
        if (t.rfind("item [", 0) == 0) {
            in_target = false;                       // a new tier begins
            continue;
        }
        if (t.rfind("class = ", 0) == 0 &&
            t.find("IntervalTier") != std::string::npos) {
            // this tier's name arrives on the next lines; mark pending
            in_target = false;
            continue;
        }
        if (t.rfind("name = ", 0) == 0) {
            const std::string name = unquote_textgrid_text(trim(t.substr(7)));
            in_target = name == tier_name;
            continue;
        }
        if (!in_target || t.rfind("xmin = ", 0) != 0) continue;
        tifa_cli::TimedInterval interval;
        interval.xmin = std::strtod(t.c_str() + 7, nullptr);
        if (!std::getline(in, line)) break;
        interval.xmax = std::strtod(trim(line).c_str() + 7, nullptr);
        if (!std::getline(in, line)) break;
        const std::string tl = trim(line);
        if (tl.rfind("text = ", 0) == 0) {
            interval.text = unquote_textgrid_text(trim(tl.substr(7)));
        }
        intervals.push_back(std::move(interval));
    }
    if (intervals.empty()) {
        throw InvalidArgument("TextGrid '" + path + "' has no intervals in tier '" +
                              tier_name + "'");
    }
    return intervals;
}

int cmd_breathe(const std::string & input, const BreatheOptions & opt) {
    if (opt.model.empty()) throw InvalidArgument("--model is required");
    const fs::path in_path(input);
    if (!fs::exists(in_path)) throw InvalidArgument("input not found: " + input);

    std::vector<fs::path> files = collect_audio(in_path);
    if (files.empty()) throw InvalidArgument("no audio files found in " + input);

    fs::path out_dir = opt.output_dir.empty()
        ? (fs::is_directory(in_path) ? in_path : in_path.parent_path())
        : fs::path(opt.output_dir);
    fs::create_directories(out_dir);

    const bool want_textgrid =
        opt.output_formats.find("textgrid") != std::string::npos;
    const bool want_json = opt.output_formats.find("json") != std::string::npos;

    if (!opt.backend.empty() && opt.backend != "auto") {
#if defined(_WIN32)
        _putenv_s("TIFA_GGML_BACKEND", opt.backend.c_str());
#else
        setenv("TIFA_GGML_BACKEND", opt.backend.c_str(), 1);
#endif
    }

    BreathModel model = BreathModel::load(opt.model);
    if (!opt.quiet) {
        std::fprintf(stderr, "breath: backend %s, model rate %d Hz, %.1f fps, threshold %.2f\n",
                     model.backend_name(), model.sample_rate(),
                     static_cast<double>(model.fps()), static_cast<double>(model.threshold()));
    }

    int ok = 0, skipped = 0;
    for (const fs::path & audio : files) {
        const std::string identifier = basename_no_ext(audio);
        try {
            internal::AudioBuffer audio_buf = internal::load_audio_file(audio.string());
            const double seconds =
                audio_buf.sample_rate > 0
                    ? static_cast<double>(audio_buf.samples.size()) / audio_buf.sample_rate
                    : 0.0;

            const auto t0 = std::chrono::steady_clock::now();
            std::vector<BreathEvent>   ap_events;
            std::vector<BreathEvent>   ep_events;
            std::vector<BreathSegment> segments;
            model.run(audio_buf.samples.data(), audio_buf.samples.size(),
                      audio_buf.sample_rate, ap_events, segments,
                      opt.ep ? &ep_events : nullptr, opt.ep);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (want_textgrid) {
                // One interval tier with the AP/SP/V timeline; unlabelled
                // stretches are impossible (the segments tile the file).
                TextGridTier breath_tier;
                breath_tier.name = "breath";
                breath_tier.intervals.reserve(segments.size());
                for (const BreathSegment & segment : segments) {
                    TextGridInterval interval;
                    interval.xmin = segment.start;
                    interval.xmax = segment.end;
                    interval.text = segment.label;
                    breath_tier.intervals.push_back(std::move(interval));
                }

                if (opt.merge.empty()) {
                    std::vector<TextGridTier> out_tiers;
                    out_tiers.push_back(std::move(breath_tier));
                    if (opt.ep && !ep_events.empty()) {
                        TextGridTier ep_tier;
                        ep_tier.name = "EP";
                        ep_tier.intervals.reserve(ep_events.size());
                        for (const BreathEvent & ev : ep_events) {
                            TextGridInterval interval;
                            interval.xmin = ev.start;
                            interval.xmax = ev.end;
                            interval.text = "EP";
                            ep_tier.intervals.push_back(std::move(interval));
                        }
                        out_tiers.push_back(std::move(ep_tier));
                    }
                    write_textgrid_file(
                        (out_dir / (identifier + ".breath.TextGrid")).string(),
                        std::move(out_tiers), seconds);
                } else {
                    // 2PASS step 2: fold the AP/SP/EP segments into the phones
                    // tier of the first-pass alignment and write the enriched
                    // TextGrid the second `align --textgrid` pass consumes.
                    fs::path alignment = opt.merge;
                    if (fs::is_directory(alignment)) {
                        alignment /= identifier + ".TextGrid";
                    }
                    const std::vector<tifa_cli::TimedInterval> phones =
                        read_interval_tier(alignment.string(), opt.phones_tier);
                    std::size_t inserted = 0;
                    const std::vector<tifa_cli::TimedInterval> merged =
                        merge_breath_into_phones(phones, segments,
                                                 opt.min_insert_ms / 1000.0,
                                                 &inserted);
                    TextGridTier merged_tier;
                    merged_tier.name = opt.phones_tier;
                    merged_tier.intervals.reserve(merged.size());
                    for (const tifa_cli::TimedInterval & interval : merged) {
                        TextGridInterval out;
                        out.xmin = interval.xmin;
                        out.xmax = interval.xmax;
                        out.text = interval.text;
                        merged_tier.intervals.push_back(std::move(out));
                    }
                    write_textgrid_file(
                        (out_dir / (identifier + ".TextGrid")).string(),
                        { std::move(merged_tier), std::move(breath_tier) }, seconds);
                    if (!opt.quiet) {
                        std::fprintf(stderr, "breath: merged %zu %s into %zu phones\n",
                                     inserted, opt.ep ? "AP/SP/EP" : "AP/SP", phones.size());
                    }
                }
            }
            if (want_json) {
                namespace bren_json = tifa_ggml::internal::json;
                using bren_json::Value;
                // The json builder has no factories; set type + payload by hand.
                const auto num = [](double v) {
                    Value x; x.type = Value::Type::Number; x.number = v; return x;
                };
                const auto text = [](const std::string & s) {
                    Value x; x.type = Value::Type::String; x.str = s; return x;
                };
                const auto obj = []() {
                    Value x; x.type = Value::Type::Object; return x;
                };
                const auto arr = []() {
                    Value x; x.type = Value::Type::Array; return x;
                };

                Value ap_list  = arr();
                Value ep_list  = arr();
                Value seg_list = arr();
                for (const BreathEvent & event : ap_events) {
                    Value item = obj();
                    item.obj.push_back({ "start", num(event.start) });
                    item.obj.push_back({ "end",   num(event.end) });
                    ap_list.arr.push_back(std::move(item));
                }
                if (opt.ep) {
                    for (const BreathEvent & event : ep_events) {
                        Value item = obj();
                        item.obj.push_back({ "start", num(event.start) });
                        item.obj.push_back({ "end",   num(event.end) });
                        ep_list.arr.push_back(std::move(item));
                    }
                }
                for (const BreathSegment & segment : segments) {
                    Value item = obj();
                    item.obj.push_back({ "label", text(segment.label) });
                    item.obj.push_back({ "start", num(segment.start) });
                    item.obj.push_back({ "end",   num(segment.end) });
                    seg_list.arr.push_back(std::move(item));
                }
                Value doc = obj();
                doc.obj.push_back({ "sample_rate", num(audio_buf.sample_rate) });
                doc.obj.push_back({ "duration",    num(seconds) });
                doc.obj.push_back({ "ap",          std::move(ap_list) });
                if (opt.ep) {
                    doc.obj.push_back({ "ep",      std::move(ep_list) });
                }
                doc.obj.push_back({ "segments",    std::move(seg_list) });
                std::ofstream out(out_dir / (identifier + ".breath.json"), std::ios::binary);
                out << bren_json::dump(doc, 2) << "\n";
            }

            ++ok;
            if (!opt.quiet) {
                if (opt.ep) {
                    std::printf("%-40s %6.1f s  %3zu AP  %3zu EP  %4zu segments  %7.1f ms\n",
                                identifier.c_str(), seconds, ap_events.size(), ep_events.size(), segments.size(), ms);
                } else {
                    std::printf("%-40s %6.1f s  %3zu AP events  %4zu segments  %7.1f ms\n",
                                identifier.c_str(), seconds, ap_events.size(), segments.size(), ms);
                }
            }
        } catch (const std::exception & e) {
            ++skipped;
            std::fprintf(stderr, "[error] %s: %s\n", identifier.c_str(), e.what());
        }
    }
    std::fprintf(stderr, "done: %d ok, %d skipped/failed\n", ok, skipped);
    return skipped == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
#if defined(_WIN32)
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) { print_usage(); return 0; }

    const std::string cmd = args[0];
    try {
        if (cmd == "--version" || cmd == "-v") { print_version(); return 0; }
        if (cmd == "--help" || cmd == "-h" || cmd == "help") { print_usage(); return 0; }
        if (cmd == "inspect") {
            if (args.size() < 2) { print_usage(); return 1; }
            return cmd_inspect(args[1]);
        }
        if (cmd == "align") {
            if (args.size() < 2) { print_usage(); return 1; }
            AlignOptions opt;
            Args it; it.values.assign(args.begin() + 2, args.end());
            while (it.has()) {
                const std::string a = it.next();
                auto value = [&]() -> std::string {
                    const char * v = it.next();
                    if (!v) throw InvalidArgument("missing value after " + a);
                    return v;
                };
                if (a == "-m" || a == "--model")                 opt.model = value();
                else if (a == "-o" || a == "--output-dir")       opt.output_dir = value();
                else if (a == "-l" || a == "--language")         opt.language = value();
                else if (a == "--phones")                        opt.phones = value();
                else if (a == "--phones-file")                   opt.phones_file = value();
                else if (a == "--textgrid")                      opt.textgrid = value();
                else if (a == "--phones-tier")                   opt.phones_tier = value();
                else if (a == "--transcriptions-csv")            opt.csv = value();
                else if (a == "--key")                           opt.key = value();
                else if (a == "--skip-handling")                 opt.skip_handling = value();
                else if (a == "--skip-penalty")                  opt.skip_penalty = std::stof(value());
                else if (a == "--output-formats")                opt.output_formats = value();
                else if (a == "--backend")                       opt.backend = value();
                else if (a == "--dump-dir")                      opt.dump_dir = value();
                else if (a == "--text")                          opt.text = value();
                else if (a == "--text-file")                     opt.text_file = value();
                else if (a == "--dict-dir")                      opt.dict_dir = value();
                else if (a == "--oov-handling")                  opt.oov_handling = value();
                else if (a == "--max-frames")                    opt.max_frames = std::stol(value());
                else if (a == "-q" || a == "--quiet")            opt.quiet = true;
                else throw InvalidArgument("unknown option: " + a);
            }
            return cmd_align(args[1], opt);
        }
        if (cmd == "breathe") {
            if (args.size() < 2) { print_usage(); return 1; }
            BreatheOptions opt;
            Args it; it.values.assign(args.begin() + 2, args.end());
            while (it.has()) {
                const std::string a = it.next();
                auto value = [&]() -> std::string {
                    const char * v = it.next();
                    if (!v) throw InvalidArgument("missing value after " + a);
                    return v;
                };
                if (a == "-m" || a == "--model")           opt.model = value();
                else if (a == "-o" || a == "--output-dir") opt.output_dir = value();
                else if (a == "--output-formats")          opt.output_formats = value();
                else if (a == "--backend")                 opt.backend = value();
                else if (a == "--merge")                   opt.merge = value();
                else if (a == "--phones-tier")             opt.phones_tier = value();
                else if (a == "--min-insert-ms")           opt.min_insert_ms = std::strtod(value().c_str(), nullptr);
                else if (a == "--ep")                      opt.ep = true;
                else if (a == "-q" || a == "--quiet")      opt.quiet = true;
                else throw InvalidArgument("unknown option: " + a);
            }
            return cmd_breathe(args[1], opt);
        }
        std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
        print_usage();
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
