// tifa — command line front-end for the TIFA ggml inference engine.

#include "phone_input.h"

#include "align_decode.h"
#include "audio_io.h"
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
    std::printf("tifa %d.%d.%d\n", TIFA_GGML_VERSION_MAJOR,
                TIFA_GGML_VERSION_MINOR, TIFA_GGML_VERSION_PATCH);
}

void print_usage() {
    std::printf(R"(tifa — TIFA (Token-Imputing Forced Aligner) inference engine

Usage:
  tifa --version | --help
  tifa inspect <model.gguf>
  tifa align <audio-or-dir> -m <model.gguf> --phones ... [options]

The engine takes audio plus an already-known phone sequence; text-to-phoneme
conversion is deliberately out of scope (see docs/scope-change.md).  Provide
phones with --phones, --phones-file, --textgrid or --transcriptions-csv.

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
      --key ID                row identifier in the csv (default: file stem)
      --skip-handling MODE    discard | omit | preserve   (default: omit)
      --skip-penalty F        raw cosine cost per skipped phone (default: 0.5)
      --output-formats LIST   textgrid,json           (default: textgrid,json)
      --backend NAME          auto | cpu | vulkan | cuda | metal
      --fill-gaps LABEL       label for stretches no token covers
                              (default: SP; empty leaves them unlabelled)
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
    float       skip_penalty = 0.5f;
    long        max_frames = 6000;
    // Label for stretches no token covers.  A Praat tier has to run
    // continuously, and the dataset convention labels silence "SP" so no
    // stretch is left unlabelled; pass an empty value to keep them empty.
    std::string fill_gaps = "SP";
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


// Same reader, but a missing tier is not an error: the 2PASS carries optional
// tiers (texts/words) through and has to cope with TextGrids that lack them.
std::vector<tifa_cli::TimedInterval> read_interval_tier(const std::string & path,
                                                        const std::string & tier_name);

std::vector<tifa_cli::TimedInterval> read_interval_tier_opt(const std::string & path,
                                                    const std::string & tier_name,
                                                    bool & found) {
    try {
        std::vector<tifa_cli::TimedInterval> v = read_interval_tier(path, tier_name);
        found = true;
        return v;
    } catch (const std::exception &) {
        found = false;
        return {};
    }
}

// The grouping behind the texts/words tiers, re-attached to a fresh alignment.
//
// The second pass reads only the phones tier, so without this the phrase line
// is lost: nothing in a re-alignment can invent it back.  We read it from the
// first-pass TextGrid and map each token to the text/word that contained it,
// which is exactly what build_alignment_tiers() aggregates by -- so the tiers
// come out with the *new* phone boundaries but the original text.
struct InheritedGrouping {
    std::vector<std::int32_t> words;          // per token, 1-based text id
    std::vector<std::int32_t> groups;         // per token, 1-based word id
    std::vector<std::string>  word_texts;
    std::vector<std::string>  group_scripts;
    bool ok = false;
};

InheritedGrouping inherit_grouping(const std::string & tg_path,
                                   const std::string & phones_tier,
                                   const std::vector<std::string> & phones,
                                   const std::vector<std::string> & stop_symbols,
                                   std::size_t n_tokens) {
    InheritedGrouping out;
    bool have_texts = false, have_words = false;
    const auto texts_raw = read_interval_tier_opt(tg_path, "texts", have_texts);
    const auto words_raw = read_interval_tier_opt(tg_path, "words", have_words);
    if (!have_texts || !have_words || texts_raw.empty() || words_raw.empty()) return out;

    // The writer pads every tier with empty intervals so it runs continuously;
    // those are not words, and an owner pointing at one would lose the label.
    const auto keep_labelled = [](const std::vector<tifa_cli::TimedInterval> & v) {
        std::vector<tifa_cli::TimedInterval> out;
        for (const auto & iv : v) {
            if (!iv.text.empty()) out.push_back(iv);
        }
        return out;
    };
    const std::vector<tifa_cli::TimedInterval> texts = keep_labelled(texts_raw);
    const std::vector<tifa_cli::TimedInterval> words = keep_labelled(words_raw);
    if (texts.empty() || words.empty()) return out;

    bool have_phones = false;
    const auto raw = read_interval_tier_opt(tg_path, phones_tier, have_phones);
    if (!have_phones) return out;

    // index of the interval containing `t`, else the last one starting before it
    const auto owner_of = [](const std::vector<tifa_cli::TimedInterval> & v, double t) {
        std::int32_t best = 1;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (t >= v[i].xmin - 1e-9 && t <= v[i].xmax + 1e-9) {
                return static_cast<std::int32_t>(i) + 1;
            }
            if (v[i].xmin <= t) best = static_cast<std::int32_t>(i) + 1;
        }
        return best;
    };

    // walk the non-empty phones (read_textgrid_tier skips empty labels, so the
    // phone list and the raw intervals have to be filtered the same way)
    std::size_t k = 0;
    for (const auto & iv : raw) {
        if (iv.text.empty()) continue;
        if (k >= phones.size()) break;
        const double mid = 0.5 * (iv.xmin + iv.xmax);
        const std::int32_t w = owner_of(texts, mid);
        const std::int32_t g = owner_of(words, mid);
        const bool stop = std::find(stop_symbols.begin(), stop_symbols.end(),
                                    phones[k]) != stop_symbols.end();
        if (!stop) {
            out.words.push_back(w);
            out.groups.push_back(g);
        }
        ++k;
    }
    if (out.words.size() != n_tokens) return out;   // shapes disagree: don't guess

    out.word_texts.reserve(texts.size());
    for (const auto & t : texts) out.word_texts.push_back(t.text);
    out.group_scripts.reserve(words.size());
    for (const auto & w : words) out.group_scripts.push_back(w.text);
    out.ok = true;
    return out;
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

    const bool have_phone_source =
        !opt.phones.empty() || !opt.phones_file.empty() ||
        !opt.textgrid.empty() || !opt.csv.empty();
    if (!have_phone_source) {
        throw InvalidArgument("no phone source: pass --phones/--phones-file/"
                              "--textgrid/--transcriptions-csv (text/G2P is out "
                              "of scope; see docs/scope-change.md)");
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
            if (!inline_phones.phones.empty()) {
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
            AlignRequest req;
            req.phones       = seq.phones;
            req.groups       = seq.groups;
            req.language     = opt.language;
            req.skip_penalty = opt.skip_penalty;
            req.skip_handling = opt.skip_handling;
            result = model.align(audio_buf.samples.data(), audio_buf.samples.size(),
                                 audio_buf.sample_rate, req);

            // 2PASS: re-attach the phrase line.  The alignment above only saw
            // phones, so texts/words would come out empty; the first pass's
            // grouping is still in the TextGrid we just read.
            if (!opt.textgrid.empty() && !result.spans.empty()) {
                fs::path tg = opt.textgrid;
                if (fs::is_directory(tg)) tg /= (identifier + ".TextGrid");
                const InheritedGrouping inh = inherit_grouping(
                    tg.string(), opt.phones_tier, seq.phones,
                    model.config().stop_symbols, result.spans.size());
                if (inh.ok) {
                    result.words         = inh.words;
                    result.groups        = inh.groups;
                    result.word_texts    = inh.word_texts;
                    result.group_scripts = inh.group_scripts;
                }
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
            // The phones tier is last; every stretch the alignment does not
            // cover gets the fill label (SP by default) so the annotation has
            // no unlabelled holes.
            if (!tiers.empty()) tiers.back().gap_label = opt.fill_gaps;
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

// ---- 2PASS: read a first-pass TextGrid back --------------------------------

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
                else if (a == "--fill-gaps")                     opt.fill_gaps = value();
                else if (a == "--backend")                       opt.backend = value();
                else if (a == "--dump-dir")                      opt.dump_dir = value();
                else if (a == "--max-frames")                    opt.max_frames = std::stol(value());
                else if (a == "-q" || a == "--quiet")            opt.quiet = true;
                else throw InvalidArgument("unknown option: " + a);
            }
            return cmd_align(args[1], opt);
        }
        std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
        print_usage();
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
