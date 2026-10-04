// BreathLab high-level API — see breath.h.

#include "breath.h"

#include "breath_mel.h"
#include "breath_net.h"
#include "gguf_io.h"

#include "tifa_ggml/errors.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace tifa_ggml {

using internal::BreathFeatureConfig;
using internal::BreathFeatures;
using internal::BreathNet;

namespace {

// ---------------------------------------------------------------------------
// post-processing constants (mirror of <name>.meta.json:postprocess)
// ---------------------------------------------------------------------------

struct PostProcess {
    int   median_filter_frames = 5;
    float min_dur_ms           = 60.0f;
    float merge_gap_ms         = 80.0f;
    float sp_threshold         = 0.5f;
    float sp_floor_percentile  = 5.0f;
    float sp_floor_margin_db   = 8.0f;
    float sp_hf_guard_db       = 6.0f;
    float sp_min_dur_ms        = 120.0f;
};

// Python's round() is half-to-even; every constant in the shipped models lands
// far from .5, so plain rounding matches (and would only shift a boundary by
// one frame otherwise).
int round_frames(float ms, int fps) {
    return static_cast<int>(std::lround(static_cast<double>(ms) * 0.001 * fps));
}

// ---- runs of a 0/1 sequence as half-open [start, end) frame ranges ----

std::vector<std::pair<int, int>> runs_of(const std::vector<char> & binary) {
    std::vector<std::pair<int, int>> out;
    const int n = static_cast<int>(binary.size());
    int i = 0;
    while (i < n) {
        if (binary[static_cast<std::size_t>(i)]) {
            int j = i;
            while (j < n && binary[static_cast<std::size_t>(j)]) ++j;
            out.emplace_back(i, j);
            i = j;
        } else {
            ++i;
        }
    }
    return out;
}

// scipy.signal.medfilt(x, 5): median over a zero-padded window of 5.
std::vector<float> median_filter5(const std::vector<float> & x) {
    const int n = static_cast<int>(x.size());
    std::vector<float> out(static_cast<std::size_t>(n));
    float w[5];
    for (int i = 0; i < n; ++i) {
        for (int k = -2; k <= 2; ++k) {
            const int j = i + k;
            w[k + 2] = (j >= 0 && j < n) ? x[static_cast<std::size_t>(j)] : 0.0f;
        }
        std::sort(w, w + 5);
        out[static_cast<std::size_t>(i)] = w[2];
    }
    return out;
}

// numpy's `percentile(..., method="linear")`.
double percentile_linear(std::vector<float> v, double pct) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = (static_cast<double>(v.size()) - 1) * pct / 100.0;
    const double lo  = std::floor(idx);
    const double hi  = std::min<double>(lo + 1, static_cast<double>(v.size()) - 1);
    const double g   = idx - lo;
    return static_cast<double>(v[static_cast<std::size_t>(lo)]) +
           g * (static_cast<double>(v[static_cast<std::size_t>(hi)]) -
                static_cast<double>(v[static_cast<std::size_t>(lo)]));
}

// ---------------------------------------------------------------------------
// infer.py:prob_to_events
// ---------------------------------------------------------------------------

std::vector<std::pair<int, int>> prob_to_events(const std::vector<float> & prob,
                                                int fps, float threshold,
                                                const PostProcess & pp) {
    const int k = pp.median_filter_frames;
    std::vector<float> p = (k >= 3 && k % 2 == 1) ? median_filter5(prob) : prob;

    std::vector<char> bin(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) bin[i] = (p[i] >= threshold) ? 1 : 0;
    std::vector<std::pair<int, int>> runs = runs_of(bin);

    const int merge_gap = round_frames(pp.merge_gap_ms, fps);
    std::vector<std::pair<int, int>> merged;
    for (const auto & r : runs) {
        if (!merged.empty() && r.first - merged.back().second <= merge_gap) {
            merged.back().second = r.second;
        } else {
            merged.push_back(r);
        }
    }

    const int min_dur = round_frames(pp.min_dur_ms, fps);
    std::vector<std::pair<int, int>> out;
    for (const auto & r : merged) {
        if (r.second - r.first >= min_dur) out.push_back(r);
    }
    return out;
}

// ---------------------------------------------------------------------------
// infer.py:full_segmentation  (returns the AP runs plus the AP/SP/V classes)
// ---------------------------------------------------------------------------

std::vector<std::pair<int, int>> full_segmentation(const std::vector<float> & ap_prob,
                                                   const std::vector<float> & sp_prob,
                                                   const std::vector<float> & energy_db,
                                                   const std::vector<float> & hf_db,
                                                   int fps, float threshold,
                                                   const PostProcess & pp,
                                                   std::vector<char> & cls) {
    const int n = static_cast<int>(ap_prob.size());
    std::vector<std::pair<int, int>> ap_events = prob_to_events(ap_prob, fps, threshold, pp);

    cls.assign(static_cast<std::size_t>(n), 0);                 // 0 = V
    for (const auto & e : ap_events) {                          // 1 = AP
        for (int i = std::max(0, e.first); i < std::min(n, e.second); ++i) {
            cls[static_cast<std::size_t>(i)] = 1;
        }
    }

    std::vector<int> non_ap;
    for (int i = 0; i < n; ++i) {
        if (cls[static_cast<std::size_t>(i)] == 0) non_ap.push_back(i);
    }
    if (!non_ap.empty()) {
        if (!sp_prob.empty()) {
            for (int i : non_ap) {
                if (sp_prob[static_cast<std::size_t>(i)] >= pp.sp_threshold) {
                    cls[static_cast<std::size_t>(i)] = 2;
                }
            }
        } else {
            // energy rule with the high-band guard (no supervised SP head)
            std::vector<float> e, h;
            e.reserve(non_ap.size());
            h.reserve(non_ap.size());
            for (int i : non_ap) {
                e.push_back(energy_db[static_cast<std::size_t>(i)]);
                h.push_back(hf_db[static_cast<std::size_t>(i)]);
            }
            const double floor_e = percentile_linear(e, pp.sp_floor_percentile);
            const double floor_h = percentile_linear(h, pp.sp_floor_percentile);
            const float  lim_e   = static_cast<float>(floor_e + pp.sp_floor_margin_db);
            const float  lim_h   = static_cast<float>(floor_h + pp.sp_hf_guard_db);
            const bool   guard   = static_cast<int>(hf_db.size()) >= n;
            for (int i : non_ap) {
                bool sp = energy_db[static_cast<std::size_t>(i)] <= lim_e;
                if (guard) sp = sp && hf_db[static_cast<std::size_t>(i)] <= lim_h;
                if (sp) cls[static_cast<std::size_t>(i)] = 2;
            }
        }
    }

    // drop SP stretches shorter than sp_min_dur_ms
    const int sp_min = round_frames(pp.sp_min_dur_ms, fps);
    std::vector<char> is_sp(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) is_sp[static_cast<std::size_t>(i)] =
        cls[static_cast<std::size_t>(i)] == 2 ? 1 : 0;
    for (const auto & r : runs_of(is_sp)) {
        if (r.second - r.first < sp_min) {
            for (int i = r.first; i < r.second; ++i) cls[static_cast<std::size_t>(i)] = 0;
        }
    }
    return ap_events;
}

// ---- GGUF metadata helpers ----

int get_int(const internal::GgufFile & g, const std::string & key, int fallback) {
    const auto v = g.get_int_opt(key);
    return v ? static_cast<int>(*v) : fallback;
}

float get_float(const internal::GgufFile & g, const std::string & key, float fallback) {
    const auto v = g.get_float_opt(key);
    return v ? static_cast<float>(*v) : fallback;
}

// breath.mean / breath.std are GGUF arrays, which GgufFile does not expose.
std::vector<float> get_float_array(gguf_context * g, const std::string & key) {
    std::vector<float> out;
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return out;
    const int64_t n = gguf_get_arr_n(g, id);
    const void *  d = gguf_get_arr_data(g, id);
    if (!d || n <= 0) return out;
    const int type = gguf_get_arr_type(g, id);
    out.resize(static_cast<std::size_t>(n));
    if (type == GGUF_TYPE_FLOAT32) {
        std::copy_n(static_cast<const float *>(d), n, out.begin());
    } else if (type == GGUF_TYPE_FLOAT64) {
        const double * p = static_cast<const double *>(d);
        for (int64_t i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = static_cast<float>(p[i]);
    } else {
        out.clear();
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// impl
// ---------------------------------------------------------------------------

struct BreathModel::Impl {
    BreathNet           net;
    BreathFeatureConfig feat;
    PostProcess         pp;
    int                 fps        = 100;
    float               threshold  = 0.5f;
    int                 win_frames = 1200;
    int                 hop_frames = 600;
    int                 head_ap    = 0;
    int                 head_sp    = -1;

    // features + windowed inference with the overlap average
    void infer(const float * wav, std::size_t n, int sample_rate,
               BreathFeatures & feats, std::vector<float> & ap_prob,
               std::vector<float> & sp_prob) const;
};

void BreathModel::Impl::infer(const float * wav, std::size_t n, int sample_rate,
                              BreathFeatures & feats, std::vector<float> & ap_prob,
                              std::vector<float> & sp_prob) const {
    if (!wav || n == 0) throw InvalidArgument("breath: empty waveform");

    // 1. resample to the model rate (the reference `load_wav` does the same)
    std::vector<float> mono;
    if (sample_rate != feat.sample_rate) {
        mono = internal::breath_resample_poly(wav, n, sample_rate, feat.sample_rate);
    } else {
        mono.assign(wav, wav + n);
    }
    if (mono.empty()) throw InvalidArgument("breath: empty waveform after resampling");

    // 2. features
    feats = internal::breath_extract_features(mono.data(), mono.size(), feat);
    const int T = feats.T;

    // 3. sliding windows, 12 s / 6 s, averaged on the overlap
    const int win  = win_frames;
    const int hop  = hop_frames;
    std::vector<int> starts;
    if (T <= win) {
        starts.push_back(0);
    } else {
        for (int s = 0; s <= T - win; s += std::max(1, hop)) starts.push_back(s);
        if (starts.back() != T - win) starts.push_back(T - win);
    }

    const int n_out = net.config().n_out;
    std::vector<double> ap_acc(static_cast<std::size_t>(T), 0.0);
    std::vector<double> sp_acc(static_cast<std::size_t>(T), 0.0);
    std::vector<double> wsum(static_cast<std::size_t>(T), 0.0);
    for (int s : starts) {
        const int w = std::min(win, T - s);
        const std::vector<float> out =
            net.run(feats.rows.data() + static_cast<std::size_t>(s), T, w);
        // `out` is the raw ggml [n_out, T] buffer, i.e. ne0 (the head index)
        // is the innermost axis.
        for (int t = 0; t < w; ++t) {
            const std::size_t idx = static_cast<std::size_t>(s + t);
            ap_acc[idx] += out[static_cast<std::size_t>(head_ap) + n_out * t];
            if (head_sp >= 0 && head_sp < n_out) {
                sp_acc[idx] += out[static_cast<std::size_t>(head_sp) + n_out * t];
            }
            wsum[idx] += 1.0;
        }
    }
    ap_prob.resize(static_cast<std::size_t>(T));
    for (int t = 0; t < T; ++t) {
        const double den = std::max(wsum[static_cast<std::size_t>(t)], 1e-8);
        ap_prob[static_cast<std::size_t>(t)] =
            static_cast<float>(ap_acc[static_cast<std::size_t>(t)] / den);
    }
    sp_prob.clear();
    if (head_sp >= 0 && head_sp < n_out) {
        sp_prob.resize(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) {
            const double den = std::max(wsum[static_cast<std::size_t>(t)], 1e-8);
            sp_prob[static_cast<std::size_t>(t)] =
                static_cast<float>(sp_acc[static_cast<std::size_t>(t)] / den);
        }
    }
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

BreathModel::BreathModel() : impl_(std::make_unique<Impl>()) {}
BreathModel::~BreathModel() = default;
BreathModel::BreathModel(BreathModel &&) noexcept = default;
BreathModel & BreathModel::operator=(BreathModel &&) noexcept = default;

int BreathModel::sample_rate() const noexcept { return impl_->feat.sample_rate; }
int BreathModel::fps() const noexcept { return impl_->fps; }
float BreathModel::threshold() const noexcept { return impl_->threshold; }
const char * BreathModel::backend_name() const noexcept { return impl_->net.backend_name(); }

BreathModel BreathModel::load(const std::string & gguf_path) {
    BreathModel m;
    auto & impl = *m.impl_;

    internal::GgufFile gguf = internal::GgufFile::open(gguf_path);
    const std::string arch = gguf.get_string("general.architecture");
    if (arch != "breath-ap") {
        throw NotImplemented("breath: architecture '" + arch + "' is not supported");
    }

    BreathFeatureConfig & f = impl.feat;
    f.sample_rate      = get_int(gguf, "breath.audio.sample_rate", 24000);
    f.n_fft            = get_int(gguf, "breath.feature.n_fft", 600);
    f.win_length       = get_int(gguf, "breath.feature.win_length", f.n_fft);
    f.hop_length       = get_int(gguf, "breath.feature.hop_length", 240);
    f.n_mels           = get_int(gguf, "breath.feature.n_mels", 64);
    f.fmin             = get_float(gguf, "breath.feature.fmin", 20.0f);
    f.fmax             = get_float(gguf, "breath.feature.fmax", 12000.0f);
    f.log_offset       = get_float(gguf, "breath.feature.log_offset", 1e-6f);
    f.f0_channels      = get_int(gguf, "breath.feature.f0_channels", 2);
    f.input_rows       = get_int(gguf, "breath.feature.input_rows", 66);
    f.norm_mode        = gguf.get_string_opt("breath.feature.norm_mode").value_or("cmvn");
    f.pitch_frame_samples = get_int(gguf, "breath.feature.pitch_frame_samples", 400);
    f.pitch_fmin_hz    = get_float(gguf, "breath.feature.pitch_fmin_hz", 70.0f);
    f.pitch_fmax_hz    = get_float(gguf, "breath.feature.pitch_fmax_hz", 1000.0f);
    f.mean             = get_float_array(gguf.handle(), "breath.mean");
    f.std              = get_float_array(gguf.handle(), "breath.std");

    impl.fps       = get_int(gguf, "breath.fps", 100);
    impl.threshold = get_float(gguf, "breath.threshold", 0.5f);

    PostProcess & pp = impl.pp;
    pp.median_filter_frames = get_int(gguf, "breath.postprocess.median_filter_frames", 5);
    pp.min_dur_ms           = get_float(gguf, "breath.postprocess.min_dur_ms", 60.0f);
    pp.merge_gap_ms         = get_float(gguf, "breath.postprocess.merge_gap_ms", 80.0f);
    pp.sp_threshold         = get_float(gguf, "breath.postprocess.sp_threshold", 0.5f);
    pp.sp_floor_percentile  = get_float(gguf, "breath.postprocess.sp_floor_percentile", 5.0f);
    pp.sp_floor_margin_db   = get_float(gguf, "breath.postprocess.sp_floor_margin_db", 8.0f);
    pp.sp_hf_guard_db       = get_float(gguf, "breath.postprocess.sp_hf_guard_db", 6.0f);
    pp.sp_min_dur_ms        = get_float(gguf, "breath.postprocess.sp_min_dur_ms", 120.0f);

    impl.win_frames = static_cast<int>(std::lround(
        get_float(gguf, "breath.eval.infer_window_sec", 12.0f) * impl.fps));
    impl.hop_frames = static_cast<int>(std::lround(
        get_float(gguf, "breath.eval.infer_hop_sec", 6.0f) * impl.fps));
    if (impl.hop_frames <= 0) impl.hop_frames = std::max(1, impl.win_frames / 2);

    impl.head_ap = get_int(gguf, "breath.head.ap", 0);
    // the supervised SP head is optional: models without it fall back to the
    // energy + high-band rule in full_segmentation
    const auto sp = gguf.get_int_opt("breath.head.sp");
    impl.head_sp = sp ? static_cast<int>(*sp) : -1;

    impl.net = BreathNet::load(gguf_path);
    return m;
}

void BreathModel::probabilities(const float * wav, std::size_t n, int sample_rate,
                                std::vector<float> & ap_prob,
                                std::vector<float> & sp_prob) const {
    BreathFeatures feats;
    impl_->infer(wav, n, sample_rate, feats, ap_prob, sp_prob);
}

void BreathModel::run(const float * wav, std::size_t n, int sample_rate,
                      std::vector<BreathEvent> & ap_events,
                      std::vector<BreathSegment> & segments) const {
    const auto & impl = *impl_;

    BreathFeatures feats;
    std::vector<float> ap_prob, sp_prob;
    impl.infer(wav, n, sample_rate, feats, ap_prob, sp_prob);

    std::vector<char> cls;
    const auto runs = full_segmentation(ap_prob, sp_prob, feats.energy_db, feats.hf_db,
                                        impl.fps, impl.threshold, impl.pp, cls);

    const double inv_fps = 1.0 / impl.fps;
    ap_events.clear();
    ap_events.reserve(runs.size());
    for (const auto & r : runs) {
        ap_events.push_back(BreathEvent{static_cast<float>(r.first * inv_fps),
                                        static_cast<float>(r.second * inv_fps)});
    }

    segments.clear();
    const int nf = static_cast<int>(cls.size());
    int i = 0;
    while (i < nf) {
        int j = i;
        while (j < nf && cls[static_cast<std::size_t>(j)] == cls[static_cast<std::size_t>(i)]) ++j;
        const char * name = cls[static_cast<std::size_t>(i)] == 1 ? "AP"
                          : cls[static_cast<std::size_t>(i)] == 2 ? "SP" : "V";
        segments.push_back(BreathSegment{name, static_cast<float>(i * inv_fps),
                                         static_cast<float>(j * inv_fps)});
        i = j;
    }
}

}  // namespace tifa_ggml
