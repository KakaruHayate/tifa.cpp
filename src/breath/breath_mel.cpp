// BreathLab feature front end — see breath_mel.h for the parity contract.

#include "breath_mel.h"

#include "tifa_ggml/errors.h"

#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace tifa_ggml::internal {

namespace {

constexpr double kPi = 3.14159265358979323846;

// ---- Slaney mel scale (infer.py:_hz_to_mel / _mel_to_hz) ----

constexpr double kFSp       = 200.0 / 3.0;      // linear-region slope (Hz per mel)
constexpr double kMinLogHz  = 1000.0;
constexpr double kMinLogMel = kMinLogHz / kFSp;
const double     kLogStep   = std::log(6.4) / 27.0;

double hz_to_mel(double hz) {
    if (hz >= kMinLogHz) return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
    return hz / kFSp;
}

double mel_to_hz(double mel) {
    if (mel >= kMinLogMel) return kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
    return kFSp * mel;
}

// Triangular Slaney filterbank [n_mels, n_fft/2+1], row-major, exactly as
// infer.py:mel_filterbank builds it (including the `2/(right-left)` enorm).
std::vector<double> make_mel_filterbank(int sr, int n_fft, int n_mels,
                                        double fmin, double fmax) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<double> fb(static_cast<std::size_t>(n_mels) * n_bins, 0.0);

    std::vector<double> all_freqs(n_bins);
    for (int k = 0; k < n_bins; ++k) {
        all_freqs[k] = static_cast<double>(sr) / 2.0 * k / (n_bins - 1);   // np.linspace
    }
    all_freqs[n_bins - 1] = sr / 2.0;                                      // endpoint exact

    const double mel_min = hz_to_mel(fmin);
    const double mel_max = hz_to_mel(fmax);
    std::vector<double> f_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double m = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        f_pts[i] = mel_to_hz(m);
    }
    f_pts[n_mels + 1] = mel_to_hz(mel_max);                                // np.linspace endpoint

    for (int m = 0; m < n_mels; ++m) {
        const double left = f_pts[m], center = f_pts[m + 1], right = f_pts[m + 2];
        double * row = fb.data() + static_cast<std::size_t>(m) * n_bins;
        for (int k = 0; k < n_bins; ++k) {
            const double f = all_freqs[k];
            if (center > left && f >= left && f <= center) {
                row[k] = (f - left) / (center - left);
            }
            if (right > center && f >= center && f <= right) {
                row[k] = (right - f) / (right - center);
            }
        }
        const double enorm = 2.0 / (right - left);
        for (int k = 0; k < n_bins; ++k) row[k] *= enorm;
    }
    return fb;
}

// ---- windows / padding ----

// torch.hann_window(periodic=True): 0.5 - 0.5*cos(2*pi*i/n)
std::vector<double> hann_periodic(int n) {
    std::vector<double> w(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / n);
    return w;
}

// np.hanning(n): the *symmetric* window (n-1 in the denominator).
std::vector<double> hann_symmetric(int n) {
    std::vector<double> w(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        w[static_cast<std::size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (n - 1));
    }
    return w;
}

// np.pad(mode="reflect"): [x[pad], ..., x[1], x..., x[n-2], ..., x[n-1-pad]]
std::vector<float> reflect_pad(const float * x, std::size_t n, int pad) {
    std::vector<float> out(n + 2 * static_cast<std::size_t>(pad));
    for (int i = 0; i < pad; ++i) out[static_cast<std::size_t>(i)] = x[pad - i];
    std::memcpy(out.data() + pad, x, n * sizeof(float));
    for (int i = 0; i < pad; ++i) {
        out[pad + n + i] = x[n - 2 - static_cast<std::size_t>(i)];
    }
    return out;
}

// ---- I0 (for the Kaiser window), series expansion ----

double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double half = x * 0.5;
    for (int k = 1; k < 64; ++k) {
        term *= (half / k) * (half / k);
        sum += term;
        if (term < 1e-17 * sum) break;
    }
    return sum;
}

// scipy.signal.get_window(('kaiser', beta), n, fftbins=False)
std::vector<double> kaiser_window(int n, double beta) {
    std::vector<double> w(static_cast<std::size_t>(n));
    const double i0b = bessel_i0(beta);
    const double alpha = (n - 1) / 2.0;
    for (int i = 0; i < n; ++i) {
        const double t = (i - alpha) / alpha;
        w[static_cast<std::size_t>(i)] = bessel_i0(beta * std::sqrt(1.0 - t * t)) / i0b;
    }
    return w;
}

// numpy's normalized sinc: sin(pi x) / (pi x)
double sinc(double x) {
    if (std::fabs(x) < 1e-12) return 1.0;
    return std::sin(kPi * x) / (kPi * x);
}

// scipy.signal.firwin(numtaps, cutoff, window=('kaiser', 5.0)) for a scalar
// low-pass cutoff: h = cutoff * sinc(cutoff*m) * win(m), normalised to unity DC.
std::vector<float> firwin_lowpass(int numtaps, double cutoff) {
    std::vector<double> w = kaiser_window(numtaps, 5.0);
    std::vector<double> h(static_cast<std::size_t>(numtaps));
    const double alpha = 0.5 * (numtaps - 1);
    double sum = 0.0;
    for (int i = 0; i < numtaps; ++i) {
        const double m = i - alpha;
        h[static_cast<std::size_t>(i)] = cutoff * sinc(cutoff * m) * w[static_cast<std::size_t>(i)];
        sum += h[static_cast<std::size_t>(i)];
    }
    std::vector<float> out(static_cast<std::size_t>(numtaps));
    for (int i = 0; i < numtaps; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<float>(h[static_cast<std::size_t>(i)] / sum);
    }
    return out;
}

std::int64_t gcd64(std::int64_t a, std::int64_t b) {
    while (b != 0) {
        const std::int64_t t = a % b;
        a = b;
        b = t;
    }
    return a < 0 ? -a : a;
}

// upfirdn output length (scipy.signal._upfirdn._output_len)
std::int64_t upfirdn_out_len(std::int64_t len_h, std::int64_t n_in,
                             std::int64_t up, std::int64_t down) {
    const std::int64_t num = (n_in - 1) * up + len_h;
    return (num + down - 1) / down;
}

}  // namespace

// ---------------------------------------------------------------------------
// Polyphase resampler == scipy.signal.resample_poly(x, up, down)
// ---------------------------------------------------------------------------

std::vector<float> breath_resample_poly(const float * x, std::size_t n,
                                        int input_rate, int output_rate) {
    if (!x || n == 0) return {};
    if (input_rate <= 0 || output_rate <= 0) {
        throw InvalidArgument("breath_resample_poly: sample rates must be positive");
    }
    if (input_rate == output_rate) return std::vector<float>(x, x + n);

    // scipy reduces up/down by their gcd and returns a copy when both are 1.
    std::int64_t up = output_rate, down = input_rate;
    const std::int64_t g = gcd64(up, down);
    up /= g;
    down /= g;
    if (up == 1 && down == 1) return std::vector<float>(x, x + n);

    const std::int64_t n_in = static_cast<std::int64_t>(n);
    const std::int64_t n_out = n_in * up / down + ((n_in * up) % down != 0 ? 1 : 0);

    // filter design: half_len = 10*max(up,down), cutoff 1/max_rate of Nyquist
    const std::int64_t max_rate = std::max(up, down);
    const double       f_c      = 1.0 / static_cast<double>(max_rate);
    const std::int64_t half_len = 10 * max_rate;
    std::vector<float> h = firwin_lowpass(static_cast<int>(2 * half_len + 1), f_c);
    for (float & v : h) v *= static_cast<float>(up);          // h *= up (float32)

    // zero-pad so the output samples land at the centre of the filter
    const std::int64_t n_pre_pad  = down - half_len % down;
    std::int64_t       n_post_pad = 0;
    const std::int64_t n_pre_remove = (half_len + n_pre_pad) / down;
    while (upfirdn_out_len(static_cast<std::int64_t>(h.size()) + n_pre_pad + n_post_pad,
                           n_in, up, down) < n_out + n_pre_remove) {
        ++n_post_pad;
    }
    std::vector<float> hpad(static_cast<std::size_t>(n_pre_pad + h.size() + n_post_pad), 0.0f);
    std::memcpy(hpad.data() + n_pre_pad, h.data(), h.size() * sizeof(float));

    // upfirdn(mode="constant", cval=0) == downsample(full_conv(upsample(x), h)):
    //   y[o] = sum_j hpad[j] * x[(o*down - j)/up]   for (o*down-j) >= 0 and % up == 0
    // evaluated polyphase-wise: for a fixed o only every `up`-th tap is used.
    const std::int64_t L     = static_cast<std::int64_t>(hpad.size());
    const std::int64_t avail = upfirdn_out_len(L, n_in, up, down);
    std::vector<double> acc(static_cast<std::size_t>(avail), 0.0);
    for (std::int64_t o = 0; o < avail; ++o) {
        const std::int64_t base = o * down;
        const std::int64_t r    = base % up;              // first tap of this phase
        std::int64_t       i    = (base - r) / up;
        double             sum  = 0.0;
        for (std::int64_t j = r; j < L && i >= 0; j += up, --i) {
            if (i < n_in) sum += static_cast<double>(hpad[static_cast<std::size_t>(j)]) *
                                 static_cast<double>(x[i]);
        }
        acc[static_cast<std::size_t>(o)] = sum;
    }

    std::vector<float> out(static_cast<std::size_t>(n_out));
    for (std::int64_t i = 0; i < n_out; ++i) {
        out[static_cast<std::size_t>(i)] =
            static_cast<float>(acc[static_cast<std::size_t>(n_pre_remove + i)]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// log-mel (infer.py:logmel)
// ---------------------------------------------------------------------------

// [n_mels, T] float32, computed in double through the FFT + filterbank.
std::vector<float> breath_logmel(const float * y, std::size_t n,
                                 const BreathFeatureConfig & cfg, int * T_out) {
    const int n_fft = cfg.n_fft, win = cfg.win_length, hop = cfg.hop_length;
    const int n_mels = cfg.n_mels, n_bins = n_fft / 2 + 1;

    std::vector<float> src(y, y + n);
    if (static_cast<int>(src.size()) < n_fft) src.resize(static_cast<std::size_t>(n_fft), 0.0f);

    const int pad = n_fft / 2;
    const std::vector<float> yp = reflect_pad(src.data(), src.size(), pad);
    const int T = static_cast<int>(1 + (yp.size() - static_cast<std::size_t>(n_fft)) / hop);
    *T_out = T;
    if (T <= 0) return {};

    std::vector<double> window = hann_periodic(win);
    if (win < n_fft) window.resize(static_cast<std::size_t>(n_fft), 0.0);
    std::vector<float> window_f(window.size());
    for (std::size_t i = 0; i < window.size(); ++i) window_f[i] = static_cast<float>(window[i]);

    // windowed frames [n_fft, T], float32 like the reference, then r2c in double
    std::vector<float> frames(static_cast<std::size_t>(n_fft) * T);
    for (int t = 0; t < T; ++t) {
        const float * s = yp.data() + static_cast<std::size_t>(t) * hop;
        float * d = frames.data() + static_cast<std::size_t>(t) * n_fft;
        for (int k = 0; k < n_fft; ++k) d[k] = s[k] * window_f[static_cast<std::size_t>(k)];
    }

    std::vector<std::complex<double>> spec(static_cast<std::size_t>(n_bins) * T);
    pocketfft::shape_t  sh  = {static_cast<std::size_t>(n_fft), static_cast<std::size_t>(T)};
    // strides are in *bytes* of the double frame buffer, not of the float one
    pocketfft::stride_t si  = {static_cast<std::ptrdiff_t>(sizeof(double)),
                               static_cast<std::ptrdiff_t>(sizeof(double) * n_fft)};
    pocketfft::stride_t so  = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>)),
                               static_cast<std::ptrdiff_t>(sizeof(std::complex<double>) * n_bins)};
    std::vector<double> frames_d(frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i) frames_d[i] = frames[i];
    pocketfft::r2c(sh, si, so, pocketfft::shape_t{0}, pocketfft::FORWARD,
                   frames_d.data(), spec.data(), 1.0);

    const std::vector<double> fb = make_mel_filterbank(cfg.sample_rate, n_fft, n_mels,
                                                       cfg.fmin, cfg.fmax);
    std::vector<float> out(static_cast<std::size_t>(n_mels) * T);
    for (int t = 0; t < T; ++t) {
        const std::complex<double> * sp = spec.data() + static_cast<std::size_t>(t) * n_bins;
        for (int m = 0; m < n_mels; ++m) {
            const double * row = fb.data() + static_cast<std::size_t>(m) * n_bins;
            double acc = 0.0;
            for (int k = 0; k < n_bins; ++k) {
                const double re = sp[k].real(), im = sp[k].imag();
                acc += row[k] * (re * re + im * im);      // power spectrum
            }
            out[static_cast<std::size_t>(m) * T + t] =
                static_cast<float>(std::log(acc + static_cast<double>(cfg.log_offset)));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// autocorrelation pitch / periodicity rows (infer.py:pitch_channels)
// ---------------------------------------------------------------------------

// [2, T] float32; returns also the frame count actually produced.
std::vector<float> breath_pitch_channels(const float * y, std::size_t n,
                                         const BreathFeatureConfig & cfg, int * T_out) {
    const int frame = cfg.pitch_frame_samples;
    const int hop   = cfg.hop_length;
    const int sr    = cfg.sample_rate;

    std::vector<float> src(y, y + n);
    if (static_cast<int>(src.size()) < frame) src.resize(static_cast<std::size_t>(frame), 0.0f);

    const std::vector<float> yp = reflect_pad(src.data(), src.size(), frame / 2);
    const int T = static_cast<int>(1 + src.size() / hop);
    *T_out = T;
    if (T <= 0) return {};

    const std::vector<double> win = hann_symmetric(frame);
    std::vector<float> window_f(win.size());
    for (std::size_t i = 0; i < win.size(); ++i) window_f[i] = static_cast<float>(win[i]);

    std::vector<double> frames(static_cast<std::size_t>(frame) * T);
    for (int t = 0; t < T; ++t) {
        const float * s = yp.data() + static_cast<std::size_t>(t) * hop;
        double * d = frames.data() + static_cast<std::size_t>(t) * frame;
        for (int k = 0; k < frame; ++k) {
            d[k] = static_cast<double>(s[k] * window_f[static_cast<std::size_t>(k)]);
        }
    }

    const int n_freq = frame / 2 + 1;
    std::vector<std::complex<double>> spec(static_cast<std::size_t>(n_freq) * T);
    pocketfft::shape_t  sh  = {static_cast<std::size_t>(frame), static_cast<std::size_t>(T)};
    pocketfft::stride_t si  = {static_cast<std::ptrdiff_t>(sizeof(double)),
                               static_cast<std::ptrdiff_t>(sizeof(double) * frame)};
    pocketfft::stride_t so  = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>)),
                               static_cast<std::ptrdiff_t>(sizeof(std::complex<double>) * n_freq)};
    pocketfft::r2c(sh, si, so, pocketfft::shape_t{0}, pocketfft::FORWARD,
                   frames.data(), spec.data(), 1.0);
    for (auto & c : spec) c = std::complex<double>(c.real() * c.real() + c.imag() * c.imag(), 0.0);

    // irfft of the power spectrum == the circular autocorrelation.  `fct` must
    // carry the 1/N that numpy's irfft applies: the reference normalises with
    // `ac / (ac[0] + 1e-9)`, so an unnormalised ac would shrink that epsilon
    // (and the lag it lets win) by a factor of N.
    std::vector<double> ac(static_cast<std::size_t>(frame) * T);
    pocketfft::stride_t sci = {static_cast<std::ptrdiff_t>(sizeof(std::complex<double>)),
                               static_cast<std::ptrdiff_t>(sizeof(std::complex<double>) * n_freq)};
    pocketfft::stride_t sco = {static_cast<std::ptrdiff_t>(sizeof(double)),
                               static_cast<std::ptrdiff_t>(sizeof(double) * frame)};
    pocketfft::c2r(sh, sci, sco, pocketfft::shape_t{0}, pocketfft::BACKWARD,
                   spec.data(), ac.data(), 1.0 / static_cast<double>(frame));

    // NOTE: the circular autocorrelation is exactly symmetric — ac[k] equals
    // ac[frame-k] for every k — so the reference's `seg.argmax` resolves those
    // lag pairs out of numpy's FFT rounding noise alone (mirroring the second
    // half here would make it deterministic but is not what the reference
    // does; the reference itself moves ~20% of the log-F0 row when the audio
    // changes by a single float32 ulp).  This port keeps the plain argmax.
    const int lo = std::max(1, sr / static_cast<int>(cfg.pitch_fmax_hz));
    const int hi = std::min(frame - 1, sr / static_cast<int>(cfg.pitch_fmin_hz));

    std::vector<double> per(static_cast<std::size_t>(T));
    std::vector<double> f0(static_cast<std::size_t>(T));
    for (int t = 0; t < T; ++t) {
        const double * a = ac.data() + static_cast<std::size_t>(t) * frame;
        const double inv = 1.0 / (a[0] + 1e-9);
        double best = -1e300;
        int    lag  = lo;
        for (int k = lo; k <= hi; ++k) {
            const double v = a[k] * inv;
            if (v > best) { best = v; lag = k; }     // np.argmax: first maximum wins
        }
        per[static_cast<std::size_t>(t)] = std::min(1.0, std::max(0.0, best));
        f0[static_cast<std::size_t>(t)]  = static_cast<double>(sr) / lag;
    }

    // per = convolve(per, ones(3)/3, "same")
    std::vector<double> sm(static_cast<std::size_t>(T));
    for (int t = 0; t < T; ++t) {
        const double a = (t + 1 < T) ? per[static_cast<std::size_t>(t + 1)] : 0.0;
        const double b = per[static_cast<std::size_t>(t)];
        const double c = (t - 1 >= 0) ? per[static_cast<std::size_t>(t - 1)] : 0.0;
        sm[static_cast<std::size_t>(t)] = (a + b + c) / 3.0;
    }

    const double lf_lo = std::log2(static_cast<double>(cfg.pitch_fmin_hz));
    const double lf_hi = std::log2(static_cast<double>(cfg.pitch_lf_fmax_hz));

    std::vector<float> out(static_cast<std::size_t>(2) * T);
    for (int t = 0; t < T; ++t) {
        const double p = sm[static_cast<std::size_t>(t)];
        double lf = 0.0;
        if (p >= cfg.pitch_per_thresh) {            // f0 > 0 <=> voiced
            // the reference computes this leg in float32 (np.log2 on a float32
            // array); the round trip costs one ulp at most
            const float f0f = static_cast<float>(f0[static_cast<std::size_t>(t)]);
            lf = std::min(1.0, std::max(0.0,
                (std::log2(static_cast<double>(f0f)) - lf_lo) / (lf_hi - lf_lo)));
        }
        out[static_cast<std::size_t>(t)]     = static_cast<float>(p);
        out[static_cast<std::size_t>(T) + t] = static_cast<float>(lf);
    }
    return out;
}

// ---------------------------------------------------------------------------
// frame RMS energy (infer.py:frame_rms_energy)
// ---------------------------------------------------------------------------

std::vector<float> breath_frame_energy(const float * y, std::size_t n,
                                       const BreathFeatureConfig & cfg) {
    const int frame = cfg.win_length, hop = cfg.hop_length;
    std::vector<float> src(y, y + n);
    if (static_cast<int>(src.size()) < frame) src.resize(static_cast<std::size_t>(frame), 0.0f);

    const std::vector<float> yp = reflect_pad(src.data(), src.size(), frame / 2);
    const int T = static_cast<int>(1 + src.size() / hop);
    std::vector<float> out(static_cast<std::size_t>(T));
    for (int t = 0; t < T; ++t) {
        const float * s = yp.data() + static_cast<std::size_t>(t) * hop;
        double acc = 0.0;
        for (int k = 0; k < frame; ++k) {
            acc += static_cast<double>(s[k]) * static_cast<double>(s[k]);
        }
        const float rms = static_cast<float>(std::sqrt(acc / frame + 1e-12));
        out[static_cast<std::size_t>(t)] =
            static_cast<float>(20.0 * std::log10(static_cast<double>(rms) + 1e-8));
    }
    return out;
}

// ---------------------------------------------------------------------------
// full feature extraction
// ---------------------------------------------------------------------------

BreathFeatures breath_extract_features(const float * wav, std::size_t n,
                                       const BreathFeatureConfig & cfg) {
    if (!wav || n == 0) throw InvalidArgument("breath: empty waveform");
    const auto & f = cfg;

    int T_mel = 0;
    std::vector<float> lm = breath_logmel(wav, n, f, &T_mel);
    if (T_mel <= 0) throw InvalidArgument("breath: audio too short for one frame");
    const int n_mels = f.n_mels;

    // high-band dB: mean of the upper third of the *raw* log-mel rows
    std::vector<float> hf(static_cast<std::size_t>(T_mel));
    for (int t = 0; t < T_mel; ++t) {
        double acc = 0.0;
        int    cnt = 0;
        for (int m = 2 * n_mels / 3; m < n_mels; ++m) {
            acc += lm[static_cast<std::size_t>(m) * T_mel + t];
            ++cnt;
        }
        hf[static_cast<std::size_t>(t)] = static_cast<float>(acc / cnt);
    }

    // normalisation, applied to the mel rows only
    std::vector<float> norm(static_cast<std::size_t>(n_mels) * T_mel);
    if (f.norm_mode == "cmvn") {
        for (int m = 0; m < n_mels; ++m) {
            const float * row = lm.data() + static_cast<std::size_t>(m) * T_mel;
            double mu = 0.0;
            for (int t = 0; t < T_mel; ++t) mu += row[t];
            mu /= T_mel;
            double var = 0.0;
            for (int t = 0; t < T_mel; ++t) {
                const double d = row[t] - mu;
                var += d * d;
            }
            const double sd = std::sqrt(var / T_mel) + 1e-5;
            float * dst = norm.data() + static_cast<std::size_t>(m) * T_mel;
            for (int t = 0; t < T_mel; ++t) {
                dst[t] = static_cast<float>((row[t] - mu) / sd);
            }
        }
    } else {
        if (f.mean.size() != static_cast<std::size_t>(n_mels) ||
            f.std.size()  != static_cast<std::size_t>(n_mels)) {
            throw InvalidArgument("breath: global norm_mode needs n_mels mean/std entries");
        }
        for (int m = 0; m < n_mels; ++m) {
            const float * row = lm.data() + static_cast<std::size_t>(m) * T_mel;
            float * dst = norm.data() + static_cast<std::size_t>(m) * T_mel;
            for (int t = 0; t < T_mel; ++t) dst[t] = (row[t] - f.mean[m]) / f.std[m];
        }
    }

    int T_pitch = 0;
    std::vector<float> pitch = breath_pitch_channels(wav, n, f, &T_pitch);
    std::vector<float> energy = breath_frame_energy(wav, n, f);

    const int T = std::min({T_mel, T_pitch, static_cast<int>(energy.size())});

    BreathFeatures out;
    out.T = T;
    out.logmel.assign(lm.begin(), lm.begin() + static_cast<std::size_t>(n_mels) * T);
    out.pitch.assign(pitch.begin(), pitch.begin() + static_cast<std::size_t>(2) * T);
    out.hf_db.assign(hf.begin(), hf.begin() + T);
    out.energy_db.assign(energy.begin(), energy.begin() + T);

    out.rows.assign(static_cast<std::size_t>(f.input_rows) * T, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        std::memcpy(out.rows.data() + static_cast<std::size_t>(m) * T,
                    norm.data() + static_cast<std::size_t>(m) * T_mel,
                    static_cast<std::size_t>(T) * sizeof(float));
    }
    for (int c = 0; c < f.f0_channels; ++c) {
        std::memcpy(out.rows.data() + static_cast<std::size_t>(n_mels + c) * T,
                    out.pitch.data() + static_cast<std::size_t>(c) * T,
                    static_cast<std::size_t>(T) * sizeof(float));
    }
    return out;
}

}  // namespace tifa_ggml::internal
