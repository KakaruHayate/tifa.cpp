#include "audio_io.h"

#include "tifa_ggml/errors.h"

#include <cmath>
#include <cstddef>
#include <vector>

// ---------------------------------------------------------------------------
// Kaiser-windowed sinc resampler, polyphase table implementation.
//
// The direct form recomputes the Kaiser window (`bessel_i0`, `sqrt`) and a
// `sinc` for every tap of every output sample — ~27 M transcendental calls for
// a 9 s clip, which dominated the per-file latency.  The kernel only depends on
// the *fractional* position of the output sample, so it is tabulated once per
// (rate pair): NTAB sub-phases, each holding the `2*taps` coefficients of that
// phase.  The inner loop is then a plain dot product.
//
//   y[j] = sum_i x[i] * h[phase(j)](i - floor(t_j)),   t_j = j * in/out
//
// Weights are normalised per output sample so edge truncation does not
// attenuate the signal.
// ---------------------------------------------------------------------------

namespace {

// I0 Bessel function (series expansion; plenty accurate for the window).
double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double half = x * 0.5;
    for (int k = 1; k < 64; ++k) {
        term *= (half / k) * (half / k);
        sum += term;
        if (term < 1e-16 * sum) break;
    }
    return sum;
}

double sinc(double x) {
    if (std::fabs(x) < 1e-12) return 1.0;
    const double pi = 3.14159265358979323846;
    return std::sin(pi * x) / (pi * x);
}

}  // namespace

namespace tifa_ggml::internal {

std::vector<float> resample_to(const std::vector<float> & input,
                               int input_rate, int output_rate)
{
    if (input_rate <= 0 || output_rate <= 0) {
        throw InvalidArgument("resample_to: sample rates must be positive");
    }
    if (input_rate == output_rate || input.empty()) {
        return input;
    }

    const double ratio = static_cast<double>(input_rate) / static_cast<double>(output_rate);
    const int    taps  = 32;                 // kernel half-width, input samples
    const int    NTAB  = 512;                // sub-phase resolution
    const double beta  = 8.6;                // Kaiser beta (~ -90 dB sidelobes)
    const double rolloff = 0.95;             // keep a little margin below Nyquist

    double cutoff = rolloff;
    if (output_rate < input_rate) {
        cutoff *= static_cast<double>(output_rate) / static_cast<double>(input_rate);
    }

    // ---- polyphase table: table[p][k] for k in [0, 2*taps) ----------------
    // Phase p corresponds to a fractional offset  p / NTAB  of the output
    // sample inside the input grid.
    const double i0_beta = bessel_i0(beta);
    std::vector<double> table(static_cast<std::size_t>(NTAB) * 2 * taps, 0.0);
    for (int p = 0; p < NTAB; ++p) {
        const double frac = static_cast<double>(p) / NTAB;
        double * row = table.data() + static_cast<std::size_t>(p) * 2 * taps;
        for (int k = 0; k < 2 * taps; ++k) {
            // distance from the output sample to input sample (i0 + k), where
            // i0 = floor(t) - taps + 1  =>  d = k - taps + 1 - frac
            const double d = static_cast<double>(k) - taps + 1.0 - frac;
            const double u = d / static_cast<double>(taps);
            if (std::fabs(u) >= 1.0) { row[k] = 0.0; continue; }
            const double w = bessel_i0(beta * std::sqrt(1.0 - u * u)) / i0_beta;
            row[k] = w * sinc(cutoff * d);
        }
    }

    const std::size_t n_in = input.size();
    const std::size_t n_out = static_cast<std::size_t>(
        std::floor(static_cast<double>(n_in) / ratio));
    std::vector<float> out(n_out, 0.0f);

    for (std::size_t j = 0; j < n_out; ++j) {
        const double t  = static_cast<double>(j) * ratio;
        const long   i0 = static_cast<long>(std::floor(t)) - taps + 1;

        const int    phase = static_cast<int>(std::floor((t - std::floor(t)) * NTAB));
        const double * h = table.data() +
            static_cast<std::size_t>(phase < NTAB ? phase : NTAB - 1) * 2 * taps;

        // interior: full dot product
        if (i0 >= 0 && static_cast<std::size_t>(i0 + 2 * taps) <= n_in) {
            const float * x = input.data() + i0;
            double acc = 0.0;
            for (int k = 0; k < 2 * taps; ++k) acc += h[k] * static_cast<double>(x[k]);
            out[j] = static_cast<float>(acc);
            continue;
        }

        // edges: skip out-of-range taps and renormalise
        double acc = 0.0, wsum = 0.0;
        for (int k = 0; k < 2 * taps; ++k) {
            const long i = i0 + k;
            if (i < 0 || static_cast<std::size_t>(i) >= n_in) continue;
            acc  += h[k] * static_cast<double>(input[static_cast<std::size_t>(i)]);
            wsum += h[k];
        }
        out[j] = (std::fabs(wsum) > 1e-12) ? static_cast<float>(acc / wsum) : 0.0f;
    }
    return out;
}

}  // namespace tifa_ggml::internal
