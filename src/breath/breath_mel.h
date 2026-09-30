#pragma once

// BreathLab feature front end: log-mel + pitch/periodicity rows + CMVN + RMS
// energy.  Mirrors `logmel` / `pitch_channels` / `frame_rms_energy` in the
// reference `infer.py` bit-for-bit up to float32 rounding.
//
// The reference computes the FFT stages with numpy (float64) and only the
// windowed frames themselves in float32, so this front end keeps double
// precision through the FFTs and the filterbank and casts to float32 at the
// same points the reference does.  That is what makes the mel rows agree to
// ~1e-7 instead of ~1e-3.
//
// Not part of the public API.

#include <cstddef>
#include <string>
#include <vector>

namespace tifa_ggml::internal {

// Feature configuration (`feature` block of <name>.meta.json).
struct BreathFeatureConfig {
    int         sample_rate   = 24000;
    int         n_fft         = 600;
    int         win_length    = 600;
    int         hop_length    = 240;
    int         n_mels        = 64;
    float       fmin          = 20.0f;
    float       fmax          = 12000.0f;
    float       log_offset    = 1e-6f;
    int         f0_channels   = 2;
    int         input_rows    = 66;
    std::string norm_mode     = "cmvn";   // "cmvn" | "global"
    // pitch (autocorrelation) rows
    int         pitch_frame_samples = 400;   // independent of n_fft
    float       pitch_fmin_hz   = 70.0f;
    float       pitch_fmax_hz   = 1000.0f;
    float       pitch_per_thresh = 0.30f;
    float       pitch_lf_fmax_hz = 1050.0f;
    // "global" normalisation ruler (unused for cmvn models)
    std::vector<float> mean;
    std::vector<float> std;
};

// Everything the network consumes plus the two side rows post-processing needs.
struct BreathFeatures {
    std::vector<float> rows;        // [input_rows, T] = normalised mel | pitch
    std::vector<float> logmel;      // [n_mels, T] raw, before normalisation
    std::vector<float> pitch;       // [f0_channels, T] periodicity | log-F0
    std::vector<float> hf_db;       // [T] raw high-band dB (un-normalised mel)
    std::vector<float> energy_db;   // [T] frame RMS dB
    int T = 0;
};

// Windowed-sinc polyphase resampler, `scipy.signal.resample_poly(x, up, down)`
// with the default Kaiser(5.0) window.  The reference `load_wav` uses exactly
// this, so a C++ resampler that differs (e.g. the in-repo Kaiser/sinc one) puts
// a ~1e-4 relative difference into the audio and can move event boundaries.
std::vector<float> breath_resample_poly(const float * x, std::size_t n,
                                        int input_rate, int output_rate);

// Full feature extraction for one mono waveform (any sample rate).
BreathFeatures breath_extract_features(const float * wav, std::size_t n,
                                       const BreathFeatureConfig & cfg);

}  // namespace tifa_ggml::internal
