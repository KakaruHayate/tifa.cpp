#pragma once

// Audio decoding + resampling helpers (internal).
//
// Decoding is dr_libs-based (WAV / FLAC / MP3) and always yields mono float32
// in [-1, 1] at the file's native sample rate.  The mel front-end wants the
// model's rate (48 kHz for TIFA-1.0), so callers resample with
// `resample_to()`.

#include <cstddef>
#include <string>
#include <vector>

namespace tifa_ggml::internal {

struct AudioBuffer {
    std::vector<float> samples;   // mono, [-1, 1]
    int                sample_rate = 0;
};

// Decode an audio file.  Throws `InvalidWav` (from tifa_ggml/errors.h) when
// the file cannot be read or its format is unsupported.
AudioBuffer load_audio_file(const std::string & path);

// Windowed-sinc (Kaiser) resampler, mono.  Quality target: comparable to
// librosa's `kaiser_best` / soxr `hq`; used when the input rate differs from
// the model rate.  No-op (copy) when the rates match.
std::vector<float> resample_to(const std::vector<float> & input,
                               int input_rate, int output_rate);

}  // namespace tifa_ggml::internal
