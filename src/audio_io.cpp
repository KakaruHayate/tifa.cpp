#include "audio_io.h"

#include "tifa_ggml/errors.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#define DR_MP3_IMPLEMENTATION
#define DR_FLAC_IMPLEMENTATION
#include <dr_flac.h>
#include <dr_mp3.h>
#include <dr_wav.h>

namespace tifa_ggml::internal {

namespace {

std::string lower_ext(const std::string & path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

// Average interleaved channels down to mono, converting to float.
void mixdown(const float * interleaved, unsigned channels,
             std::vector<float> & out)
{
    const std::size_t frames = out.size();
    if (channels <= 1) return;
    for (std::size_t i = 0; i < frames; ++i) {
        float acc = 0.0f;
        for (unsigned c = 0; c < channels; ++c) acc += interleaved[i * channels + c];
        out[i] = acc / static_cast<float>(channels);
    }
}

}  // namespace

AudioBuffer load_audio_file(const std::string & path) {
    const std::string ext = lower_ext(path);
    AudioBuffer buf;

    if (ext == ".mp3") {
        drmp3_config cfg{};
        drmp3_uint64 frames = 0;
        float * data = drmp3_open_file_and_read_pcm_frames_f32(path.c_str(), &cfg, &frames, nullptr);
        if (!data || frames == 0) {
            if (data) drmp3_free(data, nullptr);
            throw InvalidWav("failed to decode mp3: " + path);
        }
        buf.sample_rate = static_cast<int>(cfg.sampleRate);
        buf.samples.assign(data, data + frames * cfg.channels);
        buf.samples.resize(frames);
        mixdown(data, cfg.channels, buf.samples);
        drmp3_free(data, nullptr);
        return buf;
    }

    if (ext == ".flac") {
        unsigned int channels = 0, rate = 0;
        drflac_uint64 frames = 0;
        float * data = drflac_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &rate, &frames, nullptr);
        if (!data || frames == 0) {
            if (data) drflac_free(data, nullptr);
            throw InvalidWav("failed to decode flac: " + path);
        }
        buf.sample_rate = static_cast<int>(rate);
        buf.samples.assign(data, data + frames * channels);
        buf.samples.resize(frames);
        mixdown(data, channels, buf.samples);
        drflac_free(data, nullptr);
        return buf;
    }

    // Default: WAV (also accepts anything libsndfile-less we do not know).
    drwav wav{};
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        throw InvalidWav("failed to open audio file (expected wav/flac/mp3): " + path);
    }
    const drwav_uint64 frames = wav.totalPCMFrameCount;
    std::vector<float> interleaved(static_cast<std::size_t>(frames) * wav.channels);
    drwav_read_pcm_frames_f32(&wav, frames, interleaved.data());
    buf.sample_rate = static_cast<int>(wav.sampleRate);
    buf.samples.resize(static_cast<std::size_t>(frames));
    if (wav.channels <= 1) {
        std::copy(interleaved.begin(), interleaved.end(), buf.samples.begin());
    } else {
        mixdown(interleaved.data(), wav.channels, buf.samples);
    }
    drwav_uninit(&wav);
    if (frames == 0) throw InvalidWav("empty audio file: " + path);
    return buf;
}

}  // namespace tifa_ggml::internal
