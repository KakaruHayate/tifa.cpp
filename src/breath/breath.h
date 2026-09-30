#pragma once

// BreathLab breath/AP detector — native ggml port of the ONNX model.
//
//   BreathModel model = BreathModel::load("breath-v5-24k.gguf");
//   std::vector<BreathEvent>   ap;
//   std::vector<BreathSegment> seg;
//   model.run(wav, n, sample_rate, ap, seg);
//
// The pipeline is the reference `infer.py` verbatim: resample to the model
// rate (scipy-compatible polyphase), log-mel + autocorrelation pitch rows,
// per-clip CMVN, 12 s / 6 s windowed inference with the overlaps averaged, then
// median-filter + threshold + merge for the AP events and the supervised SP
// head for the AP/SP/V timeline.
//
// `run` is const: the model holds no per-call state, so one loaded model can
// serve several threads (each call keeps its own buffers).

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tifa_ggml {

// One AP (audible breath) span, in seconds from the start of the waveform.
struct BreathEvent {
    float start = 0.0f;
    float end   = 0.0f;
};

// One contiguous stretch of the AP / SP / V timeline (`label` is "AP", "SP"
// or "V"); together the segments tile [0, n_samples / sample_rate].
struct BreathSegment {
    std::string label;
    float start = 0.0f;
    float end   = 0.0f;
};

class BreathModel {
public:
    static BreathModel load(const std::string & gguf_path);

    ~BreathModel();
    BreathModel(BreathModel &&) noexcept;
    BreathModel & operator=(BreathModel &&) noexcept;
    BreathModel(const BreathModel &) = delete;
    BreathModel & operator=(const BreathModel &) = delete;

    // AP events + the full AP/SP/V timeline for one mono waveform (any rate).
    void run(const float * wav, std::size_t n, int sample_rate,
             std::vector<BreathEvent> & ap_events,
             std::vector<BreathSegment> & segments) const;

    // ---- diagnostics -----------------------------------------------------

    // Window-averaged per-frame probabilities ([T] each; `sp` is empty when
    // the model has no SP head).
    void probabilities(const float * wav, std::size_t n, int sample_rate,
                       std::vector<float> & ap_prob,
                       std::vector<float> & sp_prob) const;

    int         sample_rate() const noexcept;   // rate the model runs at
    int         fps() const noexcept;           // frames per second
    float       threshold() const noexcept;     // AP threshold from the GGUF
    const char * backend_name() const noexcept;

private:
    BreathModel();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tifa_ggml
