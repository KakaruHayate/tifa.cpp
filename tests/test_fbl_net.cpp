// Golden check for the FoxBreatheLabeler ggml port.
//
// The reference is the exported ONNX run by scripts/make_fbl_golden.py on a
// deterministic frame matrix, so this test catches a wrong op, a wrong layout
// or a wrong scale anywhere in the graph.  Skipped unless the converted GGUF is
// pointed at (TIFA_TEST_FBL_GGUF), since the model is a 74 MB download.

#include "breath/fbl_net.h"

#include "../src/json.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::string();
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

// The same deterministic waveform and framing as make_fbl_golden.py: the
// reference graph pads by 71 and unfolds with hop 882 itself, so the test has
// to do that step to feed the ggml net its [spec_win, T] input.
std::vector<float> golden_frames(int spec_win, int hop, int T) {
    const int n = T * hop;
    std::vector<float> w(static_cast<std::size_t>(n));
    for (int t = 0; t < n; ++t) {
        w[static_cast<std::size_t>(t)] =
            std::sin(0.017f * t) + 0.3f * std::cos(0.003f * t);
    }
    const int pad = (spec_win - hop) / 2;                       // 71
    std::vector<float> f(static_cast<std::size_t>(spec_win) * T, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < spec_win; ++k) {
            const int j = t * hop + k - pad;                    // into the waveform
            if (j >= 0 && j < n) {
                f[static_cast<std::size_t>(k) + static_cast<std::size_t>(spec_win) * t] = w[j];
            }
        }
    }
    return f;
}

}  // namespace

TEST(FblNet, MatchesReferenceApProbability) {
    const char * gguf = std::getenv("TIFA_TEST_FBL_GGUF");
    if (!gguf || !*gguf) {
        GTEST_SKIP() << "TIFA_TEST_FBL_GGUF not set";
    }
    const std::string text = read_file(std::string(GOLDEN_DIR) + "/fbl_ap.json");
    ASSERT_FALSE(text.empty()) << "golden missing";
    const tifa_ggml::internal::json::Value doc = tifa_ggml::internal::json::parse(text);

    int spec_win = 0;
    int T = 0;
    std::vector<float> want;
    for (const auto & kv : doc.obj) {
        if (kv.first == "spec_win") spec_win = static_cast<int>(kv.second.number);
        if (kv.first == "frames") T = static_cast<int>(kv.second.number);
        if (kv.first == "ap_prob") {
            for (const auto & v : kv.second.arr) want.push_back(static_cast<float>(v.number));
        }
    }
    ASSERT_GT(T, 0);
    ASSERT_EQ(want.size(), static_cast<std::size_t>(T));

    const tifa_ggml::internal::FblNet net =
        tifa_ggml::internal::FblNet::load(gguf);
    EXPECT_EQ(net.config().spec_win, spec_win);

    const std::vector<float> frames =
        golden_frames(spec_win, net.config().hop, T);
    const std::vector<float> got = net.run(frames.data(), T);
    ASSERT_EQ(got.size(), want.size());

    // F16 weights against an F32 reference: the port is only allowed to drift
    // by rounding, and the whole point of the test is that a layout or scale
    // mistake moves the numbers far more than this.
    double max_err = 0.0;
    double mean_err = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double e = std::fabs(static_cast<double>(got[i]) - want[i]);
        max_err = std::max(max_err, e);
        mean_err += e;
    }
    mean_err /= static_cast<double>(got.size());
    std::fprintf(stderr, "fbl: max_err=%.5f mean_err=%.5f (T=%d)\n",
                 max_err, mean_err, T);

    EXPECT_LT(max_err, 5e-3) << "ggml AP probability drifted from the reference";
    EXPECT_LT(mean_err, 1e-3);
}
