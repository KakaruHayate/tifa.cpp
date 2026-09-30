// Backend registry smoke test: whatever is compiled in must initialise (CPU
// at minimum) and report a name.

#include "backend.h"

#include "tifa_ggml/version.h"

#include <gtest/gtest.h>

TEST(Backend, BestBackendInitialises) {
    ggml_backend_t backend = tifa_ggml::internal::init_best_backend();
    ASSERT_NE(backend, nullptr);
    const char * name = tifa_ggml::internal::backend_name(backend);
    ASSERT_NE(name, nullptr);
    EXPECT_GT(std::string(name).size(), 0u);
    tifa_ggml::internal::free_backend(backend);
}

TEST(Backend, VersionStringIsPinnedTo0190) {
    // The library is built against ggml v0.19.0; a mismatch means the pin
    // drifted (see AGENT.md §1).
    EXPECT_STREQ(tifa_ggml::ggml_version_string(), "v0.19.0");
}
