#pragma once

// Version constants surfaced as preprocessor macros by the build system.
// Prefer `tifa_ggml::version_string()` for runtime display.

#ifndef TIFA_GGML_VERSION_MAJOR
#define TIFA_GGML_VERSION_MAJOR 0
#endif
#ifndef TIFA_GGML_VERSION_MINOR
#define TIFA_GGML_VERSION_MINOR 1
#endif
#ifndef TIFA_GGML_VERSION_PATCH
#define TIFA_GGML_VERSION_PATCH 5
#endif

namespace tifa_ggml {

// Returns the project version as "major.minor.patch".
const char * version_string() noexcept;

// Returns the ggml runtime version (e.g. "0.11.0") that the library was
// compiled against.  Useful for logging.
const char * ggml_version_string() noexcept;

}  // namespace tifa_ggml
