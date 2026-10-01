#include "backend.h"

#include "tifa_ggml/errors.h"
#include "tifa_ggml/version.h"

#include <ggml-backend.h>
#include <ggml.h>

#if defined(TIFA_GGML_HAS_METAL)
    #include <ggml-metal.h>
#endif
#if defined(TIFA_GGML_HAS_CUDA)
    #include <ggml-cuda.h>
#endif
#if defined(TIFA_GGML_HAS_VULKAN)
    #include <ggml-vulkan.h>
#endif
#include <ggml-cpu.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

// -----------------------------------------------------------------------------
// Public version helpers (declared in version.h)
// -----------------------------------------------------------------------------
namespace tifa_ggml {

namespace {
    // Synthesized once at static-init time so callers get a stable pointer.
    const std::string & g_version() {
        static const std::string v = [] {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d.%d.%d",
                TIFA_GGML_VERSION_MAJOR, TIFA_GGML_VERSION_MINOR, TIFA_GGML_VERSION_PATCH);
            return std::string(buf);
        }();
        return v;
    }
}

const char * version_string() noexcept {
    return g_version().c_str();
}

const char * ggml_version_string() noexcept {
    // Report the tag pinned in cmake/Dependencies.cmake (URL + URL_HASH) so
    // --version cannot silently drift.  TEMPORARY ANCHOR: v0.19.0.
    return "v0.19.0";
}

// -----------------------------------------------------------------------------
// Backend enumeration (declared in tifa_ggml.h)
// -----------------------------------------------------------------------------
namespace {
    std::array<const char *, 4> g_backend_names = {nullptr, nullptr, nullptr, nullptr};
    int g_backend_count = 0;

    void populate_backend_names() {
        if (g_backend_count != 0) return;
        // Preference order mirrors init_best_backend().
#if defined(TIFA_GGML_HAS_METAL)
        g_backend_names[g_backend_count++] = "metal";
#endif
#if defined(TIFA_GGML_HAS_CUDA)
        g_backend_names[g_backend_count++] = "cuda";
#endif
#if defined(TIFA_GGML_HAS_VULKAN)
        g_backend_names[g_backend_count++] = "vulkan";
#endif
        g_backend_names[g_backend_count++] = "cpu";
    }
}

const char * const * available_backends() noexcept {
    populate_backend_names();
    return g_backend_names.data();
}

int available_backends_count() noexcept {
    populate_backend_names();
    return g_backend_count;
}

}  // namespace tifa_ggml

// -----------------------------------------------------------------------------
// Internal: backend init helpers
// -----------------------------------------------------------------------------
namespace tifa_ggml::internal {

namespace {
// CPU threadpool registry: ggml v0.19 creates a *disposable* threadpool on
// every graph compute when none is attached (thread spawn per call).  We hold
// one persistent pool per CPU backend and free it together with the backend.
std::mutex              g_tp_mutex;
std::unordered_map<ggml_backend_t, ggml_threadpool_t> g_tp;

ggml_threadpool_t make_cpu_threadpool(int n_threads) {
    // Default params: hybrid polling (poll=50) keeps the worker threads warm
    // across the many small ops of this model's graphs without pure busy-wait.
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    return ggml_threadpool_new(&tpp);
}
}  // namespace

ggml_backend_t init_backend(Backend which) {
    switch (which) {
        case Backend::Metal:
#if defined(TIFA_GGML_HAS_METAL)
            // GPU backends may throw instead of returning nullptr when the
            // system has headers/loader compiled in but no usable runtime
            // (e.g. CI runners, VMs, machines without drivers).  Any failure
            // means "this backend is unavailable" so init_best_backend()
            // keeps its documented fall-through semantics.
            try {
                return ggml_backend_metal_init();
            } catch (const std::exception & e) {
                std::fprintf(stderr, "metal backend init failed: %s\n", e.what());
                return nullptr;
            } catch (...) {
                return nullptr;
            }
#else
            return nullptr;
#endif
        case Backend::CUDA:
#if defined(TIFA_GGML_HAS_CUDA)
            try {
                return ggml_backend_cuda_init(0);
            } catch (const std::exception & e) {
                std::fprintf(stderr, "cuda backend init failed: %s\n", e.what());
                return nullptr;
            } catch (...) {
                return nullptr;
            }
#else
            return nullptr;
#endif
        case Backend::Vulkan:
#if defined(TIFA_GGML_HAS_VULKAN)
            // ggml v0.19's Vulkan init propagates vk::SystemError
            // (createInstance: ErrorIncompatibleDriver) when the Vulkan
            // loader is present but no ICD claims the machine.  Catching this
            // only works because we compile with /EHs and not CMake's default
            // /EHsc -- see the note in CMakeLists.txt.
            try {
                return ggml_backend_vk_init(0);
            } catch (const std::exception & e) {
                std::fprintf(stderr, "vulkan backend init failed: %s\n", e.what());
                return nullptr;
            } catch (...) {
                return nullptr;
            }
#else
            return nullptr;
#endif
        case Backend::CPU: {
            ggml_backend_t b = ggml_backend_cpu_init();
            if (b) {
                // Default: use all hardware threads so CPU inference is not
                // accidentally single-threaded.  Override with
                // TIFA_GGML_THREADS=<n> (1..INT_MAX, full string must parse).
                unsigned n = std::thread::hardware_concurrency();
                if (n == 0) n = 1;
                if (const char * env = std::getenv("TIFA_GGML_THREADS"); env && *env) {
                    char * end = nullptr;
                    errno = 0;
                    const long v = std::strtol(env, &end, 10);
                    if (v >= 1 && v <= std::numeric_limits<int>::max() &&
                        end && *end == '\0') {
                        n = static_cast<unsigned>(v);
                    } else {
                        std::fprintf(stderr,
                            "warning: ignoring invalid TIFA_GGML_THREADS '%s' "
                            "(expected 1..%d)\n", env,
                            std::numeric_limits<int>::max());
                    }
                }
                ggml_backend_cpu_set_n_threads(b, static_cast<int>(n));
                ggml_threadpool_t tp = make_cpu_threadpool(static_cast<int>(n));
                if (tp) {
                    ggml_backend_cpu_set_threadpool(b, tp);
                    std::lock_guard<std::mutex> lock(g_tp_mutex);
                    g_tp[b] = tp;
                }
            }
            return b;
        }
    }
    return nullptr;
}

ggml_backend_t init_best_backend() {
    // TIFA_GGML_BACKEND=cpu|vulkan|cuda|metal|auto forces one backend
    // (debugging / reproducible CPU-vs-GPU comparisons).
    if (const char * env = std::getenv("TIFA_GGML_BACKEND"); env && *env) {
        const std::string want(env);
        if (want != "auto") {
            Backend which = Backend::CPU;
            if      (want == "cpu")    which = Backend::CPU;
            else if (want == "vulkan") which = Backend::Vulkan;
            else if (want == "cuda")   which = Backend::CUDA;
            else if (want == "metal")  which = Backend::Metal;
            else {
                std::fprintf(stderr, "warning: unknown TIFA_GGML_BACKEND '%s'\n", env);
            }
            if (auto * b = init_backend(which)) return b;
            throw BackendError(std::string("requested backend '") + want +
                               "' is unavailable in this build/system");
        }
    }
#if defined(TIFA_GGML_HAS_METAL)
    if (auto * b = init_backend(Backend::Metal)) return b;
#endif
#if defined(TIFA_GGML_HAS_CUDA)
    if (auto * b = init_backend(Backend::CUDA)) return b;
#endif
#if defined(TIFA_GGML_HAS_VULKAN)
    if (auto * b = init_backend(Backend::Vulkan)) return b;
#endif
    if (auto * b = init_backend(Backend::CPU)) return b;
    throw BackendError("failed to initialize any ggml backend (including CPU)");
}

void free_backend(ggml_backend_t backend) {
    if (backend == nullptr) return;
    {
        std::lock_guard<std::mutex> lock(g_tp_mutex);
        auto it = g_tp.find(backend);
        if (it != g_tp.end()) {
            ggml_threadpool_free(it->second);
            g_tp.erase(it);
        }
    }
    ggml_backend_free(backend);
}

const char * backend_name(ggml_backend_t backend) {
    if (!backend) return "<null>";
    return ggml_backend_name(backend);
}

}  // namespace tifa_ggml::internal
