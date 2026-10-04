# ----------------------------------------------------------------------------
# External dependencies — all via FetchContent, nothing vendored.
# ----------------------------------------------------------------------------
include(FetchContent)

# Don't let child projects surprise us with their own versions.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)

# Propagate backend toggles as ggml's own option names *before* add_subdirectory.
# Note: GGML_LLAMAFILE deliberately does NOT use FORCE so an explicit
# -DGGML_LLAMAFILE=OFF survives; TIFA_GGML_LLAMAFILE is the public switch.
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_METAL           ${TIFA_GGML_METAL}  CACHE BOOL "ggml: enable Metal"  FORCE)
set(GGML_CUDA            ${TIFA_GGML_CUDA}   CACHE BOOL "ggml: enable CUDA"   FORCE)
set(GGML_VULKAN          ${TIFA_GGML_VULKAN} CACHE BOOL "ggml: enable Vulkan" FORCE)
set(GGML_LLAMAFILE       ${TIFA_GGML_LLAMAFILE} CACHE BOOL "ggml: llamafile sgemm kernels (CPU)")

# Apple cold-start optimisation: pre-compile a default.metallib instead of
# embedding source; paired with the binary-archive patch below this cuts ~7 s
# off first-run launch on macOS.
if(APPLE AND TIFA_GGML_METAL)
    set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "ggml: embed Metal library" FORCE)
endif()

# ---------------------------------------------------------------------------
# Helper: apply a git patch once; idempotent across re-configures.
# (Defined in ApplyPatch.cmake so non-ggml dependencies can reuse it in
# builds where this file is skipped entirely.)
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/ApplyPatch.cmake)

# ---------------------------------------------------------------------------
# SPIRV-Headers shim (Windows Vulkan only)
#
# ggml v0.20.x's Vulkan backend hard-requires find_package(SPIRV-Headers CONFIG).
# Windows Vulkan SDKs older than ~1.4.35x ship the headers but not that CMake
# config file, so the windows-x64-vulkan CI job fails at configure time.
# Generate a minimal config pointing at the SDK headers when the SDK doesn't
# already provide one; headers live under <SDK>/Include/spirv-headers.
# ---------------------------------------------------------------------------
if(TIFA_GGML_VULKAN AND WIN32 AND NOT EXISTS "$ENV{VULKAN_SDK}/Lib/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")
    string(REPLACE "\\" "/" _spirv_inc "$ENV{VULKAN_SDK}/Include")
    set(_spirv_cfg_dir "${CMAKE_BINARY_DIR}/SPIRV-Headers")
    file(MAKE_DIRECTORY "${_spirv_cfg_dir}")
    file(WRITE "${_spirv_cfg_dir}/SPIRV-HeadersConfig.cmake"
        "set(SPIRV-Headers_FOUND TRUE)\n"
        "if(NOT TARGET SPIRV-Headers::SPIRV-Headers)\n"
        "    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)\n"
        "    set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${_spirv_inc}\")\n"
        "endif()\n")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")
endif()

# ---------------------------------------------------------------------------
# ggml (MIT) — tensor engine.  Fetched + Metal binary-archive patch applied
# on Apple for fast cold start.
#
# > TEMPORARY ANCHOR — ggml v0.19.0 (see AGENT.md "ggml version gate").  Do
# > NOT bump to v0.20.x until backend.cpp's direct references
# > (ggml_backend_cpu_init, ggml_threadpool_new, ggml_backend_cpu_set_threadpool)
# > are moved behind a GGML_BACKEND_DL dlopen load: on v0.20.x CPU variants
# > require GGML_BACKEND_DL, DL unlinks the CPU backend from the umbrella
# > target (link errors), and NATIVE + DL are mutually exclusive upstream.
# > The CI linux-x64-cpu job is GGML_NATIVE=ON because of this anchor.
# ---------------------------------------------------------------------------
# URL archive (not GIT): with FETCHCONTENT_UPDATES_DISCONNECTED=ON the
# populate gitupdate step must resolve the pinned ref locally, and a fresh
# shallow clone cannot — a tag a few commits behind main is not on it, so
# populate aborts with "ref not present locally".  GPU CI jobs also restore
# a stale _deps cache whose ggml-populate dir has no .git (URL download),
# and a GIT pin fails there with "not a git repository: '.git'"; URL keeps
# those cached builds working.  Patches still apply via `git apply` --check.
FetchContent_Declare(
    ggml
    URL      https://github.com/ggerganov/ggml/archive/refs/tags/v0.19.0.tar.gz
    URL_HASH SHA256=cfb6512adda2853e6500a7c5b23f326987cb4c723e9f8f93c6c5a7e7e4861648
)

FetchContent_GetProperties(ggml)
if(NOT ggml_POPULATED)
    FetchContent_Populate(ggml)
endif()

# The patches live OUTSIDE the populate guard on purpose: a re-populate
# (stale download stamp, a restored _deps cache, a fresh checkout) silently
# replaces the patched sources, and a build that quietly loses the Vulkan
# pipeline cache looks identical until you measure cold-start time.  The
# helper is idempotent, so running it on every configure is free.
if(ggml_SOURCE_DIR)
    if(APPLE AND TIFA_GGML_METAL)
        tifa_ggml_apply_patch(
            "${ggml_SOURCE_DIR}"
            "${CMAKE_CURRENT_LIST_DIR}/patches/ggml-metal-binary-archive.patch"
            "ggml Metal MTLBinaryArchive PSO cache"
        )
    endif()

    # Vulkan cold-start fix: persist VkPipelineCache to disk.  Only touches
    # ggml-vulkan.cpp, so CPU-only and GPU builds can share one source tree.
    tifa_ggml_apply_patch(
        "${ggml_SOURCE_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/patches/ggml-vulkan-pipeline-cache.patch"
        "ggml Vulkan disk-backed VkPipelineCache"
    )

    # Fused batched LSTM sweep (English G2P; see cmake/patches/ggml-lstm-op.md).
    tifa_ggml_apply_patch(
        "${ggml_SOURCE_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/patches/ggml-lstm-op.patch"
        "ggml fused batched LSTM (GGML_OP_LSTM)"
    )

    # Fail loudly rather than shipping a "fast cold start" that is not there.
    if(TIFA_GGML_VULKAN)
        file(STRINGS "${ggml_SOURCE_DIR}/src/ggml-vulkan/ggml-vulkan.cpp" _vk_hits
             REGEX "GGML_VK_PIPELINE_CACHE_PATH")
        if(_vk_hits STREQUAL "")
            message(FATAL_ERROR
                "the ggml Vulkan pipeline-cache patch is not present in "
                "${ggml_SOURCE_DIR} (see cmake/patches/).  Delete the stale "
                "_deps/ggml-src-populate stamp (or the whole build dir) and "
                "re-configure.")
        endif()
    endif()

    file(STRINGS "${ggml_SOURCE_DIR}/include/ggml.h" _lstm_hits
         REGEX "GGML_OP_LSTM")
    if(_lstm_hits STREQUAL "")
        message(FATAL_ERROR
            "the ggml LSTM op patch is not present in ${ggml_SOURCE_DIR} "
            "(see cmake/patches/).  Delete the stale _deps/ggml-src-populate "
            "stamp (or the whole build dir) and re-configure.")
    endif()

    add_subdirectory("${ggml_SOURCE_DIR}" "${ggml_BINARY_DIR}")
endif()

# ---------------------------------------------------------------------------
# pocketfft (BSD-3-Clause) — header-only STFT / r2c FFT used by mel.cpp.
# Pinned to a known-good commit on the `cpp` branch.  The repo carries
# only a single usable header so we expose it via an INTERFACE target
# rather than pull in any build system.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        32424d2067c2e8043dc646a4e49754b2b40cc549   # cpp @ 2025-10
)
FetchContent_MakeAvailable(pocketfft)
# Idempotent: when aggregated by a parent (e.g. a sibling project), the parent
# may already define the same helper target.
if(NOT TARGET pocketfft)
    add_library(pocketfft INTERFACE)
    target_include_directories(pocketfft SYSTEM INTERFACE "${pocketfft_SOURCE_DIR}")
endif()

# ---------------------------------------------------------------------------
# dr_libs (Public Domain / MIT-0 dual) — single-header WAV reader used
# by the CLI layer.  The repo bundles many libs; we only include dr_wav.h.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG        243e26ffa08a24dc8ae2e7a8c57123d9e504690c   # master @ 2025-10
)
FetchContent_MakeAvailable(dr_libs)
if(NOT TARGET dr_wav)
    add_library(dr_wav INTERFACE)
    target_include_directories(dr_wav SYSTEM INTERFACE "${dr_libs_SOURCE_DIR}")
endif()

# Emit a NOTICE line so users know exactly which third-party libs got fetched.
message(STATUS "Third-party fetched:")
message(STATUS "  ggml       ${ggml_SOURCE_DIR}")
message(STATUS "  pocketfft  ${pocketfft_SOURCE_DIR}")
message(STATUS "  dr_libs    ${dr_libs_SOURCE_DIR}")