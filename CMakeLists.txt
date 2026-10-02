cmake_minimum_required(VERSION 3.18)
project(tifa_ggml
    VERSION 0.1.2
    DESCRIPTION "TIFA (Token-Imputing Forced Aligner) native ggml inference backend"
    LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(TIFA_GGML_CUDA        "Enable ggml CUDA backend"     OFF)
option(TIFA_GGML_VULKAN      "Enable ggml Vulkan backend"   OFF)
# OFF builds only tifa_ggml_g2p (G2P + PFML, no ggml): the library downstream
# editors link against, and the way to build/test the G2P on its own.
option(TIFA_GGML_BUILD_MODEL "Build the aligner model backend (needs ggml)" ON)
option(TIFA_GGML_BUILD_CLI   "Build tifa_ggml_cli executable"   ON)
option(TIFA_GGML_BUILD_TESTS "Build unit tests (requires GoogleTest via FetchContent)" OFF)
# Optional: use ggml-llamafile's tuned sgemm kernels for CPU mul_mat
# (Q8_0/F32/BF16, AVX2+).  Off yields the stock ggml vec-dot kernels.
option(TIFA_GGML_LLAMAFILE   "Enable ggml-llamafile sgemm kernels (CPU)" ON)

# On Apple, default to Metal backend on. Use TIFA_GGML_METAL to override.
if(APPLE)
    option(TIFA_GGML_METAL "Enable ggml Metal backend" ON)
else()
    option(TIFA_GGML_METAL "Enable ggml Metal backend" OFF)
endif()

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Release bundles are assembled by copying build/bin/* verbatim, so every binary
# has to find its sibling shared libraries relative to itself.  Without this the
# linker bakes in absolute build-tree paths ($ORIGIN on Linux, @loader_path on
# macOS), and the shipped CLI cannot start on a machine that never saw the build
# directory.  Windows is unaffected (it searches the exe's directory first).
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
if(MSVC)
    # CMake's default C++ flags carry /EHsc, whose 'c' means "assume functions
    # declared extern "C" never throw".  ggml's whole API is extern "C", and its
    # Vulkan backend throws vk::SystemError from behind it (createInstance:
    # ErrorIncompatibleDriver on a machine with the loader but no working ICD).
    # With /EHsc the compiler therefore deleted our try/catch around
    # ggml_backend_vk_init outright, and the exception sailed past init_backend()
    # to main() -- the engine died instead of falling back to CPU, which is what
    # every GPU-less Windows box would have hit.  /EHs keeps synchronous C++
    # unwinding without the "extern C cannot throw" assumption.
    string(REPLACE "/EHsc" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${_cfg}" _CFG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/lib")
    endforeach()
    # NVCC treats a bare /utf-8 as an input path. Keep this MSVC option on
    # native C/C++ sources; CUDA sources already contain ASCII identifiers.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

if(MSVC)
    set(TIFA_GGML_WARN_FLAGS /W3 /wd4244 /wd4267 /wd4996)
else()
    set(TIFA_GGML_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter)
endif()

# -----------------------------------------------------------------------------
# Dependencies (ggml via FetchContent, plus optional Metal patch).
# Skipped entirely for a G2P-only build (TIFA_GGML_BUILD_MODEL=OFF), which
# needs no ggml at all.
# -----------------------------------------------------------------------------
if(TIFA_GGML_BUILD_MODEL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake)
endif()

# -----------------------------------------------------------------------------
# G2P / PFML library: tifa_ggml_g2p
# -----------------------------------------------------------------------------
# The grapheme-to-phoneme pipeline plus the PFML parser/serializer, with no
# ggml dependency.  Downstream editors (MaxLabel) link this one: it needs no
# model, no backend and no GGUF.
#
# The English OOV LSTM converter is the single ggml-dependent piece, so it is
# compiled out here (TIFA_G2P_NO_LSTM).  When TIFA_GGML_BUILD_MODEL is ON the
# full tifa_ggml library carries its own copy of these sources *with* the LSTM,
# so link exactly one of the two — they define the same symbols.
set(TIFA_GGML_G2P_SOURCES
    src/json.cpp
    src/g2p/pipeline.cpp
    src/g2p/encoding.cpp
    src/g2p/pinyin_engine.cpp
    src/g2p/pfml.cpp
    src/g2p/select.cpp
    src/g2p/public_api.cpp
)

add_library(tifa_ggml_g2p STATIC ${TIFA_GGML_G2P_SOURCES})
add_library(tifa_ggml::g2p ALIAS tifa_ggml_g2p)

target_include_directories(tifa_ggml_g2p
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(tifa_ggml_g2p PRIVATE ${TIFA_GGML_WARN_FLAGS})
target_compile_definitions(tifa_ggml_g2p PUBLIC TIFA_G2P_NO_LSTM=1)

# GGML-free self-tests.  These run in both configurations: TIFA_GGML_BUILD_TESTS
# only controls the gtest suite, which needs the model backend.
option(TIFA_GGML_BUILD_G2P_TESTS "Build the ggml-free G2P/PFML self-tests" ON)
if(TIFA_GGML_BUILD_G2P_TESTS)
    enable_testing()

    add_executable(tifa_ggml_pfml_roundtrip tests/pfml_roundtrip_standalone.cpp)
    target_link_libraries(tifa_ggml_pfml_roundtrip PRIVATE tifa_ggml::g2p)
    # Exercises the internal parser/serializer directly, so it needs src/.
    target_include_directories(tifa_ggml_pfml_roundtrip
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME pfml_roundtrip COMMAND tifa_ggml_pfml_roundtrip)

    add_executable(tifa_ggml_g2p_api tests/g2p_public_api_standalone.cpp)
    target_link_libraries(tifa_ggml_g2p_api PRIVATE tifa_ggml::g2p)
    add_test(NAME g2p_public_api COMMAND tifa_ggml_g2p_api)
endif()

if(TIFA_GGML_BUILD_MODEL)

# -----------------------------------------------------------------------------
# Core library: tifa_ggml
# -----------------------------------------------------------------------------
# NOTE: sources are added as the port progresses; every listed file must exist.
set(TIFA_GGML_CORE_SOURCES
    src/backend.cpp
    src/gguf_io.cpp
    src/json.cpp
    src/tensor_utils.cpp
    src/ops_basic.cpp
    src/ops_ffn.cpp
    src/ops_rope.cpp
    src/ops_attn.cpp
    src/ops_jebf.cpp
    src/model_tifa.cpp
    src/mel.cpp
    src/audio_io.cpp
    src/resample.cpp
    src/align_decode.cpp
    src/metrics.cpp
    src/textgrid.cpp
    src/g2p/pinyin_engine.cpp
    src/g2p/pipeline.cpp
    src/g2p/encoding.cpp
    src/g2p/select.cpp
    src/g2p/lstm_g2p.cpp
    src/g2p/pfml.cpp
    src/g2p/public_api.cpp
    src/breath/breath_mel.cpp
    src/breath/breath_net.cpp
    src/breath/breath.cpp
)

add_library(tifa_ggml STATIC ${TIFA_GGML_CORE_SOURCES})
add_library(tifa_ggml::tifa_ggml ALIAS tifa_ggml)

target_include_directories(tifa_ggml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ggml targets exported by the FetchContent'd ggml source tree.
# v0.11.0 provides `ggml` as the umbrella interface target.
target_link_libraries(tifa_ggml
    PUBLIC  ggml
    PRIVATE pocketfft
    PRIVATE dr_wav          # dr_libs headers (dr_wav / dr_flac / dr_mp3)
)

target_compile_options(tifa_ggml PRIVATE ${TIFA_GGML_WARN_FLAGS})
target_compile_definitions(tifa_ggml PUBLIC
    TIFA_GGML_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    TIFA_GGML_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    TIFA_GGML_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Propagate optional-backend feature macros so consumers can branch on them.
if(TIFA_GGML_METAL)
    target_compile_definitions(tifa_ggml PUBLIC TIFA_GGML_HAS_METAL=1)
endif()
if(TIFA_GGML_CUDA)
    target_compile_definitions(tifa_ggml PUBLIC TIFA_GGML_HAS_CUDA=1)
endif()
if(TIFA_GGML_VULKAN)
    target_compile_definitions(tifa_ggml PUBLIC TIFA_GGML_HAS_VULKAN=1)
endif()

# -----------------------------------------------------------------------------
# CLI executable
# -----------------------------------------------------------------------------
if(TIFA_GGML_BUILD_CLI)
    add_executable(tifa_ggml_cli
        src/cli/main.cpp
        src/cli/phone_input.cpp
    )
    target_include_directories(tifa_ggml_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(tifa_ggml_cli PRIVATE
        tifa_ggml::tifa_ggml
        dr_wav
    )
    target_compile_options(tifa_ggml_cli PRIVATE ${TIFA_GGML_WARN_FLAGS})

    # Ship the MSVC runtime next to the engine.  The release bundles are copied
    # out of build/bin verbatim, and a machine without the VC++ redistributable
    # (a clean Windows install, a fresh VM) has no msvcp140.dll / vcruntime140.dll
    # -- the engine would not start there at all.  Chromium links its CRT
    # statically, so nothing else in the Electron bundle pulls the
    # redistributable in as a side effect.
    if(MSVC)
        set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
        include(InstallRequiredSystemLibraries)
        foreach(_crt IN LISTS CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
            add_custom_command(TARGET tifa_ggml_cli POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_crt}" "$<TARGET_FILE_DIR:tifa_ggml_cli>")
        endforeach()
    endif()
endif()

# -----------------------------------------------------------------------------
# Tests (optional)
# -----------------------------------------------------------------------------
if(TIFA_GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

endif()  # TIFA_GGML_BUILD_MODEL

# -----------------------------------------------------------------------------
# Install (minimal; mainly so consumers can embed via add_subdirectory)
# -----------------------------------------------------------------------------
install(DIRECTORY include/tifa_ggml
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

install(TARGETS tifa_ggml_g2p
        EXPORT tifa_ggml-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

if(TIFA_GGML_BUILD_MODEL)
    install(TARGETS tifa_ggml
            EXPORT tifa_ggml-targets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
endif()

message(STATUS "")
message(STATUS "tifa_ggml ${PROJECT_VERSION} configuration:")
message(STATUS "  Build model ........ ${TIFA_GGML_BUILD_MODEL}")
message(STATUS "  Metal .............. ${TIFA_GGML_METAL}")
message(STATUS "  CUDA ............... ${TIFA_GGML_CUDA}")
message(STATUS "  Vulkan ............. ${TIFA_GGML_VULKAN}")
message(STATUS "  Build CLI .......... ${TIFA_GGML_BUILD_CLI}")
message(STATUS "  Build Tests ........ ${TIFA_GGML_BUILD_TESTS}")
message(STATUS "")
