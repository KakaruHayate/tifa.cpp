# ----------------------------------------------------------------------------
# MeCab (BSD-3) — the Japanese morphological analyzer behind the
# `japanese-mecab` G2P converter.  Upstream segments lyrics with MeCab and
# takes UniDic's `pron` field as the kana reading, which then flows through
# the kana converter; this mirrors that pipeline natively.
#
# Upstream has no CMake build, so we compile the tokenizer subset (no
# training tools, no CRF learner, no iconv) straight from the fetched
# sources.  Not gated on TIFA_GGML_BUILD_MODEL: the ggml-free tifa_ggml_g2p
# library carries the converter too.
#
# Patches live in cmake/patches/mecab-portability.patch (+ sibling .md):
#   * mecab.h     — static-build guard for the Windows dllimport attributes
#   * common.h    — WPATH_FORCE is only defined for MinGW upstream, while the
#                   MSVC branch of WPATH expands to it
#   * thread.h    — `_stdcall` is not a keyword outside MSVC; `__stdcall` is
#                   accepted by both
#   * dictionary.cpp — std::binary_function was removed in C++17
#
# The compiled dictionary itself (unidic-lite, ~260 MB extracted) is NOT part
# of the build: it ships as a separate release asset and is probed at runtime
# (see the japanese-mecab branch in src/g2p/pipeline.cpp).
# ----------------------------------------------------------------------------
option(TIFA_GGML_MECAB "Build the MeCab-backed Japanese G2P converter" ON)

if(NOT TIFA_GGML_MECAB)
    message(STATUS "MeCab converter disabled (TIFA_GGML_MECAB=OFF); the "
                   "japanese-mecab converter will fall back to kana")
    return()
endif()

include(FetchContent)
include(${CMAKE_CURRENT_LIST_DIR}/ApplyPatch.cmake)

FetchContent_Declare(
    mecab
    URL      https://github.com/taku910/mecab/archive/61b90ba6e669dc2d7d533d4a80d206f3b31d52b1.tar.gz
    URL_HASH SHA256=7ad44f987ae0b7fd345c72a4b67e14dbe0f4bd1669d247a8aaeb0f15218a3fd1
)

FetchContent_GetProperties(mecab)
if(NOT mecab_POPULATED)
    FetchContent_Populate(mecab)
endif()

# Patches live OUTSIDE the populate guard (same reasoning as ggml above: a
# re-populate silently replaces the patched sources; the helper is idempotent).
tifa_ggml_apply_patch(
    "${mecab_SOURCE_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/patches/mecab-portability.patch"
    "MeCab portability fixes"
)

# Fail loudly rather than silently building against an unpatched tree.
file(STRINGS "${mecab_SOURCE_DIR}/mecab/src/mecab.h" _mecab_hits
     REGEX "MECAB_STATIC_LIB")
if(_mecab_hits STREQUAL "")
    message(FATAL_ERROR
        "the MeCab portability patch is not present in ${mecab_SOURCE_DIR} "
        "(see cmake/patches/).  Delete the stale _deps/mecab-src-populate "
        "stamp (or the whole build dir) and re-configure.")
endif()

set(TIFA_MECAB_SRC "${mecab_SOURCE_DIR}/mecab/src")
set(TIFA_MECAB_SOURCES
    "${TIFA_MECAB_SRC}/tagger.cpp"
    "${TIFA_MECAB_SRC}/tokenizer.cpp"
    "${TIFA_MECAB_SRC}/viterbi.cpp"
    "${TIFA_MECAB_SRC}/dictionary.cpp"
    "${TIFA_MECAB_SRC}/connector.cpp"
    "${TIFA_MECAB_SRC}/char_property.cpp"
    "${TIFA_MECAB_SRC}/context_id.cpp"
    "${TIFA_MECAB_SRC}/param.cpp"
    "${TIFA_MECAB_SRC}/utils.cpp"
    "${TIFA_MECAB_SRC}/string_buffer.cpp"
    "${TIFA_MECAB_SRC}/feature_index.cpp"
    "${TIFA_MECAB_SRC}/dictionary_rewriter.cpp"
    "${TIFA_MECAB_SRC}/iconv_utils.cpp"
    "${TIFA_MECAB_SRC}/nbest_generator.cpp"
    "${TIFA_MECAB_SRC}/writer.cpp"
    "${TIFA_MECAB_SRC}/libmecab.cpp"
)

# Idempotent for parent aggregations that may already define the target.
if(NOT TARGET mecab)
    add_library(mecab STATIC ${TIFA_MECAB_SOURCES})
    target_include_directories(mecab PUBLIC "${TIFA_MECAB_SRC}")
    # MECAB_STATIC_LIB must be PUBLIC: it also gates the dllimport attributes
    # in mecab.h for every consumer of the header.
    target_compile_definitions(mecab PUBLIC MECAB_STATIC_LIB)
    target_compile_definitions(mecab PRIVATE
        MECAB_USE_UTF8_ONLY                # dictionary and input are both UTF-8;
                                           # without iconv that path is a no-op
        DIC_VERSION=102
        PACKAGE="mecab"
        VERSION="0.996"
        MECAB_DEFAULT_RC="unused"          # always passed explicitly (-r)
        HAVE_GETENV
        HAVE_UNSIGNED_LONG_LONG_INT        # x64 size_t overloads in StringBuffer
        HAVE_LONG_LONG_INT
    )
    if(WIN32)
        target_compile_definitions(mecab PRIVATE
            HAVE_WINDOWS_H
            _CRT_SECURE_NO_DEPRECATE
            NOMINMAX
        )
        # registry lookup in load_dictionary_resource (utils.cpp)
        target_link_libraries(mecab PRIVATE advapi32)
    endif()
    # Third-party code: silence warnings entirely (register in darts.h, the
    # never-returning die() destructor, ...).  Our own targets keep theirs.
    if(MSVC)
        target_compile_options(mecab PRIVATE /w)
    else()
        target_compile_options(mecab PRIVATE -w)
    endif()
endif()

message(STATUS "Third-party fetched: mecab     ${mecab_SOURCE_DIR}")
