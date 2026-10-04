# ----------------------------------------------------------------------------
# Helper: apply a git patch once; idempotent across re-configures.
# Extracted from Dependencies.cmake so non-ggml dependencies (MeCab) can
# reuse it in builds where Dependencies.cmake is skipped entirely
# (TIFA_GGML_BUILD_MODEL=OFF).
# ----------------------------------------------------------------------------
include_guard(GLOBAL)

function(tifa_ggml_apply_patch source_dir patch_file patch_name)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        message(FATAL_ERROR "Git is required to apply ${patch_name}")
    endif()

    # git apply resolves patch paths against the enclosing repository root:
    # a fetched source tree without its own .git (URL archive, extracted
    # FetchContent) makes git walk up to OUR repo, treat the patch paths as
    # outside the current prefix and silently skip them with exit 0.  Make
    # sure the source tree is itself a repository so paths resolve locally;
    # a no-op when it already is one.
    if(NOT EXISTS "${source_dir}/.git")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" init -q
            RESULT_VARIABLE _init
            OUTPUT_QUIET ERROR_QUIET
        )
        if(NOT _init EQUAL 0)
            message(FATAL_ERROR "git init failed in ${source_dir}; cannot apply ${patch_name}")
        endif()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply --check "${patch_file}"
        RESULT_VARIABLE _check
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_check EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply "${patch_file}"
            RESULT_VARIABLE _apply
            ERROR_VARIABLE _err
        )
        if(NOT _apply EQUAL 0)
            message(FATAL_ERROR "Failed to apply ${patch_name}: ${_err}")
        endif()
        message(STATUS "Applied ${patch_name}")
        return()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply --reverse --check "${patch_file}"
        RESULT_VARIABLE _rcheck
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_rcheck EQUAL 0)
        message(STATUS "${patch_name} already applied")
    else()
        message(FATAL_ERROR "Could not apply ${patch_name}; upstream source layout changed")
    endif()
endfunction()
