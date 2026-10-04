# mecab-portability.patch

Four small fixes that let the MeCab tokenizer subset compile from the
upstream master tree with this repo's toolchains (MSVC x64 for Windows,
gcc/clang elsewhere) as a static library with no iconv.  Applied by
`cmake/Mecab.cmake` via `tifa_ggml_apply_patch`; the sentinel check in that
file greps `mecab.h` for `MECAB_STATIC_LIB`, so a stale `_deps` tree that
silently lost the patch fails the configure instead of the build.

## Touches

| file | problem | fix |
| --- | --- | --- |
| `mecab/src/mecab.h` | On Windows the API attributes are `dllexport` when `DLL_EXPORT` is defined and `dllimport` otherwise — there is no static-library spelling, so a static link fails with unresolved `__imp_*` symbols. | New `MECAB_STATIC_LIB` branch that drops the attributes entirely. |
| `mecab/src/common.h` | `WPATH` expands to `WPATH_FORCE` on MSVC, but `WPATH_FORCE` is only defined inside the MinGW (`__GNUC__`) branch — the upstream MSVC path cannot compile. | Hoist the `WPATH_FORCE` definition out of the MinGW branch. |
| `mecab/src/thread.h` | The `BEGINTHREAD` cast spells `_stdcall`, which is not a keyword outside MSVC (MinGW). | `__stdcall`, accepted by both. |
| `mecab/src/dictionary.cpp` | `pair_1st_cmp` derives from `std::binary_function`, removed in C++17. | Plain struct with a `const` `operator()`. |
| `darts.h`, `viterbi.cpp`, `char_property.h` | The removed `register` storage class is an error-severity diagnostic on current clang in C++17 mode (neither `-w` nor our warning set can downgrade it). | Drop the keyword; it has been meaningless for decades. |

## Build defines (cmake/Mecab.cmake)

`MECAB_USE_UTF8_ONLY` (dictionary and input are both UTF-8, so the no-iconv
path is a no-op), `MECAB_STATIC_LIB`, `DIC_VERSION=102`, `PACKAGE`/`VERSION`,
`MECAB_DEFAULT_RC` (always passed explicitly via `-r`), `HAVE_GETENV`,
`HAVE_STDINT_H` (MeCab's fallback `uint32_t` typedef is wrong on LP64),
`HAVE_UNSIGNED_LONG_LONG_INT`/`HAVE_LONG_LONG_INT` (without them the x64
`size_t` `operator<<` overloads of `StringBuffer` become ambiguous).  Windows
adds `HAVE_WINDOWS_H` + advapi32 (registry lookup in
`load_dictionary_resource`).  Everything POSIX adds
`HAVE_SYS_TYPES_H/HAVE_SYS_STAT_H/HAVE_FCNTL_H/HAVE_STRING_H/
HAVE_SYS_MMAN_H/HAVE_UNISTD_H/HAVE_DIRENT_H` (MeCab normally gets these from
config.h; mmap.h does not compile without them) plus `HAVE_MMAP`, so sys.dic
is mapped read-only instead of `read()` into a ~190 MB heap buffer.  All
warnings are suppressed for the third-party target (`register` storage class
in darts.h, the never-returning `die()` destructor, ...).
