include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tifa_ggml_tests
    support/ggml_test_env.cpp
    support/reference_io.cpp
    test_backend.cpp
    test_decode.cpp
    test_g2p_lstm.cpp
    test_breathe_merge.cpp
    test_textgrid.cpp
    # the 2PASS merge lives with the CLI's format readers; compiling the unit
    # in keeps the test off the CLI executable
    ../src/cli/phone_input.cpp
)

target_compile_definitions(tifa_ggml_tests PRIVATE
    GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
)

target_link_libraries(tifa_ggml_tests PRIVATE
    tifa_ggml::tifa_ggml
    GTest::gtest_main
)

target_include_directories(tifa_ggml_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(MSVC)
    target_compile_options(tifa_ggml_tests PRIVATE /W3 /wd4244 /wd4267 /wd4996 /utf-8)
else()
    target_compile_options(tifa_ggml_tests PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

include(GoogleTest)
gtest_discover_tests(tifa_ggml_tests)
