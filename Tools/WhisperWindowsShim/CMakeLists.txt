cmake_minimum_required(VERSION 3.21)

project(whisper_unity_windows_shim LANGUAGES CXX)

if(NOT DEFINED WHISPER_CPP_ROOT)
    message(FATAL_ERROR "WHISPER_CPP_ROOT must point to a checked out whisper.cpp repository.")
endif()

set(WHISPER_INCLUDE_DIR "${WHISPER_CPP_ROOT}/include")
set(GGML_INCLUDE_DIR "${WHISPER_CPP_ROOT}/ggml/include")

add_library(libwhisper SHARED
    src/libwhisper_shim.cpp
)

target_compile_features(libwhisper PRIVATE cxx_std_17)
target_compile_definitions(libwhisper PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    UNICODE
    _UNICODE
)
target_include_directories(libwhisper PRIVATE
    "${WHISPER_INCLUDE_DIR}"
    "${GGML_INCLUDE_DIR}"
)

set_target_properties(libwhisper PROPERTIES
    OUTPUT_NAME "libwhisper"
)
