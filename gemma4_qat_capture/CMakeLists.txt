cmake_minimum_required(VERSION 3.21)
project(gemma4_qat_capture LANGUAGES C CXX)

# Point at a pinned local checkout of ggml-org/llama.cpp.
# This example is Gemma 4-specific; generic GGUF editing stays on ara.
if (NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set -DLLAMA_CPP_DIR=/absolute/path/to/llama.cpp")
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}" llama-cpp-build)

add_executable(gemma4-qat-capture main.cpp)
target_compile_features(gemma4-qat-capture PRIVATE cxx_std_17)
target_link_libraries(gemma4-qat-capture PRIVATE llama llama-common)
target_include_directories(gemma4-qat-capture PRIVATE
    "${LLAMA_CPP_DIR}/common"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/include"
)
