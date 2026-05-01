cmake_minimum_required(VERSION 3.20)

project(dinov3_trt_cpp LANGUAGES CXX)

add_library(dinov3_trt_cpp INTERFACE)
target_compile_features(dinov3_trt_cpp INTERFACE cxx_std_17)
target_include_directories(
  dinov3_trt_cpp
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

option(DINOV3_TRT_CPP_BUILD_TESTS "Build C++ contract tests" ON)
option(DINOV3_TRT_CPP_ENABLE_TENSORRT "Build C++ TensorRT utilities" OFF)
set(DINOV3_TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
set(DINOV3_CUDA_ROOT "" CACHE PATH "CUDA Toolkit installation root")

if(DINOV3_TRT_CPP_ENABLE_TENSORRT)
  if(NOT DINOV3_TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
    set(DINOV3_TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
  endif()
  if(NOT DINOV3_TENSORRT_ROOT)
    message(FATAL_ERROR "Set DINOV3_TENSORRT_ROOT or TENSORRT_ROOT to build TensorRT utilities")
  endif()
  if(NOT DINOV3_CUDA_ROOT AND DEFINED ENV{CUDA_PATH})
    set(DINOV3_CUDA_ROOT "$ENV{CUDA_PATH}")
  endif()
  if(NOT DINOV3_CUDA_ROOT AND DEFINED ENV{CUDA_HOME})
    set(DINOV3_CUDA_ROOT "$ENV{CUDA_HOME}")
  endif()
  if(NOT DINOV3_CUDA_ROOT)
    message(FATAL_ERROR "Set DINOV3_CUDA_ROOT, CUDA_PATH, or CUDA_HOME to build TensorRT utilities")
  endif()

  set(DINOV3_TENSORRT_INCLUDE_DIR "${DINOV3_TENSORRT_ROOT}/include")
  set(DINOV3_CUDA_INCLUDE_DIR "${DINOV3_CUDA_ROOT}/include")
  if(WIN32)
    set(DINOV3_TENSORRT_NVINFER_LIB "${DINOV3_TENSORRT_ROOT}/lib/nvinfer_10.lib")
    set(DINOV3_CUDA_RUNTIME_LIB "${DINOV3_CUDA_ROOT}/lib/x64/cudart.lib")
  else()
    set(DINOV3_TENSORRT_NVINFER_LIB "${DINOV3_TENSORRT_ROOT}/lib/libnvinfer.so")
    set(DINOV3_CUDA_RUNTIME_LIB "${DINOV3_CUDA_ROOT}/lib64/libcudart.so")
  endif()

  if(NOT EXISTS "${DINOV3_TENSORRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR "TensorRT include directory not found: ${DINOV3_TENSORRT_INCLUDE_DIR}")
  endif()
  if(NOT EXISTS "${DINOV3_CUDA_INCLUDE_DIR}/cuda_runtime_api.h")
    message(FATAL_ERROR "CUDA include directory not found: ${DINOV3_CUDA_INCLUDE_DIR}")
  endif()
  if(NOT EXISTS "${DINOV3_TENSORRT_NVINFER_LIB}")
    message(FATAL_ERROR "TensorRT nvinfer library not found: ${DINOV3_TENSORRT_NVINFER_LIB}")
  endif()
  if(NOT EXISTS "${DINOV3_CUDA_RUNTIME_LIB}")
    message(FATAL_ERROR "CUDA runtime library not found: ${DINOV3_CUDA_RUNTIME_LIB}")
  endif()

  add_library(
    dinov3_trt_cpp_tensorrt
    STATIC
      src/engine_inspector.cpp
      src/trt_inferer.cpp
  )
  target_link_libraries(dinov3_trt_cpp_tensorrt PUBLIC dinov3_trt_cpp)
  target_include_directories(
    dinov3_trt_cpp_tensorrt
    PRIVATE
      "${DINOV3_TENSORRT_INCLUDE_DIR}"
      "${DINOV3_CUDA_INCLUDE_DIR}"
  )
  target_link_libraries(
    dinov3_trt_cpp_tensorrt
    PRIVATE
      "${DINOV3_TENSORRT_NVINFER_LIB}"
      "${DINOV3_CUDA_RUNTIME_LIB}"
  )

  add_executable(dinov3_trt_inspect_engine tools/inspect_engine.cpp)
  target_link_libraries(dinov3_trt_inspect_engine PRIVATE dinov3_trt_cpp_tensorrt)

  add_executable(dinov3_trt_runtime_smoke tools/runtime_smoke.cpp)
  target_link_libraries(dinov3_trt_runtime_smoke PRIVATE dinov3_trt_cpp_tensorrt)

  add_executable(dinov3_trt_runtime_benchmark tools/runtime_benchmark.cpp)
  target_link_libraries(dinov3_trt_runtime_benchmark PRIVATE dinov3_trt_cpp_tensorrt)

  add_executable(dinov3_trt_dump_outputs tools/dump_outputs.cpp)
  target_link_libraries(dinov3_trt_dump_outputs PRIVATE dinov3_trt_cpp_tensorrt)
endif()

if(DINOV3_TRT_CPP_BUILD_TESTS)
  enable_testing()

  add_executable(dinov3_trt_cpp_contract_tests tests/test_contracts.cpp)
  target_link_libraries(dinov3_trt_cpp_contract_tests PRIVATE dinov3_trt_cpp)

  add_test(NAME dinov3_trt_cpp_contract_tests COMMAND dinov3_trt_cpp_contract_tests)
endif()
