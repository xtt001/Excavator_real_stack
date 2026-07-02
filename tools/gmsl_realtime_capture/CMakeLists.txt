cmake_minimum_required(VERSION 3.16)
project(gmsl_realtime_capture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
  gmsl_realtime_capture
  src/gmsl_realtime_capture.cpp
  src/v4l2_camera.cpp
)

target_include_directories(
  gmsl_realtime_capture
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../bridge/third_party
)

find_package(Threads REQUIRED)
target_link_libraries(gmsl_realtime_capture PRIVATE Threads::Threads)

find_path(
  GMSL_NVBUF_INCLUDE_DIR
  nvbufsurface.h
  PATHS
    /usr/src/jetson_multimedia_api/include
)
find_path(
  GMSL_CUDA_INCLUDE_DIR
  cudaEGL.h
  PATHS
    /usr/local/cuda/targets/aarch64-linux/include
    /usr/local/cuda/include
)
find_library(
  GMSL_NVBUFSURFACE_LIBRARY
  nvbufsurface
  PATHS
    /usr/lib/aarch64-linux-gnu/nvidia
    /usr/lib/aarch64-linux-gnu
)
find_library(
  GMSL_CUDA_DRIVER_LIBRARY
  cuda
  PATHS
    /usr/lib/aarch64-linux-gnu/nvidia
    /usr/lib/aarch64-linux-gnu
)
find_library(
  GMSL_CUDART_LIBRARY
  cudart
  PATHS
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/local/cuda/lib64
)
find_library(
  GMSL_EGL_LIBRARY
  EGL
  PATHS
    /usr/lib/aarch64-linux-gnu
    /usr/lib/aarch64-linux-gnu/tegra-egl
)

if(GMSL_NVBUF_INCLUDE_DIR AND GMSL_CUDA_INCLUDE_DIR AND GMSL_NVBUFSURFACE_LIBRARY AND GMSL_CUDA_DRIVER_LIBRARY AND GMSL_CUDART_LIBRARY AND GMSL_EGL_LIBRARY)
  target_compile_definitions(gmsl_realtime_capture PRIVATE GMSL_HAVE_JETSON_ZERO_COPY=1)
  target_include_directories(
    gmsl_realtime_capture
    PRIVATE
      ${GMSL_NVBUF_INCLUDE_DIR}
      ${GMSL_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(
    gmsl_realtime_capture
    PRIVATE
      ${GMSL_NVBUFSURFACE_LIBRARY}
      ${GMSL_CUDA_DRIVER_LIBRARY}
      ${GMSL_CUDART_LIBRARY}
      ${GMSL_EGL_LIBRARY}
  )
else()
  message(WARNING "Jetson zero-copy headers/libs not found; --memory dmabuf will be unavailable.")
endif()

include(CheckLanguage)
if(NOT CMAKE_CUDA_COMPILER)
  find_program(
    GMSL_CUDA_NVCC
    nvcc
    PATHS
      /usr/local/cuda/bin
      /usr/local/cuda-12.6/bin
      /usr/local/cuda-12/bin
  )
  if(GMSL_CUDA_NVCC)
    set(CMAKE_CUDA_COMPILER "${GMSL_CUDA_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND GMSL_NVBUF_INCLUDE_DIR AND GMSL_CUDA_INCLUDE_DIR AND GMSL_NVBUFSURFACE_LIBRARY)
  enable_language(CUDA)
  find_package(OpenCV REQUIRED)

  add_executable(
    gmsl_realtime_preprocess_probe
    src/gmsl_realtime_preprocess_probe.cpp
    src/v4l2_camera.cpp
    src/uyvy_preprocess_kernel.cu
  )
  target_include_directories(
    gmsl_realtime_preprocess_probe
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/../../bridge/third_party
      ${GMSL_NVBUF_INCLUDE_DIR}
      ${GMSL_CUDA_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_compile_definitions(gmsl_realtime_preprocess_probe PRIVATE GMSL_HAVE_JETSON_ZERO_COPY=1)
  target_link_libraries(
    gmsl_realtime_preprocess_probe
    PRIVATE
      Threads::Threads
      ${GMSL_NVBUFSURFACE_LIBRARY}
      ${GMSL_CUDA_DRIVER_LIBRARY}
      ${GMSL_CUDART_LIBRARY}
      ${GMSL_EGL_LIBRARY}
      ${OpenCV_LIBS}
  )
  set_target_properties(
    gmsl_realtime_preprocess_probe
    PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_SEPARABLE_COMPILATION OFF
  )
else()
  message(WARNING "CUDA compiler or Jetson zero-copy dependencies not found; gmsl_realtime_preprocess_probe will not be built.")
endif()
