cmake_minimum_required(VERSION 3.16)
project(gmsl_latency_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(gmsl_latency_benchmark src/gmsl_latency_benchmark.cpp)
target_include_directories(
  gmsl_latency_benchmark
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../bridge/third_party
)
target_link_libraries(gmsl_latency_benchmark PRIVATE ${OpenCV_LIBS})

set(GMSL_HAS_CUDA_WARPING OFF)
foreach(lib IN LISTS OpenCV_LIBS)
  if("${lib}" MATCHES "cudawarping")
    set(GMSL_HAS_CUDA_WARPING ON)
  endif()
endforeach()

if(GMSL_HAS_CUDA_WARPING)
  target_compile_definitions(gmsl_latency_benchmark PRIVATE GMSL_HAVE_CUDA_WARPING=1)
endif()

add_executable(v4l2_timestamp_probe src/v4l2_timestamp_probe.cpp)
target_include_directories(
  v4l2_timestamp_probe
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../bridge/third_party
)
