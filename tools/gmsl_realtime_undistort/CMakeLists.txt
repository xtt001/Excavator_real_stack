cmake_minimum_required(VERSION 3.16)
project(gmsl_realtime_undistort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(gmsl_realtime_undistort src/gmsl_realtime_undistort.cpp)
target_include_directories(
  gmsl_realtime_undistort
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../bridge/third_party
)
target_link_libraries(gmsl_realtime_undistort PRIVATE ${OpenCV_LIBS})

set(GMSL_HAS_CUDA_WARPING OFF)
foreach(lib IN LISTS OpenCV_LIBS)
  if("${lib}" MATCHES "cudawarping")
    set(GMSL_HAS_CUDA_WARPING ON)
  endif()
endforeach()

if(GMSL_HAS_CUDA_WARPING)
  target_compile_definitions(gmsl_realtime_undistort PRIVATE GMSL_HAVE_CUDA_WARPING=1)
endif()
