cmake_minimum_required(VERSION 3.20)
project(excavator_real_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_filename_component(EXCAVATOR_REAL_STACK_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

add_subdirectory(
    "${EXCAVATOR_REAL_STACK_ROOT}/control"
    "${CMAKE_BINARY_DIR}/control_build"
)

add_executable(excavator_real_bridge
    src/excavator_real_bridge.cpp
)

target_include_directories(excavator_real_bridge PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/third_party"
)

target_link_libraries(excavator_real_bridge PRIVATE
    robot_control_api
)

if(UNIX AND NOT APPLE)
    target_link_libraries(excavator_real_bridge PRIVATE rt)
endif()
