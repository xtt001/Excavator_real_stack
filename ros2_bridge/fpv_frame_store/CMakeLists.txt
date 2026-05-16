cmake_minimum_required(VERSION 3.20)
project(fpv_frame_store LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(fpv_frame_store STATIC
    fpv_frame_store.cpp
    fpv_read_state.cpp
)

target_include_directories(fpv_frame_store PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(fpv_frame_store PUBLIC pthread rt)
endif()
