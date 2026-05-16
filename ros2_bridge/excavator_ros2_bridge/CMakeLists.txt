cmake_minimum_required(VERSION 3.20)
project(excavator_ros2_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

get_filename_component(_excavator_ros2_bridge_dir "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_excavator_ros2_bridge_root "${_excavator_ros2_bridge_dir}/.." ABSOLUTE)
add_subdirectory(
  "${_excavator_ros2_bridge_root}/fpv_frame_store"
  ${CMAKE_CURRENT_BINARY_DIR}/fpv_frame_store_build
)

add_executable(fpv_image_subscriber
  src/fpv_image_subscriber_node.cpp
)

target_include_directories(fpv_image_subscriber PRIVATE
  "${_excavator_ros2_bridge_root}/fpv_frame_store"
)

ament_target_dependencies(fpv_image_subscriber
  rclcpp
  sensor_msgs
  cv_bridge
)

target_link_libraries(fpv_image_subscriber
  fpv_frame_store
  ${OpenCV_LIBS}
)

if(UNIX AND NOT APPLE)
  target_link_libraries(fpv_image_subscriber pthread rt)
endif()

install(TARGETS fpv_image_subscriber
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
