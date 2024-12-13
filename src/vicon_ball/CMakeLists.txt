cmake_minimum_required(VERSION 3.8)
project(vicon_ball)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(ball_vicon src/ball_vicon.cpp)
target_link_libraries(ball_vicon ViconDataStreamSDK_CPP)

target_include_directories(ball_vicon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(ball_vicon PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
  ball_vicon
  "rclcpp"
  "std_msgs"
  "geometry_msgs"
  "visualization_msgs"
)

install(TARGETS ball_vicon
  DESTINATION lib/${PROJECT_NAME})

ament_package()
