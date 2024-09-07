cmake_minimum_required(VERSION 3.8)
project(gazebo_ros_actor_pose_plugin)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(include/${PROJECT_NAME} ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(gazebo_ros_actor_pose SHARED src/gazebo_ros_actor_pose.cpp)
target_link_libraries(gazebo_ros_actor_pose ${GAZEBO_LIBRARIES} ${rclcpp_LIBRARIES})

ament_target_dependencies(gazebo_ros_actor_pose 
  rclcpp 
  geometry_msgs
  std_msgs
  gazebo
)

install(TARGETS
  gazebo_ros_actor_pose
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
