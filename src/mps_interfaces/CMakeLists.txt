cmake_minimum_required(VERSION 3.8)
project(mps_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(map_interfaces REQUIRED)

set(msg_files
  "msg/Schedule.msg"
  "msg/ScheduleArray.msg"
  "msg/ScheduleStamped.msg"
  "msg/ScheduleArrayStamped.msg"
  "msg/PathSchedule.msg"
  "msg/PathScheduleStamped.msg"
)

set(srv_files
  "srv/GetRobotSchedule.srv"
  "srv/GetInflatedMap.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs map_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
