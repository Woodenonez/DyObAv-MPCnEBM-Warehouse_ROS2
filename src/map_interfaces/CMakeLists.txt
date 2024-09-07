cmake_minimum_required(VERSION 3.8)
project(map_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(msg_files
  "msg/ObjectMotionType.msg"
  "msg/ObjectSemanticType.msg"
  "msg/ObjectShapeType.msg"
  "msg/PolygonObject.msg"
  "msg/PolygonObjectStamped.msg"
  "msg/PolygonObjectArray.msg"
  "msg/PolygonObjectArrayStamped.msg"
  "msg/GeometricMap.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ament_export_include_directories(include)
# ament_export_dependencies(rosidl_default_runtime std_msgs geometry_msgs)

ament_package()
