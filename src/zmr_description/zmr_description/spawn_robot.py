"""
ROS 2 node to spawn a mobile robot.

Usage:
    Use the launch file in the launch directory to run the node.

Reference:
    1. Addison Sears-Collins (https://automaticaddison.com)
    2. ROS URDF tutorial (https://wiki.ros.org/urdf/Tutorials)
"""

import os
import sys

import rclpy
import xacro # for parsing xacro files

from launch_ros.substitutions import FindPackageShare

from gazebo_msgs.srv import SpawnEntity


def main():
    pkg_name = "zmr_description"

    argv = sys.argv[1:] # input arguments from user
    model_file_name, namespace, x, y, z, theta, *_ = argv

    model_name = model_file_name.split('.')[0]
    model_format = model_file_name.split('.')[-1]
    model_file_path = os.path.join(FindPackageShare(pkg_name).find(pkg_name), 
                                   "urdf", model_file_name)

    rclpy.init()

    node = rclpy.create_node("entity_spawner")

    node.get_logger().info(
        'Creating Service client to connect to `/spawn_entity`')
    client = node.create_client(SpawnEntity, "/spawn_entity")

    node.get_logger().info("Connecting to `/spawn_entity` service...")
    if not client.service_is_ready():
        client.wait_for_service()
        node.get_logger().info("...connected!")
    print(f"robot_model={model_file_path}")
    
    request = SpawnEntity.Request()
    request.name = namespace + '/' + model_name
    request.robot_namespace = namespace
    request.initial_pose.position.x = float(x)
    request.initial_pose.position.y = float(y)
    request.initial_pose.position.z = float(z)
    request.initial_pose.orientation.z = float(theta)

    if model_format == "xacro":
        doc = xacro.process_file(model_file_path, mappings={"namespace": namespace})
        request.xml = doc.toprettyxml(indent="  ")
    elif model_format in ["urdf", "sdf"]:
        request.xml = open(model_file_path, 'r').read()
    else:
        node.get_logger().error("Invalid model format. Must be xacro, urdf, or sdf.")

    node.get_logger().info(f"Sending service request (name: {request.name}) to `/spawn_entity`")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print('response: %r' % future.result())
    else:
        raise RuntimeError(
            'exception while calling service: %r' % future.exception())

    node.get_logger().info("Done! Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()