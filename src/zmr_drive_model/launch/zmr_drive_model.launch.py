import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'zmr_drive_model'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name) 

    ### Default Values ###
    default_robot_namespace = 'zmr_X' # letters_numbers
    default_init_x = 0.0
    default_init_y = 0.0
    default_init_theta = 0.0

    ### Declare Launch Variables ###
    robot_namespace = LaunchConfiguration('robot_namespace')
    init_x = LaunchConfiguration('init_x')
    init_y = LaunchConfiguration('init_y')
    init_theta = LaunchConfiguration('init_theta')
    keep_cmd_vel = LaunchConfiguration('keep_cmd_vel')

    ### Declare Launch Arguments ###
    declare_robot_namespace_arg = DeclareLaunchArgument(
        name="robot_namespace",
        default_value=str(default_robot_namespace),
        description="Prefix for the robot namespace",
    )
    ld.add_action(declare_robot_namespace_arg)

    declare_init_x_arg = DeclareLaunchArgument(
        name="init_x",
        default_value=str(default_init_x),
        description="Initial x position",
    )
    ld.add_action(declare_init_x_arg)

    declare_init_y_arg = DeclareLaunchArgument(
        name="init_y",
        default_value=str(default_init_y),
        description="Initial y position",
    )
    ld.add_action(declare_init_y_arg)

    declare_init_theta_arg = DeclareLaunchArgument(
        name="init_theta",
        default_value=str(default_init_theta),
        description="Initial theta position",
    )
    ld.add_action(declare_init_theta_arg)

    declare_keep_cmd_vel_arg = DeclareLaunchArgument(
        name="keep_cmd_vel",
        default_value=str(True),
        description="Keep the cmd_vel topic alive even if no subscriber is present",
    )
    ld.add_action(declare_keep_cmd_vel_arg)
    
    ### Nodes ###
    diff_drive_odom_node = Node(
        package=pkg_name,
        executable='diff_drive_odom_node',
        name='diff_drive_odom_node',
        namespace=robot_namespace,
        parameters=[
            {'timer_period': 0.05},
            {'robot_namespace': robot_namespace},
            {'init_x': init_x},
            {'init_y': init_y},
            {'init_theta': init_theta},
            {'keep_cmd_vel': keep_cmd_vel},
        ],
        output='screen',
    )
    ld.add_action(diff_drive_odom_node)

    map_transform_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', [robot_namespace, '/odom']],
    )
    ld.add_action(map_transform_node)

    return ld