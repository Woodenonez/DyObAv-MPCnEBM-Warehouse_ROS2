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

    pkg_name = 'zmr_pure_pursuit'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name) 

    ### Default Values ###
    default_timer_period = 0.2
    default_robot_id = '0'
    default_robot_namespace = 'zmr_X' # letters_numbers
    default_config_file_name = 'robot_spec.yaml'

    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')
    robot_id = LaunchConfiguration('robot_id')
    robot_namespace = LaunchConfiguration('robot_namespace')
    config_file_name = LaunchConfiguration('config_file_name')

    ### Declare Launch Arguments ###
    declare_timer_period_arg = DeclareLaunchArgument(
        name="timer_period",
        default_value=str(default_timer_period),
        description="Period of timer in seconds",
    )
    ld.add_action(declare_timer_period_arg)

    declare_robot_id_arg = DeclareLaunchArgument(
        name="robot_id",
        default_value=str(default_robot_id),
        description="Robot ID",
    )
    ld.add_action(declare_robot_id_arg)

    declare_robot_namespace_arg = DeclareLaunchArgument(
        name="robot_namespace",
        default_value=str(default_robot_namespace),
        description="Prefix for the robot namespace",
    )
    ld.add_action(declare_robot_namespace_arg)

    declare_config_file_name_arg = DeclareLaunchArgument(
        name="config_file_name",
        default_value=str(default_config_file_name),
        description="File name of the robot specification",
    )
    ld.add_action(declare_config_file_name_arg)
    
    ### Nodes ###
    pure_pursuit_controller_node = Node(
        package=pkg_name,
        executable='pure_pursuit_controller_node',
        name='pure_pursuit_controller_node',
        namespace=robot_namespace,
        output='screen',
        parameters=[
            {'timer_period': timer_period},
            {'robot_id': robot_id},
            {'config_file_name': config_file_name},
        ],
    )
    ld.add_action(pure_pursuit_controller_node)

    return ld