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

    pkg_name = 'zmr_manager'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name) 

    ### Default Values ###
    default_robot_ids_str = '0, 1' # need to parse in the node
    default_robot_prefix = 'zmr_' # 'zmr_' or 'mir'

    ### Declare Launch Variables ###
    robot_ids_str = LaunchConfiguration('robot_ids_str')
    robot_prefix = LaunchConfiguration('robot_prefix')

    ### Declare Launch Arguments ###
    declare_robot_ids_str_arg = DeclareLaunchArgument(
        name='robot_ids_str',
        default_value=str(default_robot_ids_str),
        description="List of robot ids (in string format, separated by comma)",
    )
    ld.add_action(declare_robot_ids_str_arg)

    declare_robot_prefix_arg = DeclareLaunchArgument(
        name='robot_prefix',
        default_value=str(default_robot_prefix),
        description="Prefix for robot names",
    )
    
    ### Nodes ###
    robot_manager_node = Node(
        package=pkg_name,
        executable='robot_manager_node',
        name='robot_manager_node',
        output='screen',
        parameters=[
            {'robot_ids_str': robot_ids_str},
            {'robot_prefix': robot_prefix},
        ],
    )
    ld.add_action(robot_manager_node)

    return ld