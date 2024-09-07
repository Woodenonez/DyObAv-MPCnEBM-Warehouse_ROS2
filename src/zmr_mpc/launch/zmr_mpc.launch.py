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

    pkg_name = 'zmr_mpc'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name) 

    ### Default Values ###
    default_timer_period = 0.2
    default_robot_id = '0'
    default_robot_namespace = 'zmr_X' # letters_numbers
    default_config_mpc_fname = 'mpc_fast.yaml'
    default_config_robot_fname = 'robot_spec_zmr.yaml'

    default_enable_fleet_manager = 'false'

    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')
    robot_id = LaunchConfiguration('robot_id')
    robot_namespace = LaunchConfiguration('robot_namespace')
    config_mpc_fname = LaunchConfiguration('config_mpc_fname')
    config_robot_fname = LaunchConfiguration('config_robot_fname')

    enable_fleet_manager = LaunchConfiguration('enable_fleet_manager')

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

    declare_config_mpc_fname_arg = DeclareLaunchArgument(
        name="config_mpc_fname",
        default_value=str(default_config_mpc_fname),
        description="File name of the MPC configuration",
    )
    ld.add_action(declare_config_mpc_fname_arg)

    declare_config_robot_fname_arg = DeclareLaunchArgument(
        name="config_robot_fname",
        default_value=str(default_config_robot_fname),
        description="File name of the robot specification",
    )
    ld.add_action(declare_config_robot_fname_arg)

    declare_enable_fleet_manager_arg = DeclareLaunchArgument(
        name="enable_fleet_manager",
        default_value=str(default_enable_fleet_manager),
        description="Enable fleet manager",
    )
    ld.add_action(declare_enable_fleet_manager_arg)
    
    ### Nodes ###
    mpc_trajectory_tracker_node = Node(
        package=pkg_name,
        executable='mpc_trajectory_tracker_node',
        name='mpc_trajectory_tracker_node',
        # remappings=[
        #     ('/inflated_geometry_map', ['/zmr_X/inflated_geometry_map']),
        # ],
        namespace=robot_namespace,
        output='screen',
        parameters=[
            {'enable_fleet_manager': enable_fleet_manager},
            {'timer_period': timer_period},
            {'robot_id': robot_id},
            {'config_mpc_fname': config_mpc_fname},
            {'config_robot_fname': config_robot_fname},
        ],
    )
    ld.add_action(mpc_trajectory_tracker_node)

    return ld