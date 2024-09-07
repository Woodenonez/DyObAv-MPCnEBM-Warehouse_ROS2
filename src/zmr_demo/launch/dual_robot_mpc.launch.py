import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.actions import IncludeLaunchDescription

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    ld = LaunchDescription()
  
    pkg_name = 'zmr_demo'
    pkg_share = FindPackageShare(package=pkg_name).find(pkg_name)

    ### Default Values ###
    default_timer_period = 0.2
    default_robot_id_list = ['0', '1']
    default_robot_namespace_list = ['zmr_'+str(i) for i in default_robot_id_list]
    default_config_mpc_fname = 'mpc_default.yaml'
    default_config_robot_fname = 'robot_spec_zmr.yaml'
    
    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')

    robot_1_id = LaunchConfiguration('robot_1_id')
    robot_2_id = LaunchConfiguration('robot_2_id')
    robot_1_namespace = LaunchConfiguration('robot_1_namespace')
    robot_2_namespace = LaunchConfiguration('robot_2_namespace')

    config_mpc_fname = LaunchConfiguration('config_mpc_fname')
    config_robot_fname = LaunchConfiguration('config_robot_fname')

    ### Declare Launch Arguments ###
    declare_timer_period_arg = DeclareLaunchArgument(
        name='timer_period',
        default_value=str(default_timer_period),
        description='Map update period'
    )
    ld.add_action(declare_timer_period_arg)

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

    # Robots - Start
    declare_robot_1_id_arg = DeclareLaunchArgument(
        name='robot_1_id',
        default_value=str(default_robot_id_list[0]),
        description='Robot ID'
    )
    ld.add_action(declare_robot_1_id_arg)

    declare_robot_1_namespace_arg = DeclareLaunchArgument(
        name='robot_1_namespace',
        default_value=str(default_robot_namespace_list[0]),
        description='Robot namespace'
    )
    ld.add_action(declare_robot_1_namespace_arg)

    declare_robot_2_id_arg = DeclareLaunchArgument(
        name='robot_2_id',
        default_value=str(default_robot_id_list[1]),
        description='Robot ID'
    )
    ld.add_action(declare_robot_2_id_arg)

    declare_robot_2_namespace_arg = DeclareLaunchArgument(
        name='robot_2_namespace',
        default_value=str(default_robot_namespace_list[1]),
        description='Robot namespace'
    )
    ld.add_action(declare_robot_2_namespace_arg)
    # Robots - End

    ### Other Launch Files ###
    pkg_name_1 = 'zmr_mpc'
    launch_file_1_1 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_1).find(pkg_name_1), '/launch/zmr_mpc.launch.py'])
    include_launch_1_1 = IncludeLaunchDescription(launch_file_1_1, launch_arguments={
        'timer_period': timer_period,
        'robot_namespace': robot_1_namespace,
        'robot_id': robot_1_id,
        'config_mpc_fname': config_mpc_fname,
        'config_robot_fname': config_robot_fname}.items()
    )
    ld.add_action(include_launch_1_1)

    launch_file_1_2 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_1).find(pkg_name_1), '/launch/zmr_mpc.launch.py'])
    include_launch_1_2 = IncludeLaunchDescription(launch_file_1_2, launch_arguments={
        'timer_period': timer_period,
        'robot_namespace': robot_2_namespace,
        'robot_id': robot_2_id,
        'config_mpc_fname': config_mpc_fname,
        'config_robot_fname': config_robot_fname}.items()
    )
    ld.add_action(include_launch_1_2)

    return ld