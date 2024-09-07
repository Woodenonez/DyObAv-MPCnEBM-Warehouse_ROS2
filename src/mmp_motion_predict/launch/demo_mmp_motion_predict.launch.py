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

    pkg_name = 'mmp_motion_predict'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name) 

    ### Default Values ###
    default_timer_period = 0.2
    default_config_mmp_fname = 'wsd_1t20_poselu_enll_train.yaml'
    default_map_fname = 'warehouse_sim_original/background.png'

    default_use_rviz = True
    default_rviz_file_name = 'demo.rviz'

    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')
    config_mmp_fname = LaunchConfiguration('config_mmp_fname')
    map_fname = LaunchConfiguration('map_fname')

    use_rviz = LaunchConfiguration('use_rviz')
    rviz_file_name = LaunchConfiguration('rviz_file_name')

    ### Declare Launch Arguments ###
    declare_timer_period_arg = DeclareLaunchArgument(
        name="timer_period",
        default_value=str(default_timer_period),
        description="Period of timer in seconds",
    )
    ld.add_action(declare_timer_period_arg)
    
    declare_config_mmp_fname_arg = DeclareLaunchArgument(
        name="config_mmp_fname",
        default_value=default_config_mmp_fname,
        description="Config file name for motion prediction",
    )
    ld.add_action(declare_config_mmp_fname_arg)

    declare_map_fname_arg = DeclareLaunchArgument(
        name="map_fname",
        default_value=default_map_fname,
        description="Map file name",
    )
    ld.add_action(declare_map_fname_arg)

    declare_use_rviz_arg = DeclareLaunchArgument(
        name="use_rviz",
        default_value=str(default_use_rviz),
        description="Whether to launch rviz",
    )
    ld.add_action(declare_use_rviz_arg)

    declare_rviz_file_name_arg = DeclareLaunchArgument(
        name="rviz_file_name", 
        default_value=str(default_rviz_file_name), 
        description="Name of rviz config file"
    )
    ld.add_action(declare_rviz_file_name_arg)
    rviz_file_path = PathJoinSubstitution([pkg_share, 'rviz', rviz_file_name])
    
    ### Nodes ###
    rviz2_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=["-d", rviz_file_path],
    )
    ld.add_action(rviz2_node)

    motion_prediction_node = Node(
        package=pkg_name,
        executable='motion_prediction_node',
        name='motion_prediction_node',
        # namespace='mmp',
        output='screen',
        parameters=[
            {'timer_period': timer_period},
            {'config_file_name': config_mmp_fname},
        ],
    )
    ld.add_action(motion_prediction_node)

    demo_traj_publisher_node = Node(
        package=pkg_name,
        executable='demo_traj_publisher',
        name='demo_traj_publisher',
        # namespace='mmp',
        output='screen',
        parameters=[
            {'timer_period': timer_period},
            {'map_file_name': map_fname},
        ],
    )
    ld.add_action(demo_traj_publisher_node)

    return ld