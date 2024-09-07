import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    ld = LaunchDescription()
  
    pkg_name = 'map_description'
    pkg_share = FindPackageShare(package=pkg_name).find(pkg_name)

    ### Default Values ###
    default_timer_period = 0.2
    default_map_file_name = 'test_map.json'
    default_rviz_file_name = 'default_setting.rviz'
    
    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')
    map_file_name = LaunchConfiguration('map_file_name')

    use_rviz = LaunchConfiguration('use_rviz')
    rviz_file_name = LaunchConfiguration('rviz_file_name')

    ### Declare Launch Arguments ###
    declare_timer_period_arg = DeclareLaunchArgument(
        name='timer_period',
        default_value=str(default_timer_period),
        description='Map update period'
    )
    ld.add_action(declare_timer_period_arg)

    declare_map_file_name_arg = DeclareLaunchArgument(
        name='map_file_name',
        default_value=str(default_map_file_name),
        description='Map file name'
    )
    ld.add_action(declare_map_file_name_arg)

    declare_use_rviz_arg = DeclareLaunchArgument(
        name='use_rviz',
        default_value='false',
        choices=["true", "false"],
        description='Whether to start RVIZ'
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

    geo_map_node = Node(
        package=pkg_name,
        executable='geometric_map_node',
        name='geometric_map_node',
        output='screen',
        parameters=[
            {'timer_period': timer_period},
            {'map_file_name': map_file_name},
            {'manual_update': False}
        ]
    )
    ld.add_action(geo_map_node)

    map_transform_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
    )
    ld.add_action(map_transform_node)

    return ld