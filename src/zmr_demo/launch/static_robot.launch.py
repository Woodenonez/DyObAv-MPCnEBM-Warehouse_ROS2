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

    pkg_map_description_share = FindPackageShare(package='map_description').find('map_description')
    pkg_mps_motion_plan_share = FindPackageShare(package='mps_motion_plan').find('mps_motion_plan')
    pkg_zmr_description_share = FindPackageShare(package='zmr_description').find('zmr_description')

    ### Default Values ###
    default_timer_period = 0.2

    default_robot_id = '0'
    default_map_file_name = 'test_map.json'
    default_graph_file_name = 'test_graph.json'
    default_schedule_file_name = 'test_schedule.csv'
    default_model_file_name = 'robot.urdf.xacro'

    default_rviz_file_name = 'static_robot.rviz'
    
    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')

    robot_id = LaunchConfiguration('robot_id')
    map_file_name = LaunchConfiguration('map_file_name')
    model_file_name = LaunchConfiguration('model_file_name')
    graph_file_name = LaunchConfiguration('graph_file_name')
    schedule_file_name = LaunchConfiguration('schedule_file_name')

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

    declare_robot_id_arg = DeclareLaunchArgument(
        name='robot_id',
        default_value=str(default_robot_id),
        description='Robot ID'
    )
    ld.add_action(declare_robot_id_arg)

    declare_graph_file_name_arg = DeclareLaunchArgument(
        name='graph_file_name',
        default_value=str(default_graph_file_name),
        description='Graph file name'
    )
    ld.add_action(declare_graph_file_name_arg)

    declare_map_file_name_arg = DeclareLaunchArgument(
        name='schedule_file_name',
        default_value=str(default_schedule_file_name),
        description='Schedule file name (in CSV format)'
    )
    ld.add_action(declare_map_file_name_arg)

    declare_model_file_name_arg = DeclareLaunchArgument(
        name="model_file_name", 
        default_value=str(default_model_file_name), 
        description="Name of the robot model file to use"
    )
    ld.add_action(declare_model_file_name_arg)

    declare_use_rviz_arg = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
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

    ### Other Launch Files ###
    launch_file_1 = PythonLaunchDescriptionSource(
        [pkg_map_description_share, '/launch/geo_map.launch.py'])
    include_launch_1 = IncludeLaunchDescription(launch_file_1, launch_arguments={
        'timer_period': timer_period,
        'map_file_name': map_file_name,
        'use_rviz': 'false'}.items()
    )
    ld.add_action(include_launch_1)

    launch_file_3 = PythonLaunchDescriptionSource(
        [pkg_mps_motion_plan_share, '/launch/motion_plan.launch.py'])
    include_launch_3 = IncludeLaunchDescription(launch_file_3, launch_arguments={
        "schedule_file_name": schedule_file_name,
        'graph_file_name': graph_file_name}.items()
    )
    ld.add_action(include_launch_3)

    launch_file_4 = PythonLaunchDescriptionSource(
        [pkg_zmr_description_share, '/launch/zmr_bringup.launch.py'])
    include_launch_4 = IncludeLaunchDescription(launch_file_4, launch_arguments={
        'model_file_name': model_file_name,
        'use_rviz': 'false',
        'use_joint_state_publisher': 'true'}.items()
    )
    ld.add_action(include_launch_4)
    
    ### Nodes ###
    schedule_viz_node = Node(
        package='mps_motion_plan',
        executable='schedule_visualize_node',
        name='schedule_visualize_node',
        output='screen',
        parameters=[{
            'timer_period': timer_period,
            'robot_id': robot_id,
        }],
    )
    ld.add_action(schedule_viz_node)

    map_transform_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'zmr_X/base_footprint'],
    )
    ld.add_action(map_transform_node)

    return ld