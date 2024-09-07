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

    default_robot_id = '0'
    default_robot_namespace = 'zmr_X'
    default_model_file_name = 'robot.urdf.xacro'

    default_map_file_name = 'warehouse_map.json'
    default_graph_file_name = 'warehouse_graph.json'
    default_schedule_file_name = 'warehouse_schedule.csv'

    default_rviz_file_name = 'single_robot_surroundings.rviz'
    
    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')

    robot_id = LaunchConfiguration('robot_id')
    robot_namespace = LaunchConfiguration('robot_namespace')
    model_file_name = LaunchConfiguration('model_file_name')

    map_file_name = LaunchConfiguration('map_file_name')
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

    declare_model_file_name_arg = DeclareLaunchArgument(
        name='model_file_name',
        default_value=str(default_model_file_name),
        description='Model file name'
    )
    ld.add_action(declare_model_file_name_arg)

    declare_robot_id_arg = DeclareLaunchArgument(
        name='robot_id',
        default_value=str(default_robot_id),
        description='Robot ID'
    )
    ld.add_action(declare_robot_id_arg)

    declare_robot_namespace_arg = DeclareLaunchArgument(
        name='robot_namespace',
        default_value=str(default_robot_namespace),
        description='Robot namespace'
    )
    ld.add_action(declare_robot_namespace_arg)

    declare_map_file_name_arg = DeclareLaunchArgument(
        name='map_file_name',
        default_value=str(default_map_file_name),
        description='Map file name'
    )
    ld.add_action(declare_map_file_name_arg)

    declare_graph_file_name_arg = DeclareLaunchArgument(
        name='graph_file_name',
        default_value=str(default_graph_file_name),
        description='Graph file name'
    )
    ld.add_action(declare_graph_file_name_arg)

    declare_schedule_file_name_arg = DeclareLaunchArgument(
        name='schedule_file_name',
        default_value=str(default_schedule_file_name),
        description='Schedule file name (in CSV format)'
    )
    ld.add_action(declare_schedule_file_name_arg)

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
    pkg_name_1 = 'map_description'
    launch_file_1 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_1).find(pkg_name_1), '/launch/geo_map.launch.py'])
    include_launch_1 = IncludeLaunchDescription(launch_file_1, launch_arguments={
        'timer_period': timer_period,
        'map_file_name': map_file_name,
        'use_rviz': 'false'}.items()
    )
    ld.add_action(include_launch_1)

    pkg_name_2 = 'mps_motion_plan'
    launch_file_2 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_2).find(pkg_name_2), '/launch/motion_plan.launch.py'])
    include_launch_2 = IncludeLaunchDescription(launch_file_2, launch_arguments={
        "schedule_file_name": schedule_file_name,
        'graph_file_name': graph_file_name,
        'rescale_map': 'false',
        'rescale_graph': 'false'}.items()
    )
    ld.add_action(include_launch_2)

    pkg_name_3 = 'zmr_description'
    launch_file_3 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_3).find(pkg_name_3), '/launch/zmr_bringup.launch.py'])
    include_launch_3 = IncludeLaunchDescription(launch_file_3, launch_arguments={
        'robot_namespace': robot_namespace,
        'model_file_name': model_file_name,
        'use_rviz': 'false',
        'use_joint_state_publisher': 'false'}.items()
    )
    ld.add_action(include_launch_3)

    pkg_name_4 = 'zmr_drive_model'
    launch_file_4 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_4).find(pkg_name_4), '/launch/zmr_drive_model.launch.py'])
    include_launch_4 = IncludeLaunchDescription(launch_file_4, launch_arguments={
        'robot_namespace': robot_namespace,
        'init_x': '1.0',
        'init_y': '-1.7',
        'init_theta': '1.57',
        'keep_cmd_vel': 'true'}.items()
    )
    ld.add_action(include_launch_4)

    pkg_name_6 = 'zmr_mpc'
    launch_file_6 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_6).find(pkg_name_6), '/launch/zmr_mpc.launch.py'])
    include_launch_6 = IncludeLaunchDescription(launch_file_6, launch_arguments={
        'timer_period': timer_period,
        'robot_id': robot_id}.items()
    )
    ld.add_action(include_launch_6)

    return ld