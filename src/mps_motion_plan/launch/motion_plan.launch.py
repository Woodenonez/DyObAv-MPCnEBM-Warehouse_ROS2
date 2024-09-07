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
  
    pkg_name = 'mps_motion_plan'
    pkg_share = FindPackageShare(package=pkg_name).find(pkg_name)

    ### Default Values ###
    default_graph_file_name = 'test_graph.json'
    default_schedule_file_name = 'test_schedule.csv'
    
    ### Declare Launch Variables ###
    graph_file_name = LaunchConfiguration('graph_file_name')
    schedule_file_name = LaunchConfiguration('schedule_file_name')

    rescale_graph = LaunchConfiguration('rescale_graph')
    rescale_map = LaunchConfiguration('rescale_map')
 
    ### Declare Launch Arguments ###
    declare_graph_file_name_arg = DeclareLaunchArgument(
        name='graph_file_name',
        default_value=str(default_graph_file_name),
        description='Graph file name'
    )
    ld.add_action(declare_graph_file_name_arg)

    declare_schedule_file_name_arg = DeclareLaunchArgument(
        name='schedule_file_name',
        default_value=str(default_schedule_file_name),
        description='Schedule file name'
    )
    ld.add_action(declare_schedule_file_name_arg)

    declare_rescale_graph_arg = DeclareLaunchArgument(
        name='rescale_graph',
        default_value='false',
        choices=['true', 'false'],
        description='Rescale graph'
    )
    ld.add_action(declare_rescale_graph_arg)

    declare_rescale_map_arg = DeclareLaunchArgument(
        name='rescale_map',
        default_value='false',
        choices=['true', 'false'],
        description='Rescale map'
    )
    ld.add_action(declare_rescale_map_arg)

    ### Nodes ###
    global_path_coordinate_node = Node(
        package=pkg_name,
        executable='global_path_coordinate_node',
        name='global_path_coordinate_node',
        output='screen',
        parameters=[
            {'graph_file_name': graph_file_name},
            {'schedule_file_name': schedule_file_name},
            {'rescale_graph': rescale_graph},
            {'rescale_map': rescale_map}
        ]
    )
    ld.add_action(global_path_coordinate_node)

    return ld