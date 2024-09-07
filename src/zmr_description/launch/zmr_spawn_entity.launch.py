import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'zmr_description'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name)

    ### Default Values ###
    default_model_file_name = 'robot.urdf.xacro'
    default_robot_namespace = 'zmr_X' # letters_numbers

    default_init_x = '0.0'
    default_init_y = '0.0'
    default_init_z = '0.0'
    default_init_theta = '0.0'
    
    ### Declare Launch Variables ###
    model_file_name = LaunchConfiguration('model_file_name')
    robot_namespace = LaunchConfiguration('robot_namespace')

    init_x = LaunchConfiguration('init_x')
    init_y = LaunchConfiguration('init_y')
    init_z = LaunchConfiguration('init_z')
    init_theta = LaunchConfiguration('init_theta')

    use_gazebo = LaunchConfiguration('use_gazebo')

    ### Declare Launch Arguments ###
    declare_model_file_name_arg = DeclareLaunchArgument(
        'model_file_name',
        default_value=default_model_file_name,
        description='Name of the model file to load')
    ld.add_action(declare_model_file_name_arg)

    declare_robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value=default_robot_namespace,
        description='Prefix for the robot namespace')
    ld.add_action(declare_robot_namespace_arg)

    declare_use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Whether to start Gazebo')
    ld.add_action(declare_use_gazebo_arg)

    declare_init_x_arg = DeclareLaunchArgument(
        'init_x',
        default_value=default_init_x,
        description='Initial x position')
    ld.add_action(declare_init_x_arg)

    declare_init_y_arg = DeclareLaunchArgument(
        'init_y',
        default_value=default_init_y,
        description='Initial y position')
    ld.add_action(declare_init_y_arg)

    declare_init_z_arg = DeclareLaunchArgument(
        'init_z',
        default_value=default_init_z,
        description='Initial z position')
    ld.add_action(declare_init_z_arg)

    declare_init_theta_arg = DeclareLaunchArgument(
        'init_theta',
        default_value=default_init_theta,
        description='Initial theta position')
    ld.add_action(declare_init_theta_arg)

    ### Commands ###
    start_gazebo_cmd = ExecuteProcess(
        condition=IfCondition(use_gazebo),
        cmd=['gazebo', '--verbose','-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen')
    ld.add_action(start_gazebo_cmd)

    ### Nodes ### (choose one of the following)
    # model_file_path = PathJoinSubstitution([pkg_share, 'urdf', model_file_name])
    # spawn_entity_cmd = Node(
    #     package='gazebo_ros', 
    #     executable='spawn_entity.py',
    #     arguments=['-entity', model_file_name,  
    #                '-file', model_file_path ], output='screen')
    # ld.add_action(spawn_entity_cmd)

    spawn_args = [model_file_name, robot_namespace, init_x, init_y, init_z, init_theta]
    spawn_entity = Node(package=pkg_name, executable='spawn_robot',
                        arguments=spawn_args,
                        output='screen')
    ld.add_action(spawn_entity)

    return ld
