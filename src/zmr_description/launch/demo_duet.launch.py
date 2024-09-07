import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.actions import IncludeLaunchDescription

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition

from launch.launch_description_sources import PythonLaunchDescriptionSource


def summon_robot(pkg_share, model_file_name, robot_namespace: str, robot_init_position: list[float]) -> list:
    """Return a list of actions (Rviz, Gazebo, MapTransform) to be added in the launch description to summon a robot."""
    launch_file_rviz = PythonLaunchDescriptionSource(
        [pkg_share, '/launch/zmr_bringup.launch.py'])
    include_launch_rviz = IncludeLaunchDescription(launch_file_rviz, launch_arguments={
        'model_file_name': model_file_name,
        'robot_namespace': robot_namespace,
        'use_rviz': 'false',
        'use_joint_state_publisher': 'false'}.items()
    )

    launch_file_gazebo = PythonLaunchDescriptionSource(
        [pkg_share, '/launch/zmr_spawn_entity.launch.py'])
    include_launch_gazebo = IncludeLaunchDescription(launch_file_gazebo, launch_arguments={
        'model_file_name': model_file_name,
        'robot_namespace': robot_namespace,
        'use_gazebo': 'false',
        'init_x': str(robot_init_position[0]),
        'init_y': str(robot_init_position[1]),
        'init_z': str(robot_init_position[2])}.items()
    )

    map_transform_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_'+robot_namespace,
        arguments=['0', '0', '0', '0', '0', '0', 'world', robot_namespace+'/odom'], # x y z yaw pitch roll frame_id child_frame_id period_in_ms
    )
    
    return [include_launch_rviz, include_launch_gazebo, map_transform_node]

def generate_launch_description():
    ld = LaunchDescription()
  
    pkg_name = 'zmr_description'
    pkg_share = FindPackageShare(package=pkg_name).find(pkg_name)

    ### Fixed Values ###
    robot_1_namespace = 'zmr_1'
    robot_1_init_position = [0.0, 0.0, 0.0]

    robot_2_namespace = 'zmr_2'
    robot_2_init_position = [1.0, 0.0, 0.0]

    ### Default Values ###
    default_model_file_name = 'robot.urdf.xacro'
    default_rviz_file_name = 'default_demo_duet.rviz'
    
    ### Declare Launch Variables ###
    use_rviz = LaunchConfiguration('use_rviz')
    use_gazebo = LaunchConfiguration('use_gazebo')
    model_file_name = LaunchConfiguration('model_file_name')
    rviz_file_name = LaunchConfiguration('rviz_file_name')

    ### Declare Launch Arguments ###
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

    declare_use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Whether to start Gazebo')
    ld.add_action(declare_use_gazebo_arg)

    ### Nodes and Commands ###
    rviz2_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=["-d", rviz_file_path],
    )
    ld.add_action(rviz2_node)

    start_gazebo_cmd = ExecuteProcess(
        condition=IfCondition(use_gazebo),
        cmd=['gazebo', '--verbose','-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen')
    ld.add_action(start_gazebo_cmd)

    ### Summon Robots ###
    robot_1_launch_list = summon_robot(pkg_share, model_file_name, robot_1_namespace, robot_1_init_position)
    for action in robot_1_launch_list:
        ld.add_action(action)
        
    robot_2_launch_list = summon_robot(pkg_share, model_file_name, robot_2_namespace, robot_2_init_position)
    for action in robot_2_launch_list:
        ld.add_action(action)

    return ld