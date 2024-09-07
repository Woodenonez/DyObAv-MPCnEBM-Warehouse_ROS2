import os
from typing import Union

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.actions import IncludeLaunchDescription

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def summon_robot(pkg_share, model_file_name, robot_namespace: Union[str, LaunchConfiguration], robot_init_state: list[float]) -> list:
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
        'init_x': str(robot_init_state[0]),
        'init_y': str(robot_init_state[1]),
        'init_theta': str(robot_init_state[2])}.items()
    )

    if isinstance(robot_namespace, LaunchConfiguration):
        map_transform_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            namespace=robot_namespace,
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', [robot_namespace, '/odom']], # x y z yaw pitch roll frame_id child_frame_id period_in_ms
        )
    elif isinstance(robot_namespace, str):
        map_transform_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_'+robot_namespace,
            arguments=['0', '0', '0', '0', '0', '0', 'world', robot_namespace+'/odom'], # x y z yaw pitch roll frame_id child_frame_id period_in_ms
        )
    else:
        raise TypeError("robot_namespace must be either a string or a LaunchConfiguration.")
    
    return [include_launch_rviz, include_launch_gazebo, map_transform_node]


def generate_launch_description():
    ld = LaunchDescription()
  
    pkg_name = 'zmr_demo'
    pkg_share = FindPackageShare(package=pkg_name).find(pkg_name)

    ### Fixed Values ###
    robot_1_init_state = [0.0, 0.0, 0.0] # initial x, y, theta
    robot_2_init_state = [10.0, 10.0, 0.0] # initial x, y, theta

    ### Default Values ###
    default_timer_period = 0.2

    default_robot_id_list = ['0', '1']
    default_robot_namespace_list = ['zmr_'+str(i) for i in default_robot_id_list]
    default_model_file_name = 'robot.urdf.xacro'
    default_config_file_name = 'robot_spec_zmr.yaml'

    default_map_file_name = 'test_map.json'
    default_graph_file_name = 'test_graph.json'
    default_schedule_file_name = 'test_schedule.csv'

    default_rviz_file_name = 'dual_robot_surroundings.rviz'
    default_world_file_name = 'test_map.world'
    
    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')

    robot_1_id = LaunchConfiguration('robot_1_id')
    robot_1_namespace = LaunchConfiguration('robot_1_namespace')
    robot_2_id = LaunchConfiguration('robot_2_id')
    robot_2_namespace = LaunchConfiguration('robot_2_namespace')
    model_file_name = LaunchConfiguration('model_file_name')
    config_file_name = LaunchConfiguration('config_file_name')

    map_file_name = LaunchConfiguration('map_file_name')
    graph_file_name = LaunchConfiguration('graph_file_name')
    schedule_file_name = LaunchConfiguration('schedule_file_name')

    use_controller = LaunchConfiguration('use_controller')

    use_rviz = LaunchConfiguration('use_rviz')
    rviz_file_name = LaunchConfiguration('rviz_file_name')
    world_file_name = LaunchConfiguration('world_file_name')

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

    declare_config_file_name_arg = DeclareLaunchArgument(
        name='config_file_name',
        default_value=str(default_config_file_name),
        description='Model file name'
    )
    ld.add_action(declare_config_file_name_arg)

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

    declare_use_controller_arg = DeclareLaunchArgument(
        name='use_controller',
        default_value='true',
        choices=["true", "false"],
        description='Whether to start controller'
    )
    ld.add_action(declare_use_controller_arg)

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

    declare_world_file_name_arg = DeclareLaunchArgument(
        name='world_file_name',
        default_value=str(default_world_file_name),
        description='Name of the world file to load'
    )
    ld.add_action(declare_world_file_name_arg)

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

    world_file_path = PathJoinSubstitution([FindPackageShare(package=pkg_name_1).find(pkg_name_1), 'worlds', world_file_name])
    gazebo = ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file_path, '-s', 'libgazebo_ros_init.so', 
            '-s', 'libgazebo_ros_factory.so'],
            output='screen')
    ld.add_action(gazebo)

    pkg_name_2 = 'mps_motion_plan'
    launch_file_2 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_2).find(pkg_name_2), '/launch/motion_plan.launch.py'])
    include_launch_2 = IncludeLaunchDescription(launch_file_2, launch_arguments={
        "schedule_file_name": schedule_file_name,
        'graph_file_name': graph_file_name}.items()
    )
    ld.add_action(include_launch_2)

    pkg_name_3 = 'zmr_description'
    pkg_share_robot = FindPackageShare(package=pkg_name_3).find(pkg_name_3)
    for r_init, r_ns in zip([robot_1_init_state, robot_2_init_state], 
                            [robot_1_namespace, robot_2_namespace]):
        robot_launch_list = summon_robot(pkg_share_robot, 
                                         model_file_name, 
                                         r_ns, 
                                         r_init)
        for action in robot_launch_list:
            ld.add_action(action)

    ### Other Launch Files ###
    pkg_name_4 = 'zmr_pure_pursuit'
    pkg_share_pure_pursuit = FindPackageShare(package=pkg_name_4).find(pkg_name_4)
    launch_file_4_1 = PythonLaunchDescriptionSource(
        [pkg_share_pure_pursuit, '/launch/zmr_pure_pursuit.launch.py'])
    include_launch_4_1 = IncludeLaunchDescription(launch_file_4_1, launch_arguments={
        'timer_period': timer_period,
        'robot_namespace': robot_1_namespace,
        'robot_id': robot_1_id,
        'config_file_name': config_file_name}.items(),
        condition=IfCondition(use_controller)
    )
    ld.add_action(include_launch_4_1)

    launch_file_4_2 = PythonLaunchDescriptionSource(
        [pkg_share_pure_pursuit, '/launch/zmr_pure_pursuit.launch.py'])
    include_launch_4_2 = IncludeLaunchDescription(launch_file_4_2, launch_arguments={
        'timer_period': timer_period,
        'robot_namespace': robot_2_namespace,
        'robot_id': robot_2_id,
        'config_file_name': config_file_name}.items(),
        condition=IfCondition(use_controller)
    )
    ld.add_action(include_launch_4_2)

    return ld