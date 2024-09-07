from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'mir_description'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name)

    ### Global Constants ###
    rviz_file_name = 'test_mir_bringup.rviz'

    ### Default Values ###
    default_namespace = 'mirX' # letters and numbers, e.g. mir2

    default_init_x = '0.0'
    default_init_y = '0.0'
    default_init_theta = '0.0'
    
    ### Declare Launch Variables ###
    namespace = LaunchConfiguration('namespace')

    init_x = LaunchConfiguration('init_x')
    init_y = LaunchConfiguration('init_y')
    init_theta = LaunchConfiguration('init_theta')

    use_gazebo = LaunchConfiguration('use_gazebo')
    use_rviz = LaunchConfiguration('use_rviz')

    ### Declare Launch Arguments ###
    declare_namespace_arg = DeclareLaunchArgument(
        name="namespace",
        default_value=str(default_namespace),
        description="Namespace for all MiR TF frames and topics",
    )
    ld.add_action(declare_namespace_arg)

    declare_use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Whether to start Gazebo')
    ld.add_action(declare_use_gazebo_arg)

    declare_use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to start RViz')
    ld.add_action(declare_use_rviz_arg)

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

    declare_init_theta_arg = DeclareLaunchArgument(
        'init_theta',
        default_value=default_init_theta,
        description='Initial theta/yaw orientation')
    ld.add_action(declare_init_theta_arg)

    ### Commands ###
    start_gazebo_cmd = ExecuteProcess(
        condition=IfCondition(use_gazebo),
        cmd=['gazebo', '--verbose','-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen')
    ld.add_action(start_gazebo_cmd)

    ### Other Launch Files ###
    launch_file_native_1 = PythonLaunchDescriptionSource(
        [pkg_share, '/launch/mir_description.launch.py'])
    include_launch_native_1 = IncludeLaunchDescription(launch_file_native_1, launch_arguments={
        'namespace': namespace,
        'use_joint_state_publisher': 'true'}.items()
    )
    ld.add_action(include_launch_native_1)

    ### Nodes ###
    rviz_file_path = PathJoinSubstitution([pkg_share, 'rviz', rviz_file_name])
    rviz2_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=["-d", rviz_file_path],
    )
    ld.add_action(rviz2_node)

    spawn_entity_cmd = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=['-entity', namespace, # robot name
                   '-topic', [namespace, '/', 'robot_description'], # or use -file
                   '-x', init_x,
                   '-y', init_y,
                   '-Y', init_theta],
        output='screen')
    ld.add_action(spawn_entity_cmd)

    return ld
