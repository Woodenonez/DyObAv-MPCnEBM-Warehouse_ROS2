from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'mir_description'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name)

    ### Global Constants ###
    rviz_file_name = 'test_mir_bringup.rviz'

    ### Default Values ###
    default_namespace = 'mirX' # letters and numbers, e.g. mir2

    ### Declare Launch Variables ###
    namespace = LaunchConfiguration('namespace')

    ### Declare Launch Arguments ###
    declare_namespace_arg = DeclareLaunchArgument(
        name="namespace",
        default_value=str(default_namespace),
        description="Namespace for all MiR TF frames and topics",
    )
    ld.add_action(declare_namespace_arg)

    ### Nodes ###
    rviz_file_path = PathJoinSubstitution([pkg_share, 'rviz', rviz_file_name])
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=["-d", rviz_file_path],
    )
    ld.add_action(rviz2_node)

    ### Other Launch Files ###
    launch_file_native_1 = PythonLaunchDescriptionSource(
        [pkg_share, '/launch/mir_description.launch.py'])
    include_launch_native_1 = IncludeLaunchDescription(launch_file_native_1, launch_arguments={
        # 'namespace': namespace,
        'use_joint_state_publisher': 'true'}.items()
    )
    ld.add_action(include_launch_native_1)

    return ld