from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from launch.conditions import IfCondition


def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'mir_description'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name)

    ### Global Constants ###
    model_file_name = 'mir.urdf.xacro'
    mir_type = 'mir_100' # 'mir_100' or 'mir_250'

    ### Default Values ###
    default_namespace = 'mirX' # letters and numbers, e.g. mir2

    ### Declare Launch Variables ###
    namespace = LaunchConfiguration('namespace')
    use_joint_state_publisher = LaunchConfiguration('use_joint_state_publisher')

    ### Declare Launch Arguments ###
    declare_namespace_arg = DeclareLaunchArgument(
        name="namespace",
        default_value=str(default_namespace),
        description="Namespace for all MiR TF frames and topics",
    )
    ld.add_action(declare_namespace_arg)

    declare_use_joint_state_publisher_arg = DeclareLaunchArgument(
        name="use_joint_state_publisher",
        default_value="false",
        choices=["true", "false"],
        description="Whether to use the joint state publisher (or a custom one somewhere else)",
    )
    ld.add_action(declare_use_joint_state_publisher_arg)

    ### Description name ###
    model_file_path = PathJoinSubstitution([pkg_share, 'urdf', model_file_name])
    robot_description = Command([
        'xacro ', model_file_path, ' namespace:=', namespace, ' mir_type:=', mir_type
    ])
    robot_description_content = ParameterValue(robot_description, value_type=str)

    ### Nodes ###
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        output='screen',
        parameters=[{'robot_description': robot_description_content}],
    )
    ld.add_action(robot_state_publisher_node)

    joint_state_publisher_node = Node(
        condition=IfCondition(use_joint_state_publisher),
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        output='screen',
        parameters=[{'robot_description': robot_description_content}],
    )
    ld.add_action(joint_state_publisher_node)

    return ld