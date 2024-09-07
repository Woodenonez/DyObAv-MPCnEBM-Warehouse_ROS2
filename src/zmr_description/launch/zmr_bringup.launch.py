import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.descriptions import ParameterValue


def xacro_to_urdf(model_file_path: str, robot_namespace: str):
    """Not used yet, it returns the robot_description in xml format.
    """
    import xacro
    doc = xacro.process_file(model_file_path, mappings={'prefix': robot_namespace})
    robot_description = doc.toprettyxml(indent='  ')
    return robot_description

def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'zmr_description'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name) 

    ### Default Values ###
    default_model_file_name = 'robot.urdf.xacro'
    default_robot_namespace = 'zmr_X' # letters_numbers

    default_rviz_file_name = 'default_zmr_bringup.rviz'

    ### Declare Launch Variables ###
    model_file_name = LaunchConfiguration('model_file_name')
    robot_namespace = LaunchConfiguration('robot_namespace')

    rviz_file_name = LaunchConfiguration('rviz_file_name')
    use_rviz = LaunchConfiguration('use_rviz')
    use_joint_state_publisher = LaunchConfiguration('use_joint_state_publisher')

    ### Declare Launch Arguments ###
    declare_model_file_name_arg = DeclareLaunchArgument(
        name="model_file_name", 
        default_value=str(default_model_file_name), 
        description="Name of the robot model file to use"
    )
    ld.add_action(declare_model_file_name_arg)
    model_file_path = PathJoinSubstitution([pkg_share, 'urdf', model_file_name])

    declare_robot_namespace_arg = DeclareLaunchArgument(
        name="robot_namespace",
        default_value=str(default_robot_namespace),
        description="Prefix for the robot namespace",
    )
    ld.add_action(declare_robot_namespace_arg)

    declare_rviz_file_name_arg = DeclareLaunchArgument(
        name="rviz_file_name", 
        default_value=str(default_rviz_file_name), 
        description="Name of rviz config file"
    )
    ld.add_action(declare_rviz_file_name_arg)
    rviz_file_path = PathJoinSubstitution([pkg_share, 'rviz', rviz_file_name])

    declare_use_rviz_arg = DeclareLaunchArgument(
        name="use_rviz",
        default_value="true",
        choices=["true", "false"],
        description="Whether to start RVIZ",
    )
    ld.add_action(declare_use_rviz_arg)

    declare_use_joint_state_publisher_arg = DeclareLaunchArgument(
        name="use_joint_state_publisher",
        default_value="true",
        choices=["true", "false"],
        description="Whether to use the joint state publisher (or a custom one somewhere else)",
    )
    ld.add_action(declare_use_joint_state_publisher_arg)

    ### Define names ###
    robot_description = Command([
        'xacro ', model_file_path, ' namespace:=', robot_namespace
    ])
    robot_description_str = ParameterValue(robot_description, value_type=str)

    ### Nodes ###
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        namespace=robot_namespace,
        parameters=[{'robot_description': robot_description_str}],
    )
    ld.add_action(robot_state_publisher_node)

    joint_state_publisher_node = Node(
        condition=IfCondition(use_joint_state_publisher),
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        namespace=robot_namespace,
        parameters=[{'robot_description': robot_description_str}],
    )
    ld.add_action(joint_state_publisher_node)

    rviz2_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=["-d", rviz_file_path],
    )
    ld.add_action(rviz2_node)

    return ld