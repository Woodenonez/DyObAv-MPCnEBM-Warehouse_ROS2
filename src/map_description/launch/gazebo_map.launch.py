# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from launch import LaunchDescription
# from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare

'''Use the script/json_to_world.py to generate a world file from a json file.
'''

def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'map_description'
    pkg_share = FindPackageShare(package=pkg_name).find(pkg_name)

    ### Default values ###
    default_world_file_name = 'test_map.world'

    ### Declare launch variables ###
    world_file_name = LaunchConfiguration('world_file_name')

    ### Declare launch arguments ###
    declare_world_file_name_arg = DeclareLaunchArgument(
        'world_file_name',
        default_value=default_world_file_name,
        description='Name of the world file to load'
    )
    ld.add_action(declare_world_file_name_arg)
    world_file_path = PathJoinSubstitution([pkg_share, 'worlds', world_file_name])

    ### Command execution ###
    gazebo = ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file_path, '-s', 'libgazebo_ros_init.so', 
            '-s', 'libgazebo_ros_factory.so'],
            output='screen')
    ld.add_action(gazebo)

    return ld