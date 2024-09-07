# MiR Description for ROS2 Simulation 
This package is used to create MiR mobile robots in ROS2. Most description files are based on [DFKI-NI/mir_robot](https://github.com/DFKI-NI/mir_robot) and this package is not affiliated with Mobile Industrial Robots. Please note that different parts of this repository are under different licenses. For details, go to the License section.

## Modifications
1. The way of importing mesh files is changed by introducing *$(find mir_description)*.
2. Gazebo sensor plugins are changed from *libgazebo_ros_sensor* to *libgazebo_ros_ray_sensor*, and from *libhector_gazebo_ros_imu* to *libgazebo_ros_imu_sensor*. Update some names and formats for some plugins (libgazebo_ros_p3d, etc.).
3. Migrate from the "prefix" style to the "namespace" style.

## License
Two licenses are included in the package: BSD and MIT. The BSD license is from [DFKI-NI/mir_robot](https://github.com/DFKI-NI/mir_robot) and the MIT license is for new code developed for ROS2 simulation. Under each folder, there is a LICENSE file to indicate its source. As a summary: 
- BSD: *urdf*, *meshes*, *config*
- MIT: *launch*, *rviz*

For other folders, if exist, no license is used.