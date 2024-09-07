# Description
This package contains the drive model for the ZMR robot. It is originally based on the [diff_drive](https://github.com/rticommunity/gazebo-dds-plugins/blob/master/src/diff_drive/README.md) plugin from Gazebo.

# Notes
The signal flow is as follows:
```
cmd_vel -> [drive_model] -> wheel_speed -> [inverse_drive_model] -> cmd_vel_real ->[forward_kinematics] -> odometry
```
The `cmd_vel` is from a controller that gives the desired velocity of the robot. The `cmd_vel_real` is the real velocity of the robot. If the inverse drive model is a perfect inverse of the drive model, then `cmd_vel` and `cmd_vel_real` should be the same. 