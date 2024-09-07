# Create (Nov. XX 2023) [TrajPlanFleet_ScheMPC_ScheFleetMPC_ROS2]
To simulate a MPC-based collision-free fleet control with a high-level scheduler in ROS2.

# 20231127 - (GitLab init)
Upload to Chalmers GitLab.
Finished packages: /map_description and /map_interfaces.
~~Previous packages: /twr_description~~

# 20231128 - (Package about motion planning and scheduling)
Create packages: /mps_scheduler and /mps_interfaces.
Scheduler-related messages are defined in /mps_interfaces.

# 20231129
The scheduler part is finished and uploaded to GitLab. 
(Ignore the visibility graph part for now.)

~~TODO: Need further modification from msg to srv.~~ [20240118: Finished].

# 20231204
The global motion plan is finished together with the visualization of it.

~~TODO: Should have a global coordinator to handle the global plan and local plan, right now it is individual path planner for each robot.~~ [20240118: Finished].

# 20231205
Modify the geo_map node so that it also publishes the map visualization (get rid the geo_map_vis node).
Create a demo in twr_demo package to visualize the global plan and the map, later add the static robot in RVIZ.
Create a script to convert json to Gazebo world file in the map_description package.

# 20231207
Create the wheel controller package to convert the linear and angular velocity to the wheel speed.
**Major Update** Go through all exist code to enable multi-robot simulation.

# 20231211
Finish the twr_wheel_controller package and connect it with the other parts by publishing the JointState.
Refine some details in all packages.

# 20231212
Build a new package "zmr_description" to fix multiple robot simulation problem!!! Cheers!!!

~~TODO: Get rid of the old twr_description package and recheck all packages.~~

# 20231218
Build a new package "zmr_drive_model" to simulate the robot motion without using Gazebo plugin.
Migrate the twr_demo package to zmr_demo package.

# 20231221
All packages are connected with the new "zmr_pure_pursuit" package. Bugs about namespace are fixed.

~~TODO: Fix the pure pursuit controller. The controller doesn't work as expected.~~

# 20231229
Fix the pure pursuit controller. The controller works as expected now.
TODO: Add timeout for keep_cmd_vel, otherwise the robot will keep moving even if the controller stops.

# 20240108
Create the MPC package.

# 20240116
Create the dual robot pure pursuit demo.

# 20240118
Finished the modification from msg to srv, and the global coordinator (the scheduler is not separated anymore).
Finished the demo for the dual robot pure pursuit.
Add the Gazebo example for a single robot.

~~TODO: Fix the problem in the Gazebo example, which is after adding the world file, the robot flies away.~~ [20240119: Finished].

# 20240119
Fix the problem in the Gazebo example. Now it works with the pure pursuit controller.

# 20240124
Create the dual robot gazebo demo. Fix the problem in the pure pursuit controller, which is that the previous version publish the cmd_vel in terms of the world frame but not the robot frame as used in the differential drive model.

# 20240125
Create the placeholder for the multimodal motion prediction package.

# 20240311
Finish the MPC package.
~~TODO: Make sure MPC works as expected and have a launch file to run the MPC.~~ [20240312: Finished].
Change inflated map from topic to service.

# 20240312
Create dual robot MPC demo.
~~TODO: Add robot manager for fleet coordination.~~ [20240313: Finished].

# 20240313
Add robot manager but the service to fetch other robots' states doesn't work as expected.

# 20240314
Fixed the robot manager service by changing call_async to call.
Now the robot manager works as expected.
Fix the robot specification in the MPC package to match the robot model used in the Gazebo example.
Change <real_time_update_rate> to 500 (default is 1000) in the world file to make the simulation smoother.

# 20240319
Finish the motion prediction package a demo showing the prediction result on rviz.
TODO: The current mmp interface is not good, need to change it to a better one.

# 20240403
Create the Gazebo world package for the AWS warehouse.
Add the actor_pose plugin but ignore it while building since it doesn't work.
Finish the warehouse rviz and Gazebo demo.
~~TODO: Connect everything together and make a launch file to run the whole system!!!~~

# 20240416
Try actor pose plugin for Gazebo again but still doesn't work.

# 20240422
The actor plugin works now!

# 20240423
The actor plugin works properly and is connected to the MMP package.

# 20240425
:) The whole system works now!
TODO: ~~Change the "skin" of the robot in the Gazebo example. Introduce the MiR robot model.~~ Also check unfinished TODOs in the code.

# 20240503
Change the robot model in the Gazebo example to MiR robot.
TODO: ~~Run the final check and record the video.~~

# 20240906
Create mir dual robot demo in warehouse.