import os
import math
import time
from typing import cast, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from ament_index_python.packages import get_package_share_directory # type: ignore

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, Polygon, Point32 # type: ignore
from visualization_msgs.msg import Marker, MarkerArray

from map_interfaces.msg import PolygonObject # type: ignore
from mps_interfaces.srv import GetRobotSchedule, GetInflatedMap # type: ignore
from zmr_interfaces.msg import CurrentFutureStates # type: ignore
from zmr_interfaces.srv import GetOtherRobotStates # type: ignore
from mmp_interfaces.msg import MotionPredictionResult # type: ignore
from mmp_interfaces.msg import HumanTrajectory, HumanTrajectoryArray # type: ignore

from .local_traj_plan import LocalTrajPlanner
from .mpc_trajectory_tracker import TrajectoryTracker
from .mpc_trajectory_tracker import DebugInfo
from .motion_model import UnicycleModel
from .configs import MpcConfiguration, CircularRobotSpecification


class MpcControllerNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")

        time.sleep(5) # wait for other nodes to start

        pkg_root_dir = get_package_share_directory('zmr_mpc')

        self.declare_parameter('timer_period', 0.2)
        self.timer_period = self.get_parameter('timer_period').value

        # Note that this will be set to False if the service is not available
        self.declare_parameter('enable_fleet_manager', False)
        self.enable_fleet_manager = self.get_parameter('enable_fleet_manager').value

        self.declare_parameter('config_mpc_fname', 'mpc_default.yaml')
        config_mpc_fname = self.get_parameter('config_mpc_fname').value
        self.config_mpc_fpath = os.path.join(pkg_root_dir, 'config', config_mpc_fname)

        self.declare_parameter('config_robot_fname', 'robot_spec.yaml')
        config_robot_fname = self.get_parameter('config_robot_fname').value
        self.config_robot_fpath = os.path.join(pkg_root_dir, 'config', config_robot_fname)

        self.declare_parameter('robot_id', 0)
        self.robot_id = self.get_parameter('robot_id').value

        self.declare_parameter('robot_namespace', 'zmr_X')
        self.robot_namespace = self.get_parameter('robot_namespace').value
        
        self.cmd_vel_name   = 'cmd_vel'
        self.odom_name      = 'odom'
        self.dynobs_name    = '/motion_prediction_result'
        self.schedule_name  = 'robot_schedule'

        # CB group
        client_cb_group = MutuallyExclusiveCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Timer for publishing cmd_vel
        self.kt = 0
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=timer_cb_group)

        # Subscriber to odometry
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_name,
            self.odom_callback,
            10
        )

        # Subscriber to dynamic obstacles
        self.dynobs_subscription = self.create_subscription(
            MotionPredictionResult,
            self.dynobs_name,
            self.dynobs_callback,
            10
        )
        self.motion_prediction_result = []

        # Get the inflated map from the service
        map_service_name = '/get_inflated_map'
        self.get_map_client = self.create_client(
            GetInflatedMap,
            map_service_name,
            callback_group=client_cb_group
        )
        while not self.get_map_client.wait_for_service(timeout_sec=self.timer_period*10):
            self.get_logger().info('Map service not available, waiting again...')
        self.map_request = GetInflatedMap.Request()

        # Get the response from the service
        service_name = '/get_robot_schedule'
        self.get_schedule_client = self.create_client(
            GetRobotSchedule,
            service_name,
            callback_group=client_cb_group
        )
        while not self.get_schedule_client.wait_for_service(timeout_sec=self.timer_period*10):
            self.get_logger().info('Schedule service not available, waiting again...')
        self.robot_schedule_request = GetRobotSchedule.Request()

        # Get the other robot states from the service
        robot_service_name = '/get_other_robot_states'
        self.get_robot_states_client = self.create_client(
            GetOtherRobotStates,
            robot_service_name,
            callback_group=client_cb_group
        )
        while not self.get_robot_states_client.wait_for_service(timeout_sec=self.timer_period*10):
            self.get_logger().info('Robot states service not available and will be disabled.')
            self.enable_fleet_manager = False
            break
        self.robot_states_request = GetOtherRobotStates.Request()

        # Publisher for cmd_vel
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            self.cmd_vel_name,
            10
        )

        # Publisher for the current and future states of the robot
        states_msg_name = 'robot_states'
        self.robot_state_publisher = self.create_publisher(
            CurrentFutureStates,
            states_msg_name,
            10
        )
        
        # Publisher for the schedule/path visualization
        viz_msg_name = f'robot_{self.robot_id}_schedule_viz'
        self.schedule_viz_publisher = self.create_publisher(
            MarkerArray,
            viz_msg_name,
            10
        ) 
        self.color_node = (1.0, 0.0, 0.0) # red
        self.color_edge = (1.0, 1.0, 1.0) # white

        self.cfg_mpc = MpcConfiguration.from_yaml(self.config_mpc_fpath)
        self.cfg_robot = CircularRobotSpecification.from_yaml(self.config_robot_fpath)

        self.planner = LocalTrajPlanner(self.cfg_mpc.ts, self.cfg_mpc.N_hor,
                                        self.cfg_robot.lin_vel_max, verbose=False)
        self.controller = TrajectoryTracker(self.cfg_mpc, self.cfg_robot)
        self.controller.load_motion_model(UnicycleModel(self.cfg_robot.ts))

        self.odom_received = False
        self.dynobs_received = False
        self.schedule_received = False
        self.map_received = False

        self.boundary_coords = None
        self.obstacle_list = None
        self.last_pred_states = None

        self.first_message = True

        ### Send prerequisite service requests
        self.send_map_request()
        self.send_schedule_request()

    def timer_callback(self):
        if not self.odom_received:
            self.get_logger().debug("Waiting for odometry message...")
            return
        if not self.schedule_received:
            self.get_logger().debug("Waiting for schedule message...")
            return
        if not self.map_received:
            self.get_logger().debug("Waiting for map info message...")
            return
        
        self.kt += 1
        
        current_state = np.array([self.x, self.y, self.theta])
        if self.first_message:
            scheduled_path_coords = self.ref_path_coords
            goal_coord = scheduled_path_coords[-1]
            goal_coord_prev = scheduled_path_coords[-2]
            goal_heading = np.arctan2(goal_coord[1]-goal_coord_prev[1], goal_coord[0]-goal_coord_prev[0])
            goal_state = np.array([*goal_coord, goal_heading])
            self.controller.load_init_states(current_state, goal_state)
            self.first_message = False

        self.controller.set_current_state(current_state)

        if self.controller.finishing:
            ref_states, ref_speed, *_ = self.planner.get_local_ref(self.kt*self.cfg_mpc.ts, (float(current_state[0]), float(current_state[1])))
        else:
            ref_states, ref_speed, *_ = self.planner.get_local_ref(self.kt*self.cfg_mpc.ts, (float(current_state[0]), float(current_state[1])), external_ref_speed=self.controller.base_speed)
        
        if self.last_pred_states is not None:
            if np.dot(ref_states[-1, :2]-ref_states[0, :2], self.last_pred_states[-1, :2]-self.last_pred_states[0, :2]) < 0:
                ref_states = ref_states
            else:
                ref_states = ref_states*0.9 + self.last_pred_states*0.1
        self.controller.set_ref_states(ref_states, ref_speed=ref_speed)

        other_robot_states = None
        if self.enable_fleet_manager:
            robot_states_response = self.send_robot_states_request()
            ors_in_order = robot_states_response.other_robot_states_in_order
            if len(ors_in_order) > 0:
                other_robot_states = [-10] * self.cfg_mpc.ns * (self.cfg_mpc.N_hor+1) * self.cfg_mpc.Nother
                idx = 0
                idx_pred = self.cfg_mpc.ns * self.cfg_mpc.Nother
                for i in range(len(ors_in_order)//(self.cfg_mpc.ns * (self.cfg_mpc.N_hor+1))):
                    current_ors = ors_in_order[i:i+self.cfg_mpc.ns * (self.cfg_mpc.N_hor+1)]
                    other_robot_states[idx : idx+self.cfg_mpc.ns] = current_ors[:self.cfg_mpc.ns]
                    other_robot_states[idx_pred : idx_pred+self.cfg_mpc.ns*self.cfg_mpc.N_hor] = current_ors[self.cfg_mpc.ns:]
                    idx += self.cfg_mpc.ns
                    idx_pred += self.cfg_mpc.ns*self.cfg_mpc.N_hor
                # self.get_logger().info(f"Other robot states: {other_robot_states}")
            #     self.get_logger().info(f"Get")
            # else:
            #     self.get_logger().info(f"No Get")

        full_dyn_obstacle_list = None
        if self.dynobs_received:
            mu_list_list, std_list_list, conf_list_list = self.motion_prediction_result
            n_obs = max([len(x) for x in mu_list_list])
            dyn_obs_list = [[[0, 0, 0, 0, 0, 1]]*(self.cfg_mpc.N_hor+1) for _ in range(n_obs)]
            for Tt, (mu_list, std_list, conf_list) in enumerate(zip(mu_list_list, std_list_list, conf_list_list)):
                for Nn, (mu, std, conf) in enumerate(zip(mu_list, std_list, conf_list)): # at each time offset
                    dyn_obs_list[Nn][Tt] = [mu[0], mu[1], std[0], std[1], 0, conf] # for each obstacle
            full_dyn_obstacle_list = dyn_obs_list

        actions, pred_states, current_refs, debug_info = self.controller.run_step(
            static_obstacles=self.obstacle_list,
            full_dyn_obstacle_list=full_dyn_obstacle_list,
            other_robot_states=other_robot_states,
            map_updated=False)
        v, w = actions[-1]

        # self.print_debug_info(v, w, debug_info)

        cmd_vel = Twist()
        ### For differential drive robot, use the robot frame ###
        cmd_vel.linear.x = v
        cmd_vel.angular.z = w
        ### For omnidirectional robot, use the world frame ###
        # cmd_vel.linear.x = v * math.cos(self.theta)
        # cmd_vel.linear.y = v * math.sin(self.theta)
        # cmd_vel.angular.z = w

        cur_future_states = [self.x, self.y, self.theta] + [x for subarray in pred_states for x in subarray]
        cur_future_states = [float(x) for x in cur_future_states]
        robot_states = CurrentFutureStates()
        robot_states.robot_states = cur_future_states

        self.cmd_vel_publisher.publish(cmd_vel)
        self.robot_state_publisher.publish(robot_states)
        self.schedule_viz_publisher.publish(
            self.robot_path_to_vis_msg(
                self.ref_path_coords.copy(),
                current_ref=current_refs.tolist(),
                pred_states=pred_states)
        )

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = 2 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.odom_received = True

    def dynobs_callback(self, msg: MotionPredictionResult):
        mu_list_list = []
        mu_list_list_msg:list[HumanTrajectoryArray] = msg.mu_list_list
        for mu_list_msg in mu_list_list_msg:
            mu_list_msg_seq:list[HumanTrajectory] = mu_list_msg.human_trajectories
            mu_list = []
            for mus_at_time_msg in mu_list_msg_seq:
                traj_pts:list[Point] = mus_at_time_msg.traj_points
                mus_at_time = [(pt.x, pt.y) for pt in traj_pts]
                mu_list += mus_at_time
            mu_list_list.append(mu_list)

        std_list_list = []
        std_list_list_msg:list[HumanTrajectoryArray] = msg.std_list_list
        for std_list_msg in std_list_list_msg:
            std_list_msg_seq:list[HumanTrajectory] = std_list_msg.human_trajectories
            std_list = []
            for stds_at_time_msg in std_list_msg_seq:
                traj_pts:list[Point] = stds_at_time_msg.traj_points # type: ignore
                stds_at_time = [(pt.x, pt.y) for pt in traj_pts]
                std_list += stds_at_time
            std_list_list.append(std_list)

        conf_list_list = []
        conf_list_list_msg:list[HumanTrajectoryArray] = msg.conf_list_list
        for conf_list_msg in conf_list_list_msg:
            conf_list_msg_seq:list[HumanTrajectory] = conf_list_msg.human_trajectories
            conf_list = []
            for confs_at_time_msg in conf_list_msg_seq:
                traj_pts:list[Point] = confs_at_time_msg.traj_points # type: ignore
                confs_at_time = [pt.x for pt in traj_pts]
                conf_list += confs_at_time
            conf_list_list.append(conf_list)

        self.motion_prediction_result = (mu_list_list, std_list_list, conf_list_list)
        self.dynobs_received = True


    def send_map_request(self) -> GetInflatedMap.Response:
        self.map_request.robot_id = self.robot_id
        future = self.get_map_client.call_async(self.map_request)
        rclpy.spin_until_future_complete(self, future)
        self.map_response = future.result()
        self.map_response = cast(GetInflatedMap.Response, self.map_response)

        boundary_coords = self.polygon_to_tuples(self.map_response.inflated_map.boundary.polygon)
        obstacle_list = []
        obstacle_objects:list[PolygonObject] = self.map_response.inflated_map.obstacle_list.polygon_objects
        for obstacle_obj in obstacle_objects:
            obstacle_list.append(self.polygon_to_tuples(obstacle_obj.polygon))
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.map_received = True
        return self.map_response

    def send_schedule_request(self) -> GetRobotSchedule.Response:
        self.robot_schedule_request.robot_id = self.robot_id
        self.robot_schedule_request.current_time = float(self.get_clock().now().nanoseconds / 1e9)
        future = self.get_schedule_client.call_async(self.robot_schedule_request)
        rclpy.spin_until_future_complete(self, future)
        self.robot_schedule_response = future.result()
        self.robot_schedule_response = cast(GetRobotSchedule.Response, self.robot_schedule_response)

        path_coords:list[Point] = self.robot_schedule_response.path_schedule.path_schedule.path_coords
        self.ref_path_coords = []
        for point in path_coords:
            self.ref_path_coords.append((point.x, point.y))
        self.planner.load_path(self.ref_path_coords, None, self.cfg_robot.lin_vel_max) # ignore the path times
        self.schedule_received = True
        return self.robot_schedule_response

    def send_robot_states_request(self) -> GetOtherRobotStates.Response:
        if not self.get_robot_states_client.wait_for_service(timeout_sec=self.timer_period/5):
            self.get_logger().debug(f'Robot states service not available for robot {self.robot_id}.')
            return GetOtherRobotStates.Response()
        self.robot_states_request.ego_robot_id = self.robot_id
        self.robot_states_response = self.get_robot_states_client.call(self.robot_states_request)
        return self.robot_states_response

    def print_debug_info(self, v: float, w: float, debug_info: DebugInfo):
        self.get_logger().info(f"Current state: {(round(self.x, 2), round(self.y, 2), round(self.theta, 2))} -> Target position: {self.planner.current_target_node}")
        self.get_logger().info(f"Mode: {self.controller.mode}, v: {round(v, 2)}, w: {round(w, 2)}")
        self.get_logger().info(f"Cost: {round(debug_info['cost'], 2)}. Solve time: {round(debug_info['step_runtime'], 4)} s")

    def robot_path_to_vis_msg(self,
                              points: list[tuple[float, float]],
                              current_ref:Optional[list[tuple[float, float]]]=None,
                              pred_states:Optional[list[tuple[float, float]]]=None,
                              name_space:str="robot_schedule_ns", 
                              id_start:int=0) -> MarkerArray:
        """Convert a PathSchedule message to a MarkerArray message"""
        marker_schedule_msg = MarkerArray()
        if self.ref_path_coords is None:
            return marker_schedule_msg
        
        marker_id = id_start

        marker_id += 1
        marker_point_msg_1 = self.tuples_to_vis_msg(points, Marker.POINTS, self.color_node, marker_id, scale=0.2)
        marker_point_msg_1.header.frame_id = "map"
        marker_point_msg_1.ns = name_space
        marker_schedule_msg.markers.append(marker_point_msg_1)

        marker_id += 1
        marker_line_msg_1 = self.tuples_to_vis_msg(points, Marker.LINE_STRIP, self.color_edge, marker_id, scale=0.05)
        marker_line_msg_1.header.frame_id = "map"
        marker_line_msg_1.ns = name_space
        marker_schedule_msg.markers.append(marker_line_msg_1)

        if current_ref is not None:
            marker_id += 1
            # color: magenta (0.8, 0.0, 0.8)
            marker_point_msg_2 = self.tuples_to_vis_msg(current_ref, Marker.POINTS, (0.8, 0.0, 0.8), marker_id, scale=0.1)
            marker_point_msg_2.header.frame_id = "map"
            marker_point_msg_2.ns = name_space
            marker_schedule_msg.markers.append(marker_point_msg_2)

        if pred_states is not None:
            marker_id += 1
            # color: yellow (1.0, 1.0, 0.0)
            marker_point_msg_3 = self.tuples_to_vis_msg(pred_states, Marker.POINTS, (1.0, 1.0, 0.0), marker_id, scale=0.1)
            marker_point_msg_3.header.frame_id = "map"
            marker_point_msg_3.ns = name_space
            marker_schedule_msg.markers.append(marker_point_msg_3)

        return marker_schedule_msg
    
    @staticmethod
    def tuples_to_vis_msg(points: list[tuple[float, float]],
                          marker_type,
                          color: tuple[float, float, float],
                          marker_id: int,
                          scale:float=0.05) -> Marker:
        """Convert points to a Marker message"""
        marker_msg = Marker()
        marker_msg.id = marker_id
        marker_msg.type = marker_type
        marker_msg.action = Marker.ADD

        marker_msg.pose.position.x = 0.0
        marker_msg.pose.position.y = 0.0
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0

        marker_msg.scale.x = scale
        marker_msg.scale.y = scale
        marker_msg.scale.z = scale

        marker_msg.color.a = 1.0
        marker_msg.color.r = color[0]
        marker_msg.color.g = color[1]
        marker_msg.color.b = color[2]

        marker_msg.lifetime.sec = 0 # forever
        marker_msg.frame_locked = False # not locked to a frame
        [marker_msg.points.append(Point(x=pt[0], y=pt[1])) for pt in points]
        return marker_msg
    
    @staticmethod
    def polygon_to_tuples(polygon: Polygon) -> list[tuple[float, float]]:
        """Convert a Polygon message to a list of tuples"""
        vertices = []
        points:list[Point32] = polygon.points
        for point in points:
            vertices.append((point.x, point.y))
        return vertices


def main(args=None):
    rclpy.init(args=args)

    node = MpcControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
