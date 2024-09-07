import os
import math
import time
from typing import cast

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory # type: ignore

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, Point32
from visualization_msgs.msg import Marker, MarkerArray

from mps_interfaces.msg import PathSchedule, PathScheduleStamped # type: ignore
from mps_interfaces.srv import GetRobotSchedule # type: ignore

from .pure_pursuit_controller import PurePursuitController
from .pure_pursuit_controller import DebugInfo


class PurePursuitControllerNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")

        time.sleep(5) # wait for other nodes to start

        pkg_root_dir = get_package_share_directory('zmr_pure_pursuit')

        self.declare_parameter('timer_period', 0.2)
        self.timer_period = self.get_parameter('timer_period').value

        self.declare_parameter('config_file_name', './config.yaml')
        self.config_file_name = self.get_parameter('config_file_name').value
        self.config_file_path = os.path.join(pkg_root_dir, 'config', self.config_file_name)

        self.declare_parameter('robot_id', 0)
        self.robot_id = self.get_parameter('robot_id').value
        
        self.cmd_vel_name   = 'cmd_vel'
        self.odom_name      = 'odom'

        # Timer for publishing cmd_vel
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Subscriber to odometry
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_name,
            self.odom_callback,
            10
        )

        # Get the response from the service
        service_name = '/get_robot_schedule'
        self.get_schedule_client = self.create_client(
            GetRobotSchedule,
            service_name
        )
        while not self.get_schedule_client.wait_for_service(timeout_sec=self.timer_period*10):
            self.get_logger().info('service not available, waiting again...')
        self.robot_schedule_request = GetRobotSchedule.Request()

        # Publisher for cmd_vel
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            self.cmd_vel_name,
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

        self.pp_ctrl = PurePursuitController.from_config_file(self.config_file_path)
        self.odom_received = False
        self.schedule_received = False
        self.ref_path_coords = None


    def timer_callback(self):
        if not self.odom_received:
            self.get_logger().debug("Waiting for odometry message...")
            return
        if not self.schedule_received:
            self.get_logger().debug("Waiting for schedule message...")
            return
        
        v, w, debug_info = self.pp_ctrl.get_control_action((self.x, self.y, self.theta))
        self.print_debug_info(debug_info)

        cmd_vel = Twist()
        ### For differential drive robot, use the robot frame ###
        cmd_vel.linear.x = v
        cmd_vel.angular.z = w
        ### For omnidirectional robot, use the world frame ###
        # cmd_vel.linear.x = v * math.cos(self.theta)
        # cmd_vel.linear.y = v * math.sin(self.theta)
        # cmd_vel.angular.z = w

        self.cmd_vel_publisher.publish(cmd_vel)
        self.schedule_viz_publisher.publish(self.robot_path_to_vis_msg())

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = 2 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        self.odom_received = True

    def send_request(self) -> GetRobotSchedule.Response:
        self.robot_schedule_request.robot_id = self.robot_id
        self.robot_schedule_request.current_time = float(self.get_clock().now().nanoseconds / 1e9)
        self.robot_shcedule_future = self.get_schedule_client.call_async(self.robot_schedule_request)
        rclpy.spin_until_future_complete(self, self.robot_shcedule_future)
        self.robot_schedule_response = self.robot_shcedule_future.result()
        self.robot_schedule_response = cast(GetRobotSchedule.Response, self.robot_schedule_response)

        path_coords:list[Point] = self.robot_schedule_response.path_schedule.path_schedule.path_coords
        self.ref_path_coords = []
        for point in path_coords:
            self.ref_path_coords.append((point.x, point.y))
        self.pp_ctrl.set_reference_path(self.ref_path_coords)
        self.schedule_received = True
        return self.robot_schedule_response

    def print_debug_info(self, debug_info: DebugInfo):
        current_time = float(self.get_clock().now().nanoseconds / 1e9)
        elapsed_time = current_time - self.robot_schedule_request.current_time
        self.get_logger().info(f"=============== Elapsed time: {round(elapsed_time, 3)}. Finished: {debug_info.terminal} ===============")
        self.get_logger().info(f"Current state: {debug_info.current_state} -> Target index/position: {debug_info.target_index}/{debug_info.target_position}")
        self.get_logger().info(f"Current yaw: {debug_info.current_yaw} -> Target yaw: {debug_info.target_yaw}")
        self.get_logger().info(f"Speed/angular discount: {debug_info.speed_discount}/{debug_info.angular_discount}")
        self.get_logger().info(f"v: {debug_info.v}, w: {debug_info.w}")

    def robot_path_to_vis_msg(self,
                              name_space:str="robot_schedule_ns", 
                              id_start:int=0) -> MarkerArray:
        """Convert a PathSchedule message to a MarkerArray message"""
        marker_schedule_msg = MarkerArray()
        if self.ref_path_coords is None:
            return marker_schedule_msg
        
        marker_id = id_start
        
        points = self.ref_path_coords.copy()

        marker_id += 1
        marker_point_msg = self.tuples_to_vis_msg(points, Marker.POINTS, self.color_node, marker_id, scale=0.2)
        marker_point_msg.header.frame_id = "map"
        marker_point_msg.ns = name_space
        marker_schedule_msg.markers.append(marker_point_msg)

        marker_id += 1
        marker_line_msg = self.tuples_to_vis_msg(points, Marker.LINE_STRIP, self.color_edge, marker_id, scale=0.05)
        marker_line_msg.header.frame_id = "map"
        marker_line_msg.ns = name_space
        marker_schedule_msg.markers.append(marker_line_msg)

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


def main(args=None):
    rclpy.init(args=args)

    node = PurePursuitControllerNode()
    node.send_request()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
