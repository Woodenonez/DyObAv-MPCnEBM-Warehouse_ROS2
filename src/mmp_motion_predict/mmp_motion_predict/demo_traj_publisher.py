import os
import time

import numpy as np
from PIL import Image # type: ignore

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory # type: ignore

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid # type: ignore

from mmp_interfaces.msg import HumanTrajectory, HumanTrajectoryArray # type: ignore

from .map_tf import ScaleOffsetReverseTransform


HUMAN_SIZE = 0.2 # meter


class DemoTrajNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")

        time.sleep(5) # wait for other nodes to start

        pkg_root_dir = get_package_share_directory('mmp_motion_predict')

        self.declare_parameter('timer_period', 0.2)
        self.timer_period = self.get_parameter('timer_period').value

        self.declare_parameter('map_file_name', 'warehouse_sim_original/background.png')
        self.map_file_name = self.get_parameter('map_file_name').value
        self.map_path = os.path.join(pkg_root_dir, 'data', self.map_file_name)

        self.tf_img2real = ScaleOffsetReverseTransform(
            scale=0.1, 
            offsetx_after=-15.0, 
            offsety_after=-15.0, 
            y_reverse=(not False), 
            y_max_before=293) # global_setting_warehouse.yaml

        # Timer for publishing cmd_vel
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Publisher for human trajectory
        demo_human_trajs_name = 'human_traj_array'
        self.demo_human_trajs_publisher = self.create_publisher(
            HumanTrajectoryArray,
            demo_human_trajs_name,
            10
        )
        self._init_trajs_msg()

        # Publisher for map
        demo_map_name = 'warehouse_map'
        self.demo_map_publisher = self.create_publisher(
            OccupancyGrid,
            demo_map_name,
            10
        )
        self._init_map_msg()

    def _init_trajs_msg(self):
        self.human_trajs_msg = HumanTrajectoryArray()
        # demo_traj = [[110.0, 20.00], [111.2, 19.00], [114.3, 19.95], [116.9, 20.35], [118.8, 20.03]]
        # demo_traj = [[170.81, 119.13], [172.17, 118.32], [173.95, 117.96], [175.03, 116.91], [178.21, 118.01]]
        demo_traj = [[160.0, 103.0], [158.74, 103.94], [158.91, 106.13], [159.92, 109.16], [158.96, 110.38]]

        demo_traj = [tuple(self.tf_img2real(x, True)) for x in demo_traj]
        demo_traj_msg = HumanTrajectory()
        demo_traj_msg.traj_points = [Point(x=float(x[0]), y=float(x[1]), z=0.0) for x in demo_traj]
        self.human_trajs_msg.human_trajectories.append(demo_traj_msg)

    def _init_map_msg(self):
        map_img = Image.open(self.map_path).convert('L')
        map_img_np = np.array(map_img)
        map_img_np = np.where(map_img_np < 128, 100, 0) # the original image has 0 for obstacles (0: free, 100: occupied)
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = 'map'
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.info.resolution = 0.1
        map_msg.info.width = map_img.width
        map_msg.info.height = map_img.height
        map_msg.info.origin.position.x = -15.0
        map_msg.info.origin.position.y = -15.0
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.x = 0.0
        map_msg.info.origin.orientation.y = 0.0
        map_msg.info.origin.orientation.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        map_msg.data = map_img_np.flatten().tolist()
        self.map_msg = map_msg

    def timer_callback(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.demo_human_trajs_publisher.publish(self.human_trajs_msg)
        self.demo_map_publisher.publish(self.map_msg)
        

def main(args=None):
    rclpy.init(args=args)

    node = DemoTrajNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
