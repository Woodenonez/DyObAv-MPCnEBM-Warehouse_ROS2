import os
import time

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory # type: ignore

from geometry_msgs.msg import Point, PoseStamped

from mmp_interfaces.msg import HumanTrajectory, HumanTrajectoryArray # type: ignore


HUMAN_SIZE = 0.2 # meter


class ActorTrajNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")

        time.sleep(5) # wait for other nodes to start

        pkg_root_dir = get_package_share_directory('mmp_motion_predict')

        self.declare_parameter('timer_period', 0.2)
        self.timer_period = self.get_parameter('timer_period').value

        # Timer for publishing human trajectories
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Subscriber to actor poses
        self.actor_pose_received = False

        actor1_pose_name = 'actor1_pose'
        self.actor1_pose_subscription = self.create_subscription(
            PoseStamped,
            actor1_pose_name,
            self.actor1_pose_callback,
            10
        )
        self.actor1_traj = []
        
        actor2_pose_name = 'actor2_pose'
        self.actor2_pose_subscription = self.create_subscription(
            PoseStamped,
            actor2_pose_name,
            self.actor2_pose_callback,
            10
        )
        self.actor2_traj = []

        # Publisher for human trajectory
        human_trajs_name = 'human_traj_array'
        self.human_trajs_publisher = self.create_publisher(
            HumanTrajectoryArray,
            human_trajs_name,
            10
        )

    def timer_callback(self):
        if not self.actor_pose_received:
            self.get_logger().debug("Waiting for actor pose message...")
            return
        
        self.assemble_trajs_msg()
        self.human_trajs_publisher.publish(self.human_trajs_msg)
        
    def actor1_pose_callback(self, msg: PoseStamped):
        acotr1_pose = [float(msg.pose.position.x), float(msg.pose.position.y)]
        self.actor1_traj.append(acotr1_pose)
        self.actor_pose_received = True

    def actor2_pose_callback(self, msg: PoseStamped):
        acotr2_pose = [float(msg.pose.position.x), float(msg.pose.position.y)]
        self.actor2_traj.append(acotr2_pose)
        self.actor_pose_received = True

    def assemble_trajs_msg(self):
        self.human_trajs_msg = HumanTrajectoryArray()

        if self.actor1_traj:
            actor1_traj_msg = HumanTrajectory()
            actor1_traj_msg.traj_points = [Point(x=float(x[0]), y=float(x[1]), z=0.0) for x in self.actor1_traj]
            self.human_trajs_msg.human_trajectories.append(actor1_traj_msg)

        if self.actor2_traj:
            actor2_traj_msg = HumanTrajectory()
            actor2_traj_msg.traj_points = [Point(x=float(x[0]), y=float(x[1]), z=0.0) for x in self.actor2_traj]
            self.human_trajs_msg.human_trajectories.append(actor2_traj_msg)

        

def main(args=None):
    rclpy.init(args=args)

    node = ActorTrajNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
