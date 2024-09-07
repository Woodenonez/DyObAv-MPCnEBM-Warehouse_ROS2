import time
import threading

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

@DeprecationWarning
class RotateWheelNode(Node):
    # XXX This class is deprecated because it is not updated to include the robot namespace.
    def __init__(self):
        super().__init__("RotateWheel")
        self.get_logger().info(f"{self.__class__.__name__} init..")

        self.joint_state_publisher = self.create_publisher(
            JointState,
            "joint_states",
            10
        )

        self._init_joint_state()
        self.joint_state_publish_rate = self.create_rate(30)
        self.thread_ = threading.Thread(target=self._thread_publish_joint_state)
        self.thread_.start()

    def _thread_publish_joint_state(self) -> None:
        last_update_time = time.time()
        while rclpy.ok():
            delta_time = time.time() - last_update_time
            last_update_time = time.time()
            
            self.joint_states.position[0] += self.joint_states.velocity[0] * delta_time
            self.joint_states.position[1] += self.joint_states.velocity[1] * delta_time
            self.joint_states.velocity = self.joint_speeds
            self.joint_states.header.stamp = self.get_clock().now().to_msg()
            self.joint_state_publisher.publish(self.joint_states)
            self.joint_state_publish_rate.sleep()

    def _init_joint_state(self) -> None:
        self.joint_speeds = [0.0, 0.0]

        self.joint_states = JointState()
        self.joint_states.header.stamp = self.get_clock().now().to_msg()
        self.joint_states.header.frame_id = ""
        self.joint_states.name = ['left_wheel_joint','right_wheel_joint']
        self.joint_states.position = [0.0, 0.0]
        self.joint_states.velocity = self.joint_speeds
        self.joint_states.effort = []

    def update_joint_speeds(self, left_speed: float, right_speed: float) -> None:
        self.joint_speeds = [left_speed, right_speed]


def main(args=None):
    rclpy.init(args=args)
    rotate_wheel = RotateWheelNode() 
    rotate_wheel.update_joint_speeds(5.0, -5.0)
    rclpy.spin(rotate_wheel)
    rotate_wheel.destroy_node()
    rclpy.shutdown()