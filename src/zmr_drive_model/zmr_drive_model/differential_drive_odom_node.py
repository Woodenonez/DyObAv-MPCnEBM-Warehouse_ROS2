import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

import math
import numpy as np

from . import forward_kinematics
from .differential_drive_controller import DifferentialDriveController
from .inverse_drive_model import SimpleInverseDriveModel


WHEEL_RADIUS = 0.032 # [m]
TRACK_WIDTH = 0.2    # [m] distance between wheels

class DiffDriveOdometryNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")

        self.declare_parameter('timer_period', 0.05)
        self.timer_period = self.get_parameter('timer_period').value

        self.declare_parameter('keep_cmd_vel', True)
        self.keep_cmd_vel = self.get_parameter('keep_cmd_vel').value

        self.declare_parameter('robot_namespace', 'zmr_X')
        self.robot_namespace = self.get_parameter('robot_namespace').value

        self.declare_parameter('init_x', 0.0)
        init_x = self.get_parameter('init_x').value

        self.declare_parameter('init_y', 0.0)
        init_y = self.get_parameter('init_y').value

        self.declare_parameter('init_theta', 0.0)
        init_theta = self.get_parameter('init_theta').value
        
        self.cmd_vel_name           = 'cmd_vel'
        self.odom_name              = 'odom'
        self.odom_child_frame_name  = 'base_footprint'
        self.joint_states_name      = 'joint_states'
        self.left_wheel_joint_name  = str(self.robot_namespace)+'/'+'left_wheel_joint'  # from the urdf file
        self.right_wheel_joint_name = str(self.robot_namespace)+'/'+'right_wheel_joint' # from the urdf file

        self.tf_odom_name = str(self.robot_namespace)+'/'+'odom'
        self.tf_odom_child_frame_name = str(self.robot_namespace)+'/'+'base_footprint'

        # Timer for publishing joint states and odometry
        if self.keep_cmd_vel:
            self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Subscriber to cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            self.cmd_vel_name,
            self.cmd_vel_callback,
            10
        )

        # Publisher for odometry
        self.odom_publisher = self.create_publisher(
            Odometry,
            self.odom_name,
            10
        )

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState,
            self.joint_states_name,
            10
        )

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)

        self.kinematics = forward_kinematics.unicycle_model
        self.drive_model = DifferentialDriveController(WHEEL_RADIUS, TRACK_WIDTH) # position -> wheel speed
        self.inverse_drive_model = SimpleInverseDriveModel(WHEEL_RADIUS, TRACK_WIDTH) # wheel speed -> position

        self.x = init_x
        self.y = init_y
        self.theta = init_theta

        self.last_time = self.get_clock().now()
        self._init_joint_state()
        self._init_odom()
        self._init_transform()

    def _init_joint_state(self) -> None:
        self.joint_states = JointState()
        self.joint_states.header.stamp = self.get_clock().now().to_msg()
        self.joint_states.header.frame_id = ""
        self.joint_states.name = [self.left_wheel_joint_name, self.right_wheel_joint_name]
        self.joint_states.position = [0.0, 0.0] # the position is an integration of the velocity in [rad] or [m]
        self.joint_states.velocity = [0.0, 0.0]
        self.joint_states.effort = []

    def _init_odom(self) -> None:
        self.odom = Odometry()
        self.odom.header.stamp = self.get_clock().now().to_msg()
        self.odom.header.frame_id = self.odom_name
        self.odom.child_frame_id = self.odom_child_frame_name
        self.odom.pose.pose.position.x = self.x
        self.odom.pose.pose.position.y = self.y
        self.odom.pose.pose.orientation.z = math.sin(self.theta / 2)
        self.odom.pose.pose.orientation.w = math.cos(self.theta / 2)
        self.odom.twist.twist.linear.x = 0.0
        self.odom.twist.twist.linear.y = 0.0
        self.odom.twist.twist.angular.z = 0.0

    def _init_transform(self) -> None:
        self.transform = TransformStamped()
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.transform.header.frame_id = self.tf_odom_name
        self.transform.child_frame_id = self.tf_odom_child_frame_name

    def _get_cmd_vel_real(self, msg: Twist) -> tuple[float, float]:
        """Get the real cmd_vel from the desired cmd_vel:
        `cmd_vel -> [drive_model] -> wheel_speed -> [inverse_drive_model] -> cmd_vel_real`
        
        Args:
            msg: desired cmd_vel
            
        Returns:
            linear_velocity: The real linear velocity
            angular_velocity: The real angular velocity
        """
        desired_linear_velocity = msg.linear.x
        desired_angular_velocity = msg.angular.z
        self.drive_model.update_wheel_speeds(desired_linear_velocity, desired_angular_velocity)
        left_wheel_speed = self.drive_model.left_wheel_speed
        right_wheel_speed = self.drive_model.right_wheel_speed
        linear_velocity, angular_velocity = self.inverse_drive_model.get_motion_profile(left_wheel_speed, right_wheel_speed)
        return linear_velocity, angular_velocity

    def timer_callback(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Keep the current cmd_vel
        linear_velocity = math.hypot(self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y)
        angular_velocity = self.odom.twist.twist.angular.z

        # Calculate odometry
        next_state = self.kinematics(np.array([self.x, self.y, self.theta]), np.array([linear_velocity, angular_velocity]), dt)
        self.x = next_state[0]
        self.y = next_state[1]
        self.theta = next_state[2]

        # Update and publish odometry
        self.odom.header.stamp = current_time.to_msg()
        self.odom.pose.pose.position.x = self.x
        self.odom.pose.pose.position.y = self.y
        self.odom.pose.pose.orientation.z = math.sin(self.theta / 2)
        self.odom.pose.pose.orientation.w = math.cos(self.theta / 2)
        self.odom.twist.twist.linear.x = linear_velocity * math.cos(self.theta)
        self.odom.twist.twist.linear.y = linear_velocity * math.sin(self.theta)
        self.odom.twist.twist.angular.z = angular_velocity
        self.odom_publisher.publish(self.odom)

        # Update and publish joint states
        self.joint_states.header.stamp = current_time.to_msg()
        self.joint_states.position[0] += self.joint_states.velocity[0] * dt
        self.joint_states.position[1] += self.joint_states.velocity[1] * dt
        self.joint_states.velocity = [self.drive_model.left_wheel_speed, self.drive_model.right_wheel_speed]
        self.joint_state_publisher.publish(self.joint_states)

        # Update and publish transform
        self.transform.header.stamp = current_time.to_msg()
        self.transform.transform.translation.x = self.x
        self.transform.transform.translation.y = self.y
        self.transform.transform.rotation.z = math.sin(self.theta / 2)
        self.transform.transform.rotation.w = math.cos(self.theta / 2)
        self.tf_broadcaster.sendTransform(self.transform)

        # For future use/debug
        dt_difference = float(dt - self.timer_period) # check if the timer is working properly

    def cmd_vel_callback(self, msg: Twist):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Get the real cmd_vel
        linear_velocity, angular_velocity = self._get_cmd_vel_real(msg)

        # Calculate odometry
        next_state = self.kinematics(np.array([self.x, self.y, self.theta]), np.array([linear_velocity, angular_velocity]), dt)
        self.x = next_state[0]
        self.y = next_state[1]
        self.theta = next_state[2]

        # Update and publish odometry
        self.odom.header.stamp = current_time.to_msg()
        self.odom.pose.pose.position.x = self.x
        self.odom.pose.pose.position.y = self.y
        self.odom.pose.pose.orientation.z = math.sin(self.theta / 2)
        self.odom.pose.pose.orientation.w = math.cos(self.theta / 2)
        self.odom.twist.twist.linear.x = linear_velocity * math.cos(self.theta)
        self.odom.twist.twist.linear.y = linear_velocity * math.sin(self.theta)
        self.odom.twist.twist.angular.z = angular_velocity
        self.odom_publisher.publish(self.odom)

        # Update and publish joint states
        self.joint_states.header.stamp = current_time.to_msg()
        self.joint_states.position[0] += self.joint_states.velocity[0] * dt
        self.joint_states.position[1] += self.joint_states.velocity[1] * dt
        self.joint_states.velocity = [self.drive_model.left_wheel_speed, self.drive_model.right_wheel_speed]
        self.joint_state_publisher.publish(self.joint_states)

        # Update and publish transform
        self.transform.header.stamp = current_time.to_msg()
        self.transform.transform.translation.x = self.x
        self.transform.transform.translation.y = self.y
        self.transform.transform.rotation.z = math.sin(self.theta / 2)
        self.transform.transform.rotation.w = math.cos(self.theta / 2)
        self.tf_broadcaster.sendTransform(self.transform)


def main(args=None):
    rclpy.init(args=args)

    node = DiffDriveOdometryNode()
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
