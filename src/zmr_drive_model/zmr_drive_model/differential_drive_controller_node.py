import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

from .differential_drive_controller import DifferentialDriveController


WHEEL_RADIUS = 0.032 # [m]
TRACK_WIDTH = 0.2    # [m] distance between wheels

class DifferentialDriveControllerNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")
        
        cmd_vel_name            = 'cmd_vel'
        join_state_name         = 'joint_states'
        left_wheel_joint_name   = 'left_wheel_joint'  # from the urdf file
        right_wheel_joint_name  = 'right_wheel_joint' # from the urdf file

        # Subscriber to cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            cmd_vel_name,
            self.listener_callback,
            10
        )

        # Publishers for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState,
            join_state_name,
            10
        )

        # Controller
        self.dd_ctrler = DifferentialDriveController(WHEEL_RADIUS, TRACK_WIDTH)

        self._init_joint_state(left_wheel_joint_name, right_wheel_joint_name)

    def _init_joint_state(self, left_wheel_joint_name: str, right_wheel_joint_name: str) -> None:
        self.last_time = self.get_clock().now()

        self.joint_states = JointState()
        self.joint_states.header.stamp = self.get_clock().now().to_msg()
        self.joint_states.header.frame_id = ""
        self.joint_states.name = [left_wheel_joint_name, right_wheel_joint_name]
        self.joint_states.position = [0.0, 0.0] # the position is an integration of the velocity in [rad] or [m]
        self.joint_states.velocity = [0.0, 0.0]
        self.joint_states.effort = []

    def listener_callback(self, msg: Twist):
        linear_velocity = msg.linear.x
        angular_velocity = msg.angular.z
        self.dd_ctrler.update_wheel_speeds(linear_velocity, angular_velocity)
        self.publish_joint_states()
        
    def publish_joint_states(self) -> None:
        delta_time = self.get_clock().now() - self.last_time
        delta_time = delta_time.nanoseconds / 1e9
        self.get_logger().info(f"delta_time: {delta_time}")
        self.last_time = self.get_clock().now()
        
        self.joint_states.position[0] += self.joint_states.velocity[0] * delta_time
        self.joint_states.position[1] += self.joint_states.velocity[1] * delta_time
        self.joint_states.velocity = [self.dd_ctrler.left_wheel_speed, self.dd_ctrler.right_wheel_speed]
        self.joint_states.header.stamp = self.last_time.to_msg()
        self.joint_state_publisher.publish(self.joint_states)

def main(args=None):
    rclpy.init(args=args)

    differential_drive_controller = DifferentialDriveControllerNode()
    rclpy.spin(differential_drive_controller)
    differential_drive_controller.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
