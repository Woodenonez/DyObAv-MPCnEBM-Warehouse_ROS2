import math
from typing import Callable

import rclpy
from rclpy.node import Node
from rclpy.node import Subscription

from ament_index_python.packages import get_package_share_directory # type: ignore


from nav_msgs.msg import Odometry

from zmr_interfaces.msg import CurrentFutureStates # type: ignore
from zmr_interfaces.srv import GetOtherRobotStates # type: ignore


class RobotManagerNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")

        pkg_root_dir = get_package_share_directory('zmr_manager')

        # self.declare_parameter('time_period', 0.2)
        # self.time_period = self.get_parameter('time_period').value

        self.declare_parameter('robot_ids_str', '0')
        robot_ids_str: str = self.get_parameter('robot_ids_str').value # type: ignore
        self.robot_ids = [int(robot_id) for robot_id in robot_ids_str.split(',')]

        self.declare_parameter('robot_prefix', 'zmr_')
        self.robot_prefix: str = self.get_parameter('robot_prefix').value # type: ignore

        self.get_logger().info(f"Robot IDs: {self.robot_ids}, Robot Prefix: {self.robot_prefix}")

        self.odom_subscription_dict: dict[int, Subscription] = {} # type: ignore
        self.robot_state_dict: dict[int, tuple[float, float, float]] = {} # type: ignore # (x, y, theta)

        self.states_subscription_dict: dict[int, Subscription] = {} # type: ignore
        self.robot_states_dict: dict[int, list[float]] = {} # type: ignore # current_future_states

        for robot_id in self.robot_ids:
            # self.create_individual_subscription(
            #     robot_id,
            #     self.robot_id_to_odom_name(robot_id),
            #     Odometry,
            #     self.return_odom_callback(robot_id),
            #     10,
            #     self.odom_subscription_dict
            # )
            self.create_individual_subscription(
                robot_id,
                self.robot_id_to_states_name(robot_id, self.robot_prefix),
                CurrentFutureStates,
                self.return_robot_states_callback(robot_id),
                10,
                self.states_subscription_dict
            )

        robot_service_name = '/get_other_robot_states'
        self.robot_service = self.create_service(
            GetOtherRobotStates, 
            robot_service_name, 
            self.get_other_robot_states_callback
        )

        self.request_times_dict = {robot_id: 0 for robot_id in self.robot_ids}

    @staticmethod
    def _check_id(f): 
        """Decorator to check if robot_id exists"""
        def wrapper(self: 'RobotManagerNode', robot_id, *args, **kwargs):
            if robot_id not in self.robot_ids:
                raise ValueError(f'Robot {robot_id} does not exist!')
            return f(self, robot_id, *args, **kwargs)
        return wrapper

    @_check_id
    def create_individual_subscription(self, robot_id: int, topic_name: str, msg_type, callback, qos_profile, subscription_dict: dict) -> None:
        if robot_id not in self.robot_ids:
            self.get_logger().warn(f"Robot {robot_id} does not exist.")
            return
        new_subscription = self.create_subscription(
            msg_type,
            topic_name,
            callback,
            qos_profile
        )
        subscription_dict[robot_id] = new_subscription

    @_check_id
    def return_odom_callback(self, robot_id: int) -> Callable:
        """Return the callback function for the robot odometry subscriber.

        Args:
            robot_id: The robot id.

        Returns:
            callback: The callback function.
        """
        def callback(msg: Odometry) -> None:
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            theta = float(2 * math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
            self.robot_state_dict[robot_id] = (x, y, theta)
        return callback

    @_check_id
    def return_robot_states_callback(self, robot_id: int) -> Callable:
        """Return the callback function for the current_future_states subscriber.

        Args:
            robot_id: The robot id.

        Returns:
            callback: The callback function.
        """
        def callback(msg: CurrentFutureStates) -> None:
            robot_states = msg.robot_states
            self.robot_states_dict[robot_id] = list(robot_states)
        return callback

    @staticmethod
    def topic_name_to_robot_id(topic_name: str) -> int:
        """ Convert the topic name to robot_id

        Args:
            msg: Should have the format of '/{robot_prefix}X/odom' or '/{robot_prefix}X/robot_states', e.g., /zmr_0/odom

        Returns:
            robot_id: An intege "X"
        """
        robot_id_str = topic_name.split('/')[1].split('_')[-1] # /zmr_X/YYY -> zmr_X -> X
        robot_id = int(robot_id_str)
        return robot_id
    
    @staticmethod
    def robot_id_to_odom_name(robot_id: int, robot_prefix:str='zmr_') -> str:
        """ Convert the robot_id to topic name

        Args:
            robot_id: An intege "X"
            robot_prefix: The prefix of the robot name

        Returns:
            topic_name: In the format of '/{robot_prefix}X/odom'
        """
        topic_name = f'/{robot_prefix}{robot_id}/odom'
        return topic_name
    
    @staticmethod
    def robot_id_to_states_name(robot_id: int, robot_prefix:str='zmr_') -> str:
        """ Convert the robot_id to topic name

        Args:
            robot_id: An intege "X"

        Returns:
            topic_name: In the format of '/{robot_prefix}X/robot_states'
        """
        topic_name = f'/{robot_prefix}{robot_id}/robot_states'
        return topic_name

    def get_other_robot_states_callback(self, request: GetOtherRobotStates.Request, response: GetOtherRobotStates.Response):
        """Service to get the states of other robots."""
        robot_id = request.ego_robot_id
        if robot_id not in self.robot_ids:
            self.get_logger().warn(f"Robot {robot_id} does not exist.")
            return response
        self.request_times_dict[robot_id] += 1

        other_robot_states_in_order = [] # Note that this is not the correct form that MPC wants
        for id_ in list(self.robot_ids):
            if id_ != robot_id:
                try:
                    other_robot_states_in_order.extend(self.robot_states_dict[id_])
                except KeyError:
                    self.get_logger().warn(f"Robot {id_} states are not received yet.")
        response.other_robot_states_in_order = other_robot_states_in_order
        
        self.get_logger().info(f"Request (for other robot states) ID: {robot_id}. Success {self.request_times_dict[robot_id]} times.")
        return response
    
    
def main(args=None):
    rclpy.init(args=args)

    ros_node = RobotManagerNode()
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()