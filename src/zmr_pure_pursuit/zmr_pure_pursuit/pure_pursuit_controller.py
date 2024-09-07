import math
from dataclasses import dataclass

from ._traj_generation import TrajectoryGeneration
from .configs import CircularRobotSpecification # type: ignore


@dataclass
class DebugInfo:
    current_state: tuple[float, float, float]
    target_index: int
    target_position: tuple[float, float]
    current_yaw: float
    target_yaw: float
    speed_discount: float
    angular_discount: float
    v: float
    w: float
    terminal: bool

    def __post_init__(self):
        self.current_state = [round(x, 2) for x in self.current_state]
        self.target_position = [round(x, 2) for x in self.target_position]
        self.current_yaw = round(self.current_yaw/3.14*180, 2)
        self.target_yaw = round(self.target_yaw/3.14*180, 2)
        self.speed_discount = round(self.speed_discount, 2)
        self.angular_discount = round(self.angular_discount, 2)
        self.v = round(self.v, 2)
        self.w = round(self.w, 2)


class PurePursuitController:
    def __init__(self, config: CircularRobotSpecification) -> None:
        self._current_idx:int = 0
        self.config = config

        self.traj_gen = TrajectoryGeneration()
        self.traj_gen.set_sample_time(config.ts)
        self.traj_gen.set_nominal_speed(config.lin_vel_max)

        self._ready = False

    @property
    def current_idx(self) -> int:
        return self._current_idx

    @property
    def ref_traj(self) -> list[tuple[float, float, float]]:
        return self._ref_traj
    
    @property
    def ref_target(self) -> list[tuple[float, float]]:
        return self._ref_target
    
    @classmethod
    def from_config_file(cls, config_file_path: str):
        config = CircularRobotSpecification.from_yaml(config_file_path)
        return cls(config)

    def set_reference_path(self, reference_path: list[tuple[float, float]]):
        self.traj_gen.set_reference_path(reference_path)
        self._ref_traj, _, self._ref_target = self.traj_gen.generate_trajectory(method='linear')
        self._ready = True

    def get_control_action(self, current_state, idx_check_range:int=10) -> tuple[float, float, DebugInfo]:
        """Get the control action (v, w) from the reference trajectory.
        
        Args:
            current_state: The current state of the robot (x, y, yaw).
            idx_check_range: The range to search for the closest node on the reference trajectory.
        """
        if not self._ready:
            raise RuntimeError("Reference path is not set yet.")
        index_look_ahead = 2

        lb_idx = max(self._current_idx-index_look_ahead-2, 0)
        ub_idx = min(self._current_idx+idx_check_range, len(self._ref_traj)-1)
        
        distances = [math.hypot(current_state[0]-x[0], current_state[1]-x[1]) for x in self._ref_traj[lb_idx:ub_idx]]
        self._current_idx = lb_idx + distances.index(min(distances)) # this is the index of the docking point on the ref_traj
        self._current_idx = min(self._current_idx+index_look_ahead, len(self._ref_traj)-1)
        
        current_target = self._ref_target[self._current_idx]

        x, y, yaw = current_state
        x_target, y_target = current_target[:2]
        yaw_target = math.atan2(y_target-y, x_target-x)

        if (self._current_idx == len(self._ref_traj)-1) and (math.hypot(x_target-x, y_target-y) < self.config.ang_vel_max*self.config.ts):
            v = 0.0
            w = 0.0
            speed_discount = math.nan
            angular_discount = math.nan
            terminal = True
        else:
            diff_angle_deg = abs(yaw - yaw_target)/math.pi*180
            speed_discount = 1.0 - min(diff_angle_deg, 45) / 45
            speed_discount = min(max(speed_discount, 0.0), 1.0)
            angular_discount = 1.0 - speed_discount

            v = self.traj_gen.speed * speed_discount
            if yaw_target - yaw > 0:
                w = self.config.ang_vel_max * angular_discount
            else:
                w = -self.config.ang_vel_max * angular_discount
            terminal = False

        debug_info = DebugInfo(
            current_state=current_state,
            target_index=self._current_idx,
            target_position=current_target,
            current_yaw=yaw,
            target_yaw=yaw_target,
            speed_discount=speed_discount,
            angular_discount=angular_discount,
            v=v,
            w=w,
            terminal=terminal
        )

        return v, w, debug_info