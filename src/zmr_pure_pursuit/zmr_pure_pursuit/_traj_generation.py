import math

from typing import Optional

class TrajectoryGeneration:
    def __init__(self) -> None:
        self._reference_path:Optional[list[tuple[float, float]]] = None
        self._reference_path_time:Optional[list[float]] = None

        self._ts:Optional[float] = None
        self._speed:Optional[float] = None

        self._reference_trajectory:Optional[list[tuple[float, float]]] = None
        self._reference_trajectory_time:Optional[list[float]] = None

    @property
    def ts(self):
        return self._ts
    
    @property
    def speed(self):
        return self._speed
    
    @property
    def reference_path(self):
        return self._reference_path
    
    @property
    def reference_path_time(self):
        return self._reference_path_time
    
    @property
    def reference_trajectory(self):
        return self._reference_trajectory
    
    @property
    def reference_trajectory_time(self):
        return self._reference_trajectory_time
    
    def set_sample_time(self, ts: float):
        self._ts = ts

    def set_nominal_speed(self, speed: float):
        self._speed = speed

    def set_reference_path(self, reference_path: list[tuple[float, float]]):
        self._reference_path = reference_path

    def set_reference_time(self, reference_time: list[float]):
        self._reference_path_time = reference_time

    def generate_trajectory(self, method:str='linear') -> tuple[list[tuple[float, float, float]], list[float], list[tuple[float, float]]]:
        """Generate the reference trajectory according to the reference path (and time).
        
        Sampling method:
            linear: Sample the reference path with a constant distance (step-size).
            time: Sample the reference path with a constant time interval.
        
        Returns:
            reference_trajectory: List of reference states (x, y, yaw).
            reference_trajectory_time: List of reference time.
            target_path_nodes: List of target path nodes.
        """
        if method == 'linear':
            self._reference_trajectory, self._reference_trajectory_time, target_path_nodes = self.linear_sampling()
        elif method == 'time':
            self._reference_trajectory, self._reference_trajectory_time, target_path_nodes = self.time_sampling()
        else:
            raise NotImplementedError
        return self.reference_trajectory, self.reference_trajectory_time, target_path_nodes

    def linear_sampling(self):
        """Sample the reference path with a constant distance (step-size)."""
        sampling_distance = self._speed * self._ts
        sampled_points = []
        sampled_target = []
        remainder = 0.0
        for i in range(len(self.reference_path)-1):
            p1 = self.reference_path[i]
            p2 = self.reference_path[i+1]
            sampled_points_i, remainder = self.single_linear_sampling(p1, p2, sampling_distance, remainder)
            sampled_points.extend(sampled_points_i)
            sampled_target.extend([p2] * len(sampled_points_i))
        if remainder > 0.0:
            last_point = list(self.reference_path[-1]) + [sampled_points[-1][2]]
            sampled_points.append(last_point)
            sampled_target.append(last_point)
        return sampled_points, None, sampled_target
    
    def time_sampling(self):
        """Sample the reference path with a constant time interval."""
        if self.reference_path_time is None:
            raise ValueError("The reference time is not set.")
        if len(self.reference_path_time) != len(self.reference_path):
            raise ValueError("The length of reference time and path are not equal.")
        sampled_points = []
        sampled_times = []
        sampled_target = []
        for i in range(len(self.reference_path)-1):
            p1 = self.reference_path[i]
            p2 = self.reference_path[i+1]
            t1 = self.reference_path_time[i]
            t2 = self.reference_path_time[i+1]
            num_samples = int((t2-t1) / self._ts)
            sampled_points.extend(self.single_uniform_sampling(p1, p2, num_samples)[1:])
            sampled_times.extend([t1 + self._ts * i for i in range(num_samples)][1:])
            sampled_target.extend([p2] * (num_samples-1))
        return sampled_points, sampled_times, sampled_target
    
    @staticmethod
    def single_linear_sampling(p1: tuple, p2: tuple, sample_distance: float, last_remainder=0.0):
        x1, y1 = p1
        x2, y2 = p2
        distance = math.hypot(x2-x1, y2-y1) + 1e-6
        unit_vector = ((x2-x1)/distance, (y2-y1)/distance)
        heading = math.atan2(unit_vector[1], unit_vector[0])
        
        num_samples = int((distance+last_remainder) / sample_distance)
        remainder = (distance+last_remainder) % sample_distance
        
        d_point = [x * sample_distance for x in unit_vector]
        x_s = x1 - last_remainder*unit_vector[0]
        y_s = y1 - last_remainder*unit_vector[1]
        sampled_points = [(x_s + i*d_point[0], y_s + i*d_point[1], heading) for i in range(1, num_samples+1)]
        
        return sampled_points, remainder
    
    @staticmethod
    def single_uniform_sampling(p1: tuple, p2: tuple, num_samples: int):
        x1, y1 = p1
        x2, y2 = p2
        heading = math.atan2(x2-x1, y2-y1)
        
        step_size = [(x2-x1) / (num_samples-1), (y2-y1) / (num_samples-1)]
        points = [(x1 + j*step_size[0], y1 + j*step_size[1], heading) for j in range(num_samples)]
        return points
    

if __name__ == "__main__":
    import timeit

    traj_gen = TrajectoryGeneration()
    traj_gen.set_sample_time(0.1)
    traj_gen.set_nominal_speed(1.0)
    traj_gen.set_reference_path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
    traj_gen.set_reference_time([0, 1, 2, 3])

    start_time = timeit.default_timer()
    for _ in range(1000):
        output = traj_gen.generate_trajectory(method='linear')
    end_time = timeit.default_timer()

    print("Time elapsed: {}s".format(end_time - start_time))
