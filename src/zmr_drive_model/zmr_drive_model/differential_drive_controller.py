
class DifferentialDriveController:
    '''Convert desired linear and angular velocities to wheel velocities'''
    def __init__(self, wheel_radius: float, track_width: float):
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0

    def update_wheel_speeds(self, linear_speed: float, angular_velocity: float) -> None:
        # Velocity of each wheel = (linear velocity ± (angular velocity * track_width / 2)) / wheel_radius
        self.left_wheel_speed  = (linear_speed - angular_velocity * self.track_width / 2) / self.wheel_radius
        self.right_wheel_speed = (linear_speed + angular_velocity * self.track_width / 2) / self.wheel_radius