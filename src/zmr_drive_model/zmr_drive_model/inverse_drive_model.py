
class SimpleInverseDriveModel:
    '''A simple inverse drive model for differential drive robots.'''
    def __init__(self, wheel_radius: float, track_width: float):
        self.wheel_radius = wheel_radius
        self.track_width = track_width

    def get_motion_profile(self, left_wheel_speed: float, right_wheel_speed: float) -> tuple:
        '''
        Returns:
            linear_velocity: float
            angular_velocity: float

        Notes:
            wheel_speed = (linear_velocity ± (angular_velocity * track_width / 2)) / wheel_radius
        '''
        linear_velocity = (left_wheel_speed + right_wheel_speed) * self.wheel_radius / 2
        angular_velocity = (right_wheel_speed - left_wheel_speed) * self.wheel_radius / self.track_width
        return linear_velocity, angular_velocity
