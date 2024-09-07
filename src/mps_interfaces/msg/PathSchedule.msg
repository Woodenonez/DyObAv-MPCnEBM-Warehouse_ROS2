### A path schedule of a robot indicates the time and real location.

# schedule content
int16 robot_id
geometry_msgs/Point[] path_coords
float32[] path_times # if [-1.0], it is None
