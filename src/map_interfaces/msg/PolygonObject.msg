### Polygon with 2D points forming the object

# Identifier of the object
int16 object_id
string object_name

# Object date
geometry_msgs/Polygon polygon
geometry_msgs/Point32 centroid

# Object extra information
map_interfaces/ObjectShapeType object_shape_t # Type of object, 1 = polygon.
map_interfaces/ObjectMotionType object_motion_t # Type of object, e.g., static, dynamic, etc.
map_interfaces/ObjectSemanticType object_semantic_t # Type of object, e.g., car, human, etc.

