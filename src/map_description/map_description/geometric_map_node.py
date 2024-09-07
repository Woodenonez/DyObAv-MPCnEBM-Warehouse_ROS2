import os

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from ament_index_python.packages import get_package_share_directory # type: ignore

from geometry_msgs.msg import Polygon, Point32, Point # type: ignore
from map_interfaces.msg import GeometricMap as MapMsg # type: ignore
from map_interfaces.msg import PolygonObject, PolygonObjectArray, PolygonObjectArrayStamped # type: ignore
from map_interfaces.msg import ObjectShapeType, ObjectMotionType, ObjectSemanticType # type: ignore
from visualization_msgs.msg import Marker, MarkerArray # type: ignore

from .map_geometric import GeometricMap
from .geometry_plain import PlainPolygon

class GeometricMapNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")

        pkg_root_dir = get_package_share_directory('map_description')

        self.geometric_map_publisher = self.create_publisher(
            MapMsg, 
            'geometry_map', 
            10
        )

        self.geometric_map_vis_publisher = self.create_publisher(
            MarkerArray, 
            'geometry_map_vis', 
            10
        )

        self.declare_parameter('timer_period', 0.2)
        self.timer_period = self.get_parameter('timer_period').value
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.declare_parameter('map_file_name', 'map.json')
        self.map_file_name = self.get_parameter('map_file_name').value
        self.json_path = os.path.join(pkg_root_dir, 'data', self.map_file_name)

        if not os.path.exists(self.json_path):
            self.get_logger().warn(f"Map file {self.json_path} does not exist.")
        else:
            self._init_map_info(self.json_path)

        self.boundary_color = (0.0, 0.0, 0.0) # black
        self.obstacle_color = (1.0, 0.0, 0.0) # red

    def _init_map_info(self, json_path: str):
        """Build map object and related message from json file
        
        Notes:
            The json file should be in the following format:
            ```
            {
                "boundary_coords": list[list[float]],
                "obstacle_list": list[list[tuple]],
                "obstacle_dict": list[{"id_": int, "name": str, "vertices": list[list[tuple]]}]
            }
            ```
            Prefered key is "obstacle_dict".

        Attributes:
            geo_map: GeometricMap object
            geo_map_msg_stamped: PolygonObjectArrayStamped message
        """
        self.geo_map = GeometricMap.from_json(json_path)
        self.geo_map_msg_stamped = MapMsg()
        self.geo_map_msg_stamped.header.frame_id = 'map'
        self.geo_map_msg_stamped.header.stamp = self.get_clock().now().to_msg()
        boundary_msg, obstacle_msg = self.geo_map_to_msg(self.geo_map)
        self.geo_map_msg_stamped.boundary = boundary_msg
        self.geo_map_msg_stamped.obstacle_list = obstacle_msg
    
    def timer_callback(self):
        if not os.path.exists(self.json_path):
            self.get_logger().warn(f"Map file {self.json_path} does not exist.")
            return

        ### Map message: PolygonObjectArrayStamped ###
        self.geo_map_msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.geometric_map_publisher.publish(self.geo_map_msg_stamped)
        self.get_logger().debug("Map published.")

        ### Map visualization message: MarkerArray ###
        map_vis_msg = self.geo_map_to_vis_msg(self.geo_map_msg_stamped, 
                                              boundary_color=self.boundary_color, 
                                              obstacle_color=self.obstacle_color)
        assert isinstance(map_vis_msg, MarkerArray), f"The map_vis_msg is not a MarkerArray, but {type(map_vis_msg)}."
        self.geometric_map_vis_publisher.publish(map_vis_msg)
        self.get_logger().debug("Map visualization published.")

    def geo_map_to_msg(self, geo_map: GeometricMap) -> tuple[PolygonObject, PolygonObjectArray]:
        """Convert a GeometricMap object to a PolygonObjectArray message"""
        boundary_msg = PolygonObject()
        obstacle_msg = PolygonObjectArray()

        motion_type_msg = ObjectMotionType()
        motion_type_msg.o_motion_type = ObjectMotionType.STATIC
        shape_type_msg = ObjectShapeType()
        shape_type_msg.o_shape_type = ObjectShapeType.POLYGON
        semantic_type_msg = ObjectSemanticType()
        semantic_type_msg.o_semantic_type = ObjectSemanticType.UNKNOWN

        boundary_msg.object_id = -1
        boundary_msg.object_name = "boundary"
        boundary_msg.polygon = self.tuples_to_polygon_msg(geo_map.boundary_coords)

        for obs_id in geo_map.id_list:
            obs_info = geo_map.get_obstacle_info(obs_id)
            obs_msg = PolygonObject()
            obs_msg.object_id = obs_info['id_']
            obs_msg.object_name = obs_info['name']
            obs_msg.polygon = self.tuples_to_polygon_msg(obs_info['vertices'])
            
            plain_poly = PlainPolygon.from_list_of_tuples(obs_info['vertices'])
            plain_poly_centroid = plain_poly.centroid()
            obs_msg.centroid = Point32(x=float(plain_poly_centroid[0]), y=float(plain_poly_centroid[1]), z=0.0)
            obs_msg.object_motion_t = motion_type_msg
            obs_msg.object_shape_t = shape_type_msg
            obs_msg.object_semantic_t = semantic_type_msg

            obstacle_msg.polygon_objects.append(obs_msg)
        return boundary_msg, obstacle_msg
    
    def geo_map_to_vis_msg(self, map_msg: MapMsg, 
                           name_space:str="map_ns", 
                           id_start:int=0, 
                           boundary_color=(0.0, 0.0, 0.0),
                           obstacle_color=(1.0, 1.0, 1.0),
                           scale:float=0.05) -> MarkerArray:
        """Convert a PolygonObjectArrayStamped message to a MarkerArray message"""
        marker_all_poly_msg = MarkerArray()
        marker_id = id_start
        boundary_msg = map_msg.boundary
        obstacle_msg = map_msg.obstacle_list

        poly_points = boundary_msg.polygon.points
        poly_points.append(poly_points[0])
        marker_poly_msg = self.points_to_vis_msg(poly_points, boundary_color, marker_id, scale)
        marker_poly_msg.header = map_msg.header
        marker_poly_msg.ns = name_space
        marker_all_poly_msg.markers.append(marker_poly_msg)

        polygon_objects:list[PolygonObject] = obstacle_msg.polygon_objects
        for poly_obj in polygon_objects:
            assert isinstance(poly_obj, PolygonObject), f"The object is not a PolygonObject, but {type(poly_obj)}."
            marker_id += 1
            poly_points = poly_obj.polygon.points
            poly_points.append(poly_points[0])
            marker_poly_msg = self.points_to_vis_msg(poly_points, obstacle_color, marker_id, scale)
            marker_poly_msg.header = map_msg.header
            marker_poly_msg.ns = name_space
            marker_all_poly_msg.markers.append(marker_poly_msg)
        return marker_all_poly_msg
    
    @staticmethod
    def tuples_to_polygon_msg(vertices: list[tuple[float, float]]) -> Polygon:
        """Convert a list of vertices to a Polygon message"""
        polygon_msg = Polygon()
        for vertex in vertices:
            point_msg = Point32(x=float(vertex[0]), y=float(vertex[1]), z=0.0)
            polygon_msg.points.append(point_msg)
        return polygon_msg
    
    @staticmethod
    def points_to_vis_msg(points: list[Point], 
                          color: tuple[float, float, float],
                          marker_id: int,
                          scale:float=0.05) -> Marker:
        """Convert a pair of points message to a Marker message"""
        marker_msg = Marker()
        marker_msg.id = marker_id
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD

        marker_msg.pose.position.x = 0.0
        marker_msg.pose.position.y = 0.0
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0

        marker_msg.scale.x = scale
        marker_msg.scale.y = scale
        marker_msg.scale.z = scale

        marker_msg.color.a = 1.0
        marker_msg.color.r = color[0]
        marker_msg.color.g = color[1]
        marker_msg.color.b = color[2]

        marker_msg.lifetime.sec = 0 # forever
        marker_msg.frame_locked = False # not locked to a frame
        [marker_msg.points.append(Point(x=point.x, y=point.y, z=point.z)) for point in points]
        return marker_msg


def main(args=None):
    rclpy.init(args=args)

    map_publisher = GeometricMapNode()
    rclpy.spin(map_publisher)
    map_publisher.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()