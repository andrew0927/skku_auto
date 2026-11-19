"""Obstacle straight ahead in the car's own lane, from the processed lidar scan.

The scan (lidar_processed) has one range per degree, 0 deg = straight ahead and angles growing to
the left (the processor flips the RPLidar's clockwise order). Each point is turned into forward (x)
and left (y) distances; an obstacle is in the lane when enough points lie in a corridor as wide as
the car in front of it. A car in the other lane is about one lane width to the side and is ignored,
which an angular sector cannot do (on curves the sector points into the other lane).

Publishes lidar_obstacle_info (Bool, debounced) and lidar_obstacle_offset (Float32: left offset in m
of the nearest corridor point, NaN when there is none).
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32

from .lib import lidar_perception_func_lib as LPFL

SUB_TOPIC_NAME = 'lidar_processed'
PUB_TOPIC_NAME = 'lidar_obstacle_info'


def corridor_points(ranges, angle_min, angle_increment, ahead_min, ahead_max, half_width):
    """(x, y) of scan points in the corridor ahead: ahead_min <= x <= ahead_max, |y| <= half_width."""
    points = []
    for index, distance in enumerate(ranges):
        if not math.isfinite(distance) or distance <= 0.0:
            continue
        angle = angle_min + index * angle_increment
        x, y = distance * math.cos(angle), distance * math.sin(angle)
        if ahead_min <= x <= ahead_max and abs(y) <= half_width:
            points.append((x, y))
    return points


class ObjectDetection(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_detector_node')
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.ahead_min = self.declare_parameter('ahead_min', 0.15).value      # m in front of the lidar
        self.ahead_max = self.declare_parameter('ahead_max', 1.5).value
        self.half_width = self.declare_parameter('half_width', 0.3).value     # m each side: car width
        self.min_points = self.declare_parameter('min_points', 2).value       # ignore single echoes
        self.detection_checker = LPFL.StabilityDetector(consec_count=self.declare_parameter('consec_count', 3).value)
        self.subscriber = self.create_subscription(LaserScan, SUB_TOPIC_NAME, self.lidar_callback, qos)
        self.publisher = self.create_publisher(Bool, PUB_TOPIC_NAME, qos)
        self.offset_pub = self.create_publisher(Float32, 'lidar_obstacle_offset', qos)
        self.last_result = None

    def lidar_callback(self, msg):
        points = corridor_points(msg.ranges, msg.angle_min, msg.angle_increment,
                                 self.ahead_min, self.ahead_max, self.half_width)
        detected = len(points) >= self.min_points
        result = self.detection_checker.check_consecutive_detections(detected)
        self.publisher.publish(Bool(data=result))
        nearest = min(points) if points else None
        self.offset_pub.publish(Float32(data=nearest[1] if nearest else float('nan')))
        if result != self.last_result:  # log changes only, with where the obstacle is
            where = f' at {nearest[0]:.2f} m ahead, {nearest[1]:+.2f} m left' if nearest else ''
            self.get_logger().info(f'Lidar Obstacle detected: {result}{where}')
            self.last_result = result


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
