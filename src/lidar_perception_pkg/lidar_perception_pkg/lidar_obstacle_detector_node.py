"""Confirm frontal obstacles from fresh scans; missing scans are not clear roads."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from .scan_geometry import scan_is_fresh, scan_is_valid, sector_distance, stamp_seconds


class LidarObstacleDetector(Node):
    def __init__(self, **kwargs):
        super().__init__('lidar_obstacle_detector_node', **kwargs)
        self.start = self.declare_parameter('start_angle', 345.0).value
        self.end = self.declare_parameter('end_angle', 15.0).value
        self.near = self.declare_parameter('range_min', 0.05).value
        self.far = self.declare_parameter('range_max', 1.5).value
        self.consecutive = self.declare_parameter('consecutive_count', 2).value
        self.source_timeout = self.declare_parameter('source_timeout', 1.0).value
        if self.consecutive < 1 or not 0 < self.near < self.far:
            raise ValueError('Invalid obstacle detector settings')
        self.pending, self.count, self.blocked = False, 0, False
        self.last_stamp = 0.0
        self.subscriber = self.create_subscription(LaserScan, 'lidar_processed', self.lidar_callback, 1)
        self.publisher = self.create_publisher(Bool, 'lidar_obstacle_info', 1)

    def lidar_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if (not scan_is_fresh(msg, now, self.source_timeout) or not scan_is_valid(msg)
                or stamp_seconds(msg) <= self.last_stamp):
            return
        self.last_stamp = stamp_seconds(msg)
        distance = sector_distance(msg, self.start, self.end)
        if distance is None:
            self.count = 0
            return
        detected = self.near <= distance <= self.far
        self.count = self.count + 1 if detected == self.pending else 1
        self.pending = detected
        if self.count >= self.consecutive:
            self.blocked = detected
        self.publisher.publish(Bool(data=self.blocked))


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
