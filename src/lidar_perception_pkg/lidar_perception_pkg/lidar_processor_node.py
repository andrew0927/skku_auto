"""Normalize the mounted lidar into 360 angular bins, preserving scan time."""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarSensorDataProcessor(Node):
    def __init__(self, **kwargs):
        super().__init__('lidar_processor_node', **kwargs)
        self.offset = self.declare_parameter('offset', 0.0).value
        self.flip = self.declare_parameter('flip', True).value
        self.subscription = self.create_subscription(LaserScan, 'lidar_raw', self.lidar_raw_cb, 1)
        self.publisher = self.create_publisher(LaserScan, 'lidar_processed', 1)

    def lidar_raw_cb(self, msg):
        if not msg.ranges or not math.isfinite(msg.angle_increment) or msg.angle_increment == 0:
            return
        result = LaserScan()
        result.header = msg.header
        result.angle_min = 0.0
        result.angle_max = math.radians(359)
        result.angle_increment = math.radians(1)
        result.scan_time = msg.scan_time
        result.range_min, result.range_max = msg.range_min, msg.range_max
        ranges, intensities = [math.inf] * 360, [0.0] * 360
        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance) or not msg.range_min <= distance <= msg.range_max:
                continue
            angle = math.degrees(msg.angle_min + index * msg.angle_increment) - self.offset
            bin_index = round(-angle if self.flip else angle) % 360
            if distance < ranges[bin_index]:
                ranges[bin_index] = distance
                if index < len(msg.intensities):
                    intensities[bin_index] = msg.intensities[index]
        result.ranges, result.intensities = ranges, intensities
        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = LidarSensorDataProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
