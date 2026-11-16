"""Publish RPLidar scans with the existing driver and explicit shutdown."""

import math
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from .lib import lidar_perception_func_lib as LPFL


class LidarSensorDataPublisher(Node):
    def __init__(self, **kwargs):
        super().__init__('lidar_publisher_node', **kwargs)
        port = self.declare_parameter('port', os.environ.get('LIDAR_PORT', '/dev/ttyUSB0')).value
        baudrate = self.declare_parameter('baudrate', 115200).value
        read_timeout = self.declare_parameter('read_timeout', 1.0).value
        if read_timeout <= 0:
            raise ValueError('Lidar read_timeout must be positive')
        self.publisher_ = self.create_publisher(LaserScan, 'lidar_raw', 1)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.lidar = None
        self.lidar = LPFL.RPLidar(port, baudrate=baudrate, timeout=read_timeout)
        self.scans = self.lidar.iter_scans(max_buf_meas=1000)
        self.timer = self.create_timer(0.01, self.publish_lidar_data)
        self.get_logger().info(f'Lidar connected: {port}')

    def publish_lidar_data(self):
        try:
            measurements = next(self.scans)
        except (StopIteration, LPFL.RPLidarException, OSError, ValueError) as error:
            raise RuntimeError(f'Lidar scan failed: {error}') from error
        ranges, intensities = [math.inf] * 360, [0.0] * 360
        for quality, angle, millimeters in measurements:
            distance = millimeters / 1000.0
            if math.isfinite(angle) and math.isfinite(distance) and 0.15 <= distance <= 12.0:
                index = int(angle) % 360
                if distance < ranges[index]:
                    ranges[index], intensities[index] = distance, float(quality)
        if not any(math.isfinite(d) for d in ranges):
            return
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_frame'
        msg.angle_min, msg.angle_max = 0.0, math.radians(359)
        msg.angle_increment = math.radians(1)
        msg.scan_time = 0.1
        msg.range_min, msg.range_max = 0.15, 12.0
        msg.ranges, msg.intensities = ranges, intensities
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'laser_frame'
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)
        self.publisher_.publish(msg)

    def close(self):
        if self.lidar is not None:
            for action in (self.lidar.stop, self.lidar.stop_motor, self.lidar.disconnect):
                try:
                    action()
                except (LPFL.RPLidarException, OSError):
                    pass
            self.lidar = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LidarSensorDataPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
