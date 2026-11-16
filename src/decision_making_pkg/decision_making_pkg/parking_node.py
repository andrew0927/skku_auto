"""Automatic parking is armed only by the keyboard parking mode."""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from interfaces_pkg.msg import MotionCommand
from lidar_perception_pkg.scan_geometry import scan_is_fresh, scan_is_valid, sector_distance, stamp_seconds
from .parking_policy import ParkingSequence


class ParkingNode(Node):
    def __init__(self, **kwargs):
        super().__init__('parking_node', **kwargs)
        defaults = dict(speed=150, obstacle=3.0, gap=3.5, wall=1.6, gain=14.0,
                        first_align=0.2, second_align=1.0, second_delay=2.0,
                        overshoot=3.0, stop=0.5, reverse=6.5, dwell=3.5,
                        exit_right=6.5, exit_straight=15.0)
        self.sequence = ParkingSequence(**{name: self.declare_parameter(name, value).value
                                           for name, value in defaults.items()})
        self.start_angle = self.declare_parameter('right_scan_start', 270.0).value
        self.end_angle = self.declare_parameter('right_scan_end', 280.0).value
        self.scan_timeout = self.declare_parameter('scan_timeout', 0.6).value
        self.source_timeout = self.declare_parameter('source_timeout', 1.0).value
        if self.scan_timeout <= 0 or self.source_timeout <= 0:
            raise ValueError('Parking scan timeouts must be positive')
        self.scan = self.received_at = self.consumed_stamp = None
        self.active = self.faulted = False
        self.last_debug = None
        self.sub = self.create_subscription(LaserScan, 'lidar_processed', self.scan_callback, 1)
        self.mode_sub = self.create_subscription(String, 'driving_mode', self.mode_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.publisher = self.create_publisher(MotionCommand, 'parking_control_signal', 1)
        self.debug_pub = self.create_publisher(String, 'parking_debug', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def scan_callback(self, msg):
        ros_now = self.get_clock().now().nanoseconds / 1e9
        if (scan_is_valid(msg) and scan_is_fresh(msg, ros_now, self.source_timeout)
                and (self.scan is None or stamp_seconds(msg) > stamp_seconds(self.scan))):
            self.scan, self.received_at = msg, time.monotonic()

    def mode_callback(self, msg):
        active = msg.data == 'parking_active'
        if not active or not self.active:
            self.sequence.reset()
            self.consumed_stamp = None
            self.faulted = False
        self.active = active

    def timer_callback(self):
        now = time.monotonic()
        ros_now = self.get_clock().now().nanoseconds / 1e9
        fresh = (self.scan is not None and self.received_at is not None
                 and now - self.received_at <= self.scan_timeout
                 and scan_is_fresh(self.scan, ros_now, self.source_timeout))
        command = MotionCommand()
        debug = self.sequence.state
        if self.active and not self.faulted:
            if not fresh:
                if self.sequence.state != 'IDLE':
                    self.faulted = True
                debug = 'lidar missing or stale; restart parking after reconnecting'
            else:
                distance = sector_distance(self.scan, self.start_angle, self.end_angle)
                if distance is None:
                    self.faulted = True
                    debug = 'right lidar sector invalid; restart parking'
                else:
                    if self.sequence.state == 'IDLE':
                        self.sequence.start(now)
                    stamp = stamp_seconds(self.scan)
                    result = self.sequence.tick(now, distance, stamp != self.consumed_stamp)
                    self.consumed_stamp = stamp
                    command = MotionCommand(steering=result.steering, left_speed=result.left_speed,
                                            right_speed=result.right_speed)
                    debug = self.sequence.state
        if self.faulted:
            self.sequence.reset()
            debug = 'parking stopped: restart with x then Space'
        if debug != self.last_debug:
            self.get_logger().info(f'Parking state: {debug}')
            self.last_debug = debug
        self.debug_pub.publish(String(data=debug))
        self.publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = ParkingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(MotionCommand())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
