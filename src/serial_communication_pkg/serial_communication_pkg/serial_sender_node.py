"""One serial writer with bounded commands, reconnect and host command timeout."""

import os
import time

import serial
import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import MotionCommand
from .lib.protocol_convert_func_lib import convert_serial_message


class SerialSenderNode(Node):
    def __init__(self, **kwargs):
        super().__init__('serial_sender_node', **kwargs)
        self.port = self.declare_parameter('port', os.environ.get('ARDUINO_PORT', '/dev/ttyACM0')).value
        self.baudrate = self.declare_parameter('baudrate', 9600).value
        self.command_timeout = self.declare_parameter('command_timeout', 0.5).value
        self.startup_delay = self.declare_parameter('startup_delay', 2.0).value
        self.send_period = self.declare_parameter('send_period', 0.1).value
        if self.command_timeout <= 0 or self.startup_delay < 0 or self.send_period < 0.05:
            raise ValueError('Invalid serial timing parameters')
        self.serial = None
        self.latest = None
        self.received_at = None
        self.ready_at = 0.0
        self.retry_at = 0.0
        self.subscription = self.create_subscription(MotionCommand, 'topic_control_signal',
                                                     self.data_callback, 1)
        self.timer = self.create_timer(self.send_period, self.timer_callback)
        self.source_check = self.create_timer(2.0, self.check_sources)
        self.warned_sources = 0

    def check_sources(self):
        # Two command sources (e.g. another laptop on the same network and ROS domain) make the car
        # alternate between their commands. Say so loudly instead of driving in staccato.
        publishers = self.get_publishers_info_by_topic('topic_control_signal')
        if len(publishers) > 1 and len(publishers) != self.warned_sources:
            names = ', '.join(p.node_name for p in publishers)
            self.get_logger().error(f'{len(publishers)} nodes send car commands ({names}): another computer '
                                    'or a leftover keyboard window is driving too')
        self.warned_sources = len(publishers)

    def data_callback(self, msg):
        now = time.monotonic()
        if self.serial is not None and now >= self.ready_at:
            self.latest = convert_serial_message(msg.steering, msg.left_speed, msg.right_speed)
            self.received_at = now

    def timer_callback(self):
        now = time.monotonic()
        if self.serial is None:
            if now < self.retry_at:
                return
            try:
                self.serial = serial.Serial(self.port, self.baudrate,
                                             timeout=0.1, write_timeout=0.2, exclusive=True)
            except (serial.SerialException, OSError) as error:
                self.retry_at = now + 2.0
                self.get_logger().error(f'Serial connection failed: {error}')
                return
            self.ready_at = now + self.startup_delay
            self.latest = self.received_at = None
            self.get_logger().info(f'Serial connected: {self.port}, {self.baudrate} baud')
        if now < self.ready_at:
            return
        message = convert_serial_message(0, 0, 0)
        if self.received_at is not None and now - self.received_at <= self.command_timeout:
            message = self.latest
        try:
            self.serial.write(message.encode('ascii'))
        except (serial.SerialException, OSError) as error:
            self.get_logger().error(f'Serial write failed: {error}')
            self.serial.close()
            self.serial = None
            self.latest = self.received_at = None
            self.retry_at = now + 2.0

    def close(self):
        if self.serial is not None:
            try:
                for _ in range(3):
                    self.serial.write(convert_serial_message(0, 0, 0).encode('ascii'))
                    time.sleep(0.05)
            except (serial.SerialException, OSError):
                pass
            finally:
                self.serial.close()
                self.serial = None


def main(args=None):
    rclpy.init(args=args)
    node = SerialSenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
