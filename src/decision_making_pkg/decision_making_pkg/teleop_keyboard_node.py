"""1: manual, 2: autonomous waiting, Space: start/pause autonomous driving."""

import os
import select
import signal
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from interfaces_pkg.msg import MotionCommand
from std_msgs.msg import Bool, String

from .control_policy import DriveModeController


class TeleopKeyboardNode(Node):
    def __init__(self, mode='manual', **kwargs):
        super().__init__('teleop_keyboard_node', **kwargs)
        limit = self.declare_parameter('manual_speed_limit', 255).value
        timeout = self.declare_parameter('command_timeout', 0.5).value
        self.controller = DriveModeController(mode, limit, timeout)
        self.control_pub = self.create_publisher(MotionCommand, 'topic_control_signal', 1)
        self.reset_pub = self.create_publisher(Bool, 'driving_mode_reset', 1)
        state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, 'driving_mode', state_qos)
        self.auto_sub = self.create_subscription(
            MotionCommand, 'autonomous_control_signal', self.auto_callback, 1)
        self.timer = self.create_timer(0.1, self.publish_command)
        self.publish_state()
        self.last_display = None

    def auto_callback(self, msg):
        self.controller.receive_auto(msg, time.monotonic())

    def publish_state(self):
        self.state_pub.publish(String(data=self.controller.state))

    def process_key(self, key):
        event = self.controller.handle_key(key)
        if event == 'auto_started':
            self.reset_pub.publish(Bool(data=True))
        if event != 'ignored':
            self.publish_state()
            self.publish_command()
        return event != 'quit'

    def publish_command(self):
        command = self.controller.command(time.monotonic())
        msg = MotionCommand(steering=command.steering, left_speed=command.left_speed,
                            right_speed=command.right_speed)
        self.control_pub.publish(msg)
        display = (self.controller.state, command)
        if display != self.last_display:
            print(f'\r\033[K{self.controller.state} | speed={command.left_speed}'
                  f' steer={command.steering}', end='', flush=True)
            self.last_display = display

    def stop(self):
        self.controller.stop()
        self.publish_state()
        # Repeated stops allow the serial timer to observe shutdown before exit.
        for _ in range(3):
            self.publish_command()
            time.sleep(0.1)


def main(args=None):
    if not sys.stdin.isatty():
        raise SystemExit('Run teleop_keyboard_node in an interactive terminal.')
    print('1: 수동주행 / 2: 자율주행 대기')
    try:
        while True:
            choice = input('모드 선택 (1/2): ').strip()
            if choice in ('1', '2'):
                break
    except (EOFError, KeyboardInterrupt):
        return
    rclpy.init(args=args)
    node = TeleopKeyboardNode('manual' if choice == '1' else 'autonomous')
    print('\n1=수동 / 2=자율 대기 / Space=자율 시작·일시정지')
    print('w/s=속도 ±10, a/d=조향 ±1, x=정지, q/Ctrl+C=종료')
    settings = termios.tcgetattr(sys.stdin)
    # Ctrl+C and terminal closure also pass through the stop in finally.
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                # TextIO.read can prefetch several keys, hiding the remaining
                # keys from select(). Read the raw terminal descriptor instead.
                key = os.read(sys.stdin.fileno(), 1).decode('ascii', errors='ignore')
                if not key or not node.process_key(key):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.stop()
        finally:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            except termios.error:
                pass  # The terminal may already have been closed.
            print()
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
