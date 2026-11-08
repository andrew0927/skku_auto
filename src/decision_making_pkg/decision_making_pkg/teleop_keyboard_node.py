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

from .control_policy import DriveCommand, DriveModeController


class TeleopKeyboardNode(Node):
    def __init__(self, mode='manual', **kwargs):
        super().__init__('teleop_keyboard_node', **kwargs)
        limit = self.declare_parameter('manual_speed_limit', 255).value
        timeout = self.declare_parameter('command_timeout', 0.5).value
        auto_limit = self.declare_parameter('auto_speed_limit', 255).value
        self.controller = DriveModeController(mode, limit, timeout, auto_limit)
        self.control_pub = self.create_publisher(MotionCommand, 'topic_control_signal', 1)
        self.reset_pub = self.create_publisher(Bool, 'driving_mode_reset', 1)
        state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, 'driving_mode', state_qos)
        self.auto_sub = self.create_subscription(
            MotionCommand, 'autonomous_control_signal', self.auto_callback, 1)
        # Remote intervention (tools/remote_control.py) wins while it keeps arriving.
        self.override = None
        self.override_time = None
        self.override_timeout = self.declare_parameter('override_timeout', 0.3).value
        self.override_sub = self.create_subscription(
            MotionCommand, 'override_control_signal', self.override_callback, 1)
        # Keys sent by tools/remote_control.py key <k>, handled like typed keys.
        self.remote_quit = False
        self.remote_sub = self.create_subscription(String, 'remote_key', self.remote_callback, 10)
        self.timer = self.create_timer(0.1, self.publish_command)
        self.publish_state()
        self.last_display = None

    def auto_callback(self, msg):
        self.controller.receive_auto(msg, time.monotonic())

    def remote_callback(self, msg):
        if msg.data in ('1', '2', ' ', 'x', 'w', 's', 'q'):
            print(f'\r\033[K[remote key {msg.data!r}]')
            if not self.process_key(msg.data):
                self.remote_quit = True

    def override_callback(self, msg):
        self.override, self.override_time = msg, time.monotonic()

    def override_active(self, now):
        return self.override_time is not None and 0 <= now - self.override_time <= self.override_timeout

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
        now = time.monotonic()
        command = self.controller.command(now)
        state = self.controller.state
        if self.override_active(now):
            command = DriveCommand.bounded(self.override.steering, self.override.left_speed,
                                           self.override.right_speed)
            state = 'REMOTE override'
        msg = MotionCommand(steering=command.steering, left_speed=command.left_speed,
                            right_speed=command.right_speed)
        self.control_pub.publish(msg)
        limit = self.controller.auto_speed_limit
        display = (state, command, limit)
        if display != self.last_display:
            print(f'\r\033[K{state} | speed={command.left_speed}'
                  f' steer={command.steering} | 자율 최고속도={limit}', end='', flush=True)
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
    choice = os.environ.get('TELEOP_MODE', '').strip()
    if choice not in ('1', '2'):
        print('1: 수동주행 / 2: 자율주행 대기')
        try:
            while True:
                choice = input('모드 선택 (1/2): ').strip()
                if choice in ('1', '2'):
                    break
        except (EOFError, KeyboardInterrupt):
            return
    else:
        print(f'모드 {choice} (TELEOP_MODE)')
    rclpy.init(args=args)
    node = TeleopKeyboardNode('manual' if choice == '1' else 'autonomous')
    print('\n1=수동 / 2=자율 대기 / Space=자율 시작·일시정지')
    print('w/s=속도 ±10 (자율 모드에서는 최고속도 ±10), a/d=조향 ±1, x=정지, q/Ctrl+C=종료')
    settings = termios.tcgetattr(sys.stdin)
    # Ctrl+C and terminal closure also pass through the stop in finally.
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok() and not node.remote_quit:
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
