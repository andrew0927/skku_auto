"""Verify the actual terminal menu and Space key without a serial writer."""

import os
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from interfaces_pkg.msg import MotionCommand
from std_msgs.msg import String


def test_menu_space_and_exit_in_a_real_pseudoterminal():
    assert os.environ.get('ROS_DOMAIN_ID') == '92', 'Use the isolated test ROS domain 92'
    master, slave = os.openpty()
    process = None
    rclpy.init()
    observer = Node('keyboard_terminal_test')
    commands, modes = [], []
    control_sub = observer.create_subscription(
        MotionCommand, 'topic_control_signal', commands.append, 1)
    state_sub = observer.create_subscription(
        String, 'driving_mode', lambda msg: modes.append(msg.data),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    auto_pub = observer.create_publisher(MotionCommand, 'autonomous_control_signal', 1)

    def run(duration):
        until = time.monotonic() + duration
        while time.monotonic() < until:
            auto_pub.publish(MotionCommand(steering=3, left_speed=120, right_speed=120))
            rclpy.spin_once(observer, timeout_sec=0.02)

    try:
        process = subprocess.Popen([sys.executable, '-m', 'decision_making_pkg.teleop_keyboard_node'],
                                   stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.write(master, b'2\n')
        run(1.0)
        assert modes[-1] == 'autonomous_waiting'
        assert commands and all(msg.left_speed == 0 for msg in commands)
        os.write(master, b' ')
        run(0.5)
        assert modes[-1] == 'autonomous_active'
        assert commands[-1].steering == 3 and commands[-1].left_speed == 120
        os.write(master, b'1')
        run(0.3)
        assert modes[-1] == 'manual' and commands[-1].left_speed == 0
        os.write(master, b'wd')
        run(0.3)
        assert commands[-1].steering == 1 and commands[-1].left_speed == 10
        os.write(master, b'q')
        run(0.5)
        assert process.wait(timeout=2) == 0
        assert commands[-1].steering == 0 and commands[-1].left_speed == 0
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        observer.destroy_subscription(control_sub)
        observer.destroy_subscription(state_sub)
        observer.destroy_node()
        rclpy.try_shutdown()
        os.close(master)
        os.close(slave)
