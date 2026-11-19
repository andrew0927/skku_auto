#!/usr/bin/env python3
"""Watch a running drive and exit (so the operator is notified) when the code needs help.

Exits when, while driving autonomously, the car has been commanded to stand still for
STUCK_SECS for a reason other than a red light, or when the driving mode leaves
autonomous_active (someone paused). State changes are printed as they happen.
"""

import sys
import time

import os
os.environ['ROS_LOCALHOST_ONLY'] = os.environ.get('CAR_LOCALHOST_ONLY', '1')  # as run_autonomous.sh
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from interfaces_pkg.msg import MotionCommand

STUCK_SECS = 2.0


def main():
    rclpy.init()
    node = Node('drive_watch')
    s = {'mode': '', 'state': '', 'sent': None, 'still_since': None, 'started': False}
    t0 = time.monotonic()

    def log(text):
        print(f'{time.monotonic() - t0:7.1f}s  {text}', flush=True)

    def on_mode(msg):
        if msg.data != s['mode']:
            log(f'mode -> {msg.data}')
        s['mode'] = msg.data
        if msg.data == 'autonomous_active':
            s['started'] = True

    def on_state(msg):
        if msg.data != s['state']:
            log(f'state -> {msg.data}')
            if '--until-green' in sys.argv and s['state'].startswith('stopped at crosswalk') \
                    and not msg.data.startswith('stopped'):
                s['green'] = True
        s['state'] = msg.data

    def on_sent(msg):
        s['sent'] = msg

    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, 'driving_mode', on_mode, qos)
    node.create_subscription(String, 'control_state', on_state, 1)
    node.create_subscription(MotionCommand, 'topic_control_signal', on_sent, 1)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if s.get('green'):
                log('EXIT: released from red light (green)')
                return 3
            if s['started'] and s['mode'] != 'autonomous_active':
                log(f'EXIT: paused ({s["mode"]})')
                return
            if s['mode'] == 'autonomous_active' and s['sent'] is not None:
                still = s['sent'].left_speed == 0 and 'red light' not in s['state']
                if still:
                    s['still_since'] = s['still_since'] or now
                    if now - s['still_since'] >= STUCK_SECS:
                        log(f'EXIT: stuck {STUCK_SECS}s [{s["state"]}]')
                        return
                else:
                    s['still_since'] = None
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
