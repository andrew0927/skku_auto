#!/usr/bin/env python3
"""Intervene in a running drive (run_traffic_light.sh / run_autonomous.sh).

  remote_control.py look                      save the current camera frame
  remote_control.py key K                     press a keyboard-window key remotely (2, space, x, q)
  remote_control.py stop                      stop the car and pause autonomy; confirms speed 0
  remote_control.py plan "S:V:T S:V:T ..." [note]
                                               run several segments back to back without stopping
  remote_control.py step STEER SPEED SECS [note]
                                               drive for SECS (<= 1.5 s) then save a frame
The keyboard node forwards these commands to the car while they keep arriving (0.3 s),
whatever mode it is in; press Space in the keyboard window to pause the code first.
Snapshots and decisions go to <latest lap>/remote/ so they stay with the recording.
"""

import csv
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import os
os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')  # same isolation as run_autonomous.sh
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces_pkg.msg import MotionCommand

WS = Path(__file__).resolve().parent.parent
MAX_SECS, MAX_SPEED, MAX_STEER = 1.5, 180, 7


def wait_for_keyboard(node, topic, timeout=5.0):
    """Publishing before discovery silently drops the message (seen 2026-11-08: a pause key reached
    only the recorder). Wait until the keyboard node itself subscribes."""
    end = time.time() + timeout
    while time.time() < end:
        if any(i.node_name == 'teleop_keyboard_node' for i in node.get_subscriptions_info_by_topic(topic)):
            time.sleep(0.2)
            return True
        rclpy.spin_once(node, timeout_sec=0.05)
    sys.exit(f'Keyboard node is not subscribed to {topic}: is the drive running?')


def emergency_stop(node):
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from std_msgs.msg import String
    override = node.create_publisher(MotionCommand, 'override_control_signal', 1)
    key = node.create_publisher(String, 'remote_key', 10)
    state = {}
    node.create_subscription(MotionCommand, 'topic_control_signal',
                             lambda m: state.__setitem__('sent', (m.steering, m.left_speed)), 1)
    node.create_subscription(String, 'driving_mode', lambda m: state.__setitem__('mode', m.data),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    end, next_key = time.time() + 5.0, 0.0
    while time.time() < end:
        override.publish(MotionCommand(steering=0, left_speed=0, right_speed=0))
        if time.time() >= next_key:
            key.publish(String(data='x'))
            next_key = time.time() + 0.5
        rclpy.spin_once(node, timeout_sec=0.05)
        if state.get('mode', '').startswith('autonomous_waiting') or state.get('mode') == 'manual':
            if state.get('sent', (0, 1))[1] == 0:
                print(f'stopped: sent {state["sent"]}, mode {state["mode"]}')
                return
    sys.exit(f'STOP NOT CONFIRMED: {state} - use the keyboard x or motor power')


def latest_frame(node, timeout=3.0):
    got = {}
    sub = node.create_subscription(Image, 'image_raw', lambda m: got.setdefault('m', m), 1)
    end = time.time() + timeout
    while 'm' not in got and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if 'm' not in got:
        sys.exit('No camera frame: is run_traffic_light.sh running?')
    m = got['m']
    return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1).copy()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('look', 'step', 'key', 'plan', 'stop'):
        sys.exit(__doc__)
    if sys.argv[1] == 'stop':
        rclpy.init()
        node = Node('remote_stop')
        emergency_stop(node)
        node.destroy_node()
        rclpy.shutdown()
        return
    if sys.argv[1] == 'key':
        from std_msgs.msg import String
        rclpy.init()
        node = Node('remote_key_sender')
        pub = node.create_publisher(String, 'remote_key', 10)
        key = ' ' if sys.argv[2] == 'space' else sys.argv[2]
        wait_for_keyboard(node, 'remote_key')
        pub.publish(String(data=key))
        time.sleep(0.3)
        print(f'sent key {key!r} to {pub.get_subscription_count()} subscriber(s)')
        node.destroy_node()
        rclpy.shutdown()
        return
    laps = sorted(glob.glob(str(WS / 'laps' / 'lap_*')))
    out = Path(laps[-1] if laps else WS / 'laps' / 'no_recording') / 'remote'
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node('remote_control')
    pub = node.create_publisher(MotionCommand, 'override_control_signal', 1)
    if sys.argv[1] in ('step', 'plan'):
        wait_for_keyboard(node, 'override_control_signal')
    note = ''
    if sys.argv[1] == 'plan':
        note = ' '.join(sys.argv[3:])
        for segment in sys.argv[2].split():
            steer, speed, secs = segment.split(':')
            steer = max(-MAX_STEER, min(MAX_STEER, int(steer)))
            speed = max(-MAX_SPEED, min(MAX_SPEED, int(speed)))
            end = time.monotonic() + max(0.0, min(3.0, float(secs)))
            while time.monotonic() < end:
                pub.publish(MotionCommand(steering=steer, left_speed=speed, right_speed=speed))
                time.sleep(0.1)
        sys.argv[2:5] = [sys.argv[2], '', '']
    elif sys.argv[1] == 'step':
        steer = max(-MAX_STEER, min(MAX_STEER, int(sys.argv[2])))
        speed = max(-MAX_SPEED, min(MAX_SPEED, int(sys.argv[3])))
        secs = max(0.0, min(MAX_SECS, float(sys.argv[4])))
        note = ' '.join(sys.argv[5:])
        end = time.monotonic() + secs
        while time.monotonic() < end:
            pub.publish(MotionCommand(steering=steer, left_speed=speed, right_speed=speed))
            time.sleep(0.1)
        # Hold still briefly so the snapshot is not blurred; then the keyboard node takes over.
        for _ in range(4):
            pub.publish(MotionCommand(steering=steer, left_speed=0, right_speed=0))
            time.sleep(0.1)
    frame = latest_frame(node)
    name = f'{datetime.now():%H%M%S_%f}'[:-3] + '.jpg'
    cv2.line(frame, (320, 240), (320, 479), (255, 0, 255), 1)
    cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log = out / 'interventions.csv'
    new = not log.exists()
    with open(log, 'a', newline='') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['wall_time', 'command', 'steering', 'speed', 'secs', 'snapshot_after', 'note'])
        writer.writerow([f'{time.time():.2f}', *sys.argv[1:5], name, note] if sys.argv[1] in ('step', 'plan')
                        else [f'{time.time():.2f}', 'look', '', '', '', name, ''])
    print(out / name)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
