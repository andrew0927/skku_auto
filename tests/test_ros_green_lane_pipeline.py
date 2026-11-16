"""User signal photos through real YOLO/ROS and a virtual serial port only."""

import json
import os
from pathlib import Path
import re
import time

import cv2
import pytest
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import String

from camera_perception_pkg import lane_inference as lane_module
from camera_perception_pkg.yolov8_node import Yolov8Node
from camera_perception_pkg.lane_info_extractor_node import Yolov8InfoExtractor
from camera_perception_pkg.traffic_light_detector_node import TrafficLightDetector
from decision_making_pkg.path_planner_node import PathPlannerNode
from decision_making_pkg.motion_planner_node import MotionPlanningNode
from decision_making_pkg.teleop_keyboard_node import TeleopKeyboardNode
from serial_communication_pkg.serial_sender_node import SerialSenderNode


GREEN = Path('/home/donghyun/Pictures/Webcam/2026-11-08-173052.jpg')
RED = Path('/home/donghyun/Camera Image_screenshot_08.11.2026.png')
MODEL = Path(__file__).resolve().parents[1] / 'best.pt'


@pytest.mark.skipif(not all(p.exists() for p in (GREEN, RED, MODEL)), reason='Reference photos/model unavailable')
def test_red_green_lane_recovery_and_lamp_flicker_without_vehicle_usb(monkeypatch):
    import torch

    torch.set_num_threads(4)
    master, slave = os.openpty()
    port = os.ttyname(slave)
    assert port.startswith('/dev/pts/')
    os.set_blocking(master, False)
    rclpy.init()
    executor = SingleThreadedExecutor()
    nodes, sender = [], None
    try:
        inputs = Node('green_photo_replay')
        nodes.append(inputs)
        images = inputs.create_publisher(Image, 'image_raw', 1)
        diagnostics = []
        inputs.create_subscription(String, 'yolo_inference_debug',
                                   lambda msg: diagnostics.append(json.loads(msg.data)), 1)
        yolo = Yolov8Node(parameter_overrides=[Parameter('model', value=str(MODEL)),
                                              Parameter('device', value='cpu')])
        nodes.append(yolo)
        assert yolo.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert yolo.trigger_activate() == TransitionCallbackReturn.SUCCESS
        extractor = Yolov8InfoExtractor(parameter_overrides=[Parameter('show_image', value=False)])
        nodes.append(extractor)
        planner = PathPlannerNode()
        nodes.append(planner)
        traffic = TrafficLightDetector()
        nodes.append(traffic)
        motion = MotionPlanningNode(parameter_overrides=[Parameter('enable_obstacle_avoidance', value=False)])
        nodes.append(motion)
        keyboard = TeleopKeyboardNode(mode='autonomous')
        nodes.append(keyboard)
        sender = SerialSenderNode(parameter_overrides=[
            Parameter('port', value=port), Parameter('startup_delay', value=0.0)])
        nodes.append(sender)
        for node in nodes:
            executor.add_node(node)
        bridge, buffer, packets = CvBridge(), b'', []
        green = cv2.resize(cv2.imread(str(GREEN)), (640, 480))
        red = cv2.resize(cv2.imread(str(RED)), (640, 480))
        original_green = green.copy()

        def run(duration, frame, until=None):
            nonlocal buffer
            deadline, next_frame = time.monotonic() + duration, 0.0
            while time.monotonic() < deadline:
                if time.monotonic() >= next_frame:
                    msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                    msg.header.stamp = inputs.get_clock().now().to_msg()
                    images.publish(msg)
                    next_frame = time.monotonic() + 0.15
                executor.spin_once(timeout_sec=0.005)
                try:
                    buffer += os.read(master, 4096)
                except BlockingIOError:
                    pass
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    match = re.fullmatch(rb's(-?\d+)l(-?\d+)r(-?\d+)', line)
                    assert match, line
                    packets.append(tuple(map(int, match.groups())))
                if until is not None and until():
                    return

        run(3.0, green, until=lambda: packets and motion.light == 'green'
            and motion.path is not None and keyboard.controller.auto.left_speed == 255)
        assert packets and packets[-1] == (0, 0, 0)
        keyboard.process_key(' ')
        run(3.0, green, until=lambda: packets[-1][1:] == (255, 255))
        assert packets[-1][1:] == (255, 255)
        run(3.0, red, until=lambda: motion.crosswalk_stop.red and packets[-1] == (0, 0, 0))
        assert motion.crosswalk_stop.red and packets[-1] == (0, 0, 0)
        run(3.5, green, until=lambda: not motion.crosswalk_stop.red
            and motion.path is not None and packets[-1][1] > 0)
        assert not motion.crosswalk_stop.red
        assert packets[-1][1] > 0 and motion.path is not None
        assert yolo.inference.prefer_gray
        # Only YOLO's geometric helper misses the lamp; traffic color still uses raw pixels.
        monkeypatch.setattr(lane_module, 'find_signal_lamp', lambda _: None)
        begin = len(packets)
        run(1.2, green)
        assert packets[begin:] and all(left > 0 and right > 0 for _, left, right in packets[begin:])
        assert yolo.inference.prefer_gray
        assert any(d['mode'] == 'green_grayscale_cached' and d['forward_passes'] == 1 for d in diagnostics)
        assert max(d['forward_passes'] for d in diagnostics) <= 2
        assert (green == original_green).all()
        run(3.0, red, until=lambda: motion.crosswalk_stop.red and packets[-1] == (0, 0, 0))
        assert motion.crosswalk_stop.red and packets[-1] == (0, 0, 0)
        keyboard.process_key('x')
        run(0.3, green)
        assert packets[-1] == (0, 0, 0)
    finally:
        if sender is not None:
            sender.close()
        for node in reversed(nodes):
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
        os.close(master)
        os.close(slave)
