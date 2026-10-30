"""Exercise real ROS messages and a virtual serial port; never open vehicle USB."""

import os
import re
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from interfaces_pkg.msg import Detection, DetectionArray, Point2D

from camera_perception_pkg.lane_info_extractor_node import Yolov8InfoExtractor
from decision_making_pkg.path_planner_node import PathPlannerNode
from decision_making_pkg.motion_planner_node import MotionPlanningNode
from decision_making_pkg.teleop_keyboard_node import TeleopKeyboardNode
from serial_communication_pkg.serial_sender_node import SerialSenderNode


def test_keyboard_path_and_serial_timeout_without_hardware():
    master, slave = os.openpty()
    virtual_port = os.ttyname(slave)
    assert virtual_port.startswith('/dev/pts/')
    os.set_blocking(master, False)
    rclpy.init()
    executor = SingleThreadedExecutor()
    nodes = []
    sender = None
    try:
        detector = Node('synthetic_lane_detector')
        pub = detector.create_publisher(DetectionArray, 'detections', 1)
        extractor = Yolov8InfoExtractor(parameter_overrides=[Parameter('show_image', value=False)])
        planner = PathPlannerNode()
        motion = MotionPlanningNode()
        keyboard = TeleopKeyboardNode(mode='autonomous')
        sender = SerialSenderNode(parameter_overrides=[
            Parameter('port', value=virtual_port), Parameter('startup_delay', value=0.0),
            Parameter('command_timeout', value=0.25), Parameter('send_period', value=0.05),
        ])
        nodes = [detector, extractor, planner, motion, keyboard, sender]
        for node in nodes:
            executor.add_node(node)
        src = np.float32([[238, 316], [402, 313], [501, 476], [155, 476]])
        dst = np.float32([[192, 0], [448, 0], [448, 480], [192, 480]])
        polygon = cv2.perspectiveTransform(
            np.float32([[[250, 0], [390, 0], [390, 479], [250, 479]]]),
            cv2.getPerspectiveTransform(dst, src))[0]
        detection = Detection(class_name='lane2', score=0.99)
        detection.mask.width, detection.mask.height = 640, 480
        detection.mask.data = [Point2D(x=float(x), y=float(y)) for x, y in polygon]
        packet_buffer = b''
        packets = []

        def run(duration, valid=True, old_stamp=False, publish=True):
            nonlocal packet_buffer
            end, next_frame = time.monotonic() + duration, 0.0
            while time.monotonic() < end:
                if publish and time.monotonic() >= next_frame:
                    msg = DetectionArray()
                    if not old_stamp:
                        msg.header.stamp = detector.get_clock().now().to_msg()
                    if valid:
                        msg.detections = [detection]
                    pub.publish(msg)
                    next_frame = time.monotonic() + 0.08
                executor.spin_once(timeout_sec=0.01)
                try:
                    packet_buffer += os.read(master, 4096)
                except BlockingIOError:
                    pass
                while b'\n' in packet_buffer:
                    line, packet_buffer = packet_buffer.split(b'\n', 1)
                    match = re.fullmatch(rb's(-?\d+)l(-?\d+)r(-?\d+)', line)
                    assert match, line
                    assert len(line) <= 14
                    packets.append(tuple(map(int, match.groups())))

        run(0.6)
        assert packets and all(packet == (0, 0, 0) for packet in packets)
        keyboard.process_key(' ')
        assert keyboard.controller.command(time.monotonic()).left_speed == 255
        run(0.6)
        assert any(left == right == 255 and abs(steer) <= 1 for steer, left, right in packets[-5:])
        keyboard.process_key(' ')
        run(0.25)
        assert packets[-1] == (0, 0, 0)
        keyboard.process_key(' ')
        run(0.65, valid=False)
        assert packets[-1] == (0, 0, 0)
        run(0.4, old_stamp=True)
        assert packets[-1] == (0, 0, 0)
        run(0.4)
        assert packets[-1][1] > 0
        run(0.9, publish=False)
        assert packets[-1] == (0, 0, 0)
        keyboard.process_key('1')
        keyboard.process_key('w')
        keyboard.process_key('d')
        run(0.3, publish=False)
        assert packets[-1] == (1, 10, 10)
        keyboard.timer.cancel()
        run(0.6, publish=False)
        assert packets[-1] == (0, 0, 0), 'Serial sender must stop if keyboard commands disappear'
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
