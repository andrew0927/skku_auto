"""An unlabelled LED signal must reach the controller through real ROS topics."""

import os
import re
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from cv_bridge import CvBridge
from interfaces_pkg.msg import DetectionArray, PathPlanningResult
from sensor_msgs.msg import Image

from camera_perception_pkg.traffic_light_detector_node import TrafficLightDetector
from decision_making_pkg.motion_planner_node import MotionPlanningNode
from decision_making_pkg.teleop_keyboard_node import TeleopKeyboardNode
from serial_communication_pkg.serial_sender_node import SerialSenderNode
from test_crosswalk_stop import crosswalk
from test_signal_geometry import signal_image


def test_missing_yolo_light_class_still_stops_at_reference_line_on_virtual_arduino():
    master, slave = os.openpty()
    port = os.ttyname(slave)
    assert port.startswith('/dev/pts/')
    os.set_blocking(master, False)
    rclpy.init()
    executor = SingleThreadedExecutor()
    nodes, sender = [], None
    try:
        inputs = Node('synthetic_signal_inputs')
        image_pub = inputs.create_publisher(Image, 'image_raw', 1)
        detection_pub = inputs.create_publisher(DetectionArray, 'detections', 1)
        path_pub = inputs.create_publisher(PathPlanningResult, 'path_planning_result', 1)
        traffic = TrafficLightDetector()
        motion = MotionPlanningNode(parameter_overrides=[Parameter('enable_obstacle_avoidance', value=False)])
        keyboard = TeleopKeyboardNode(mode='autonomous')
        sender = SerialSenderNode(parameter_overrides=[
            Parameter('port', value=port), Parameter('startup_delay', value=0.0)])
        nodes = [inputs, traffic, motion, keyboard, sender]
        for node in nodes:
            executor.add_node(node)
        bridge = CvBridge()
        buffer, packets = b'', []

        def run(duration, color=(0, 255, 0), position=0.4, lamp=True):
            nonlocal buffer
            end, next_frame = time.monotonic() + duration, 0.0
            while time.monotonic() < end:
                if time.monotonic() >= next_frame:
                    frame = signal_image(color)
                    if not lamp:
                        frame[:] = 170
                    image = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                    image.header.stamp = inputs.get_clock().now().to_msg()
                    detections = DetectionArray()
                    detections.header = image.header
                    detections.detections = [crosswalk(position)]
                    path = PathPlanningResult(tracking_source='direct', target_lane='lane2')
                    path.header = image.header
                    path.x_points, path.y_points = [320.0] * 3, [5.0, 80.0, 179.0]
                    image_pub.publish(image)
                    detection_pub.publish(detections)
                    path_pub.publish(path)
                    next_frame = time.monotonic() + 0.06
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

        run(1.0)
        assert packets and packets[-1] == (0, 0, 0)
        keyboard.process_key(' ')
        run(0.4)
        assert packets[-1] == (0, 255, 255)
        run(0.5, color=(0, 0, 255))
        assert packets[-1] == (0, 70, 70)
        run(0.35, color=(0, 0, 255), position=325.1 / 480)
        assert packets[-1] == (0, 0, 0)
        run(0.3, lamp=False)
        assert packets[-1] == (0, 0, 0)
        run(0.6)
        assert packets[-1] == (0, 255, 255)
        keyboard.process_key('x')
        run(0.2)
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
