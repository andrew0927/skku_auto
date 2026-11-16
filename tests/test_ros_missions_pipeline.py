"""Run mission ROS topics through the sole keyboard gate to a virtual Arduino."""

import os
import re
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from interfaces_pkg.msg import DetectionArray, PathPlanningResult
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from decision_making_pkg.motion_planner_node import MotionPlanningNode
from decision_making_pkg.parking_node import ParkingNode
from decision_making_pkg.teleop_keyboard_node import TeleopKeyboardNode
from lidar_perception_pkg.lidar_obstacle_detector_node import LidarObstacleDetector
from serial_communication_pkg.serial_sender_node import SerialSenderNode

from test_missions import scan
from test_crosswalk_stop import crosswalk


def test_red_avoidance_and_parking_reach_only_the_virtual_serial_port():
    master, slave = os.openpty()
    virtual_port = os.ttyname(slave)
    assert virtual_port.startswith('/dev/pts/')
    os.set_blocking(master, False)
    rclpy.init()
    executor = SingleThreadedExecutor()
    nodes, sender = [], None
    try:
        inputs = Node('synthetic_mission_inputs')
        path_pub = inputs.create_publisher(PathPlanningResult, 'path_planning_result', 1)
        light_pub = inputs.create_publisher(String, 'yolov8_traffic_light_info', 1)
        presence_pub = inputs.create_publisher(Bool, 'traffic_light_presence', 1)
        scan_pub = inputs.create_publisher(LaserScan, 'lidar_processed', 1)
        detection_pub = inputs.create_publisher(DetectionArray, 'detections', 1)
        motion = MotionPlanningNode(parameter_overrides=[Parameter('crosswalk_stop_y_ratio', value=0.5)])
        detector = LidarObstacleDetector()
        parking = ParkingNode()
        keyboard = TeleopKeyboardNode(mode='autonomous')
        sender = SerialSenderNode(parameter_overrides=[
            Parameter('port', value=virtual_port), Parameter('startup_delay', value=0.0)])
        nodes = [inputs, motion, detector, parking, keyboard, sender]
        for node in nodes:
            executor.add_node(node)
        packet_buffer, packets = b'', []

        def run(duration, light='Green', obstacle=False, publish_scan=True, position=None):
            nonlocal packet_buffer
            end, next_frame = time.monotonic() + duration, 0.0
            while time.monotonic() < end:
                if time.monotonic() >= next_frame:
                    path = PathPlanningResult(tracking_source='direct', target_lane=motion.current_lane)
                    path.header.stamp = inputs.get_clock().now().to_msg()
                    path.x_points, path.y_points = [320.0] * 3, [5.0, 80.0, 179.0]
                    detections = DetectionArray()
                    detections.header = path.header
                    if position is not None:
                        detections.detections = [crosswalk(position)]
                    detection_pub.publish(detections)
                    path_pub.publish(path)
                    light_pub.publish(String(data=light))
                    presence_pub.publish(Bool(data=light.casefold() != 'none'))
                    if publish_scan:
                        scan_pub.publish(scan(inputs, front=0.5 if obstacle else 4.0))
                    next_frame = time.monotonic() + 0.06
                executor.spin_once(timeout_sec=0.005)
                try:
                    packet_buffer += os.read(master, 4096)
                except BlockingIOError:
                    pass
                while b'\n' in packet_buffer:
                    line, packet_buffer = packet_buffer.split(b'\n', 1)
                    match = re.fullmatch(rb's(-?\d+)l(-?\d+)r(-?\d+)', line)
                    assert match, line
                    packets.append(tuple(map(int, match.groups())))

        run(0.8)
        assert packets and packets[-1] == (0, 0, 0)
        keyboard.process_key(' ')
        run(0.4)
        assert packets[-1] == (0, 255, 255)
        run(0.3, light='Red', obstacle=True)
        assert packets[-1] == (0, 0, 0)
        assert motion.current_lane == 'lane2'
        run(0.4, obstacle=True)
        assert packets[-1] == (-7, 80, 80)
        assert motion.current_lane == 'lane1'
        run(0.3, light='Red')
        assert packets[-1] == (0, 0, 0)
        run(0.4)
        assert packets[-1][1:] == (255, 255)
        run(0.3, position=0.3)
        run(0.4, light='Red', position=0.3)
        assert packets[-1][1:] == (70, 70)
        run(0.3, light='None', position=0.49)
        assert packets[-1][1:] == (70, 70)
        run(0.3, light='Red', position=0.5)
        assert packets[-1] == (0, 0, 0)
        run(0.3, light='None')
        assert packets[-1] == (0, 0, 0)
        run(0.3)
        assert packets[-1][1:] == (255, 255)
        keyboard.process_key('3')
        run(0.25)
        assert packets[-1] == (0, 0, 0)
        keyboard.process_key(' ')
        run(0.4)
        assert packets[-1] == (0, 150, 150)
        run(0.95, publish_scan=False)
        assert packets[-1] == (0, 0, 0) and parking.faulted
        run(0.3)
        assert packets[-1] == (0, 0, 0), 'A reconnected scan cannot restart parking by itself'
        keyboard.process_key('x')
        run(0.2)
        keyboard.process_key(' ')
        run(0.4)
        assert packets[-1] == (0, 150, 150)
        keyboard.process_key('1')
        keyboard.process_key('w')
        run(0.3)
        assert packets[-1] == (0, 10, 10), 'Parking output must not leak into manual mode'
        keyboard.process_key('q')
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
