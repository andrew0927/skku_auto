"""A white LED core with a colored rim needs a housing, not just red pixels."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import rclpy
from interfaces_pkg.msg import DetectionArray
from rclpy.time import Time

from camera_perception_pkg.signal_geometry import ColorConfirmation, find_signal_lamp
from camera_perception_pkg.traffic_light_detector_node import TrafficLightDetector


def signal_image(color):
    image = np.full((480, 640, 3), 170, dtype=np.uint8)
    cv2.rectangle(image, (70, 65), (290, 140), (15, 15, 15), -1)
    cv2.circle(image, (112, 103), 28, color, -1)
    cv2.circle(image, (112, 103), 20, (255, 255, 255), -1)
    return image


@pytest.mark.parametrize('color, expected', [
    ((0, 0, 255), 'Red'), ((0, 180, 255), 'Yellow'), ((0, 255, 0), 'Green')])
def test_white_hot_lamp_core_can_be_classified_from_its_colored_rim(color, expected):
    found = find_signal_lamp(signal_image(color))
    assert found is not None and found['color'] == expected


def test_floor_reflection_and_a_colored_rectangle_do_not_make_a_signal():
    image = np.full((480, 640, 3), 170, dtype=np.uint8)
    image[300:] = (0, 0, 255)
    assert find_signal_lamp(image) is None
    image[50:150, 50:150] = (0, 0, 255)
    assert find_signal_lamp(image) is None


def test_round_color_without_dark_housing_is_not_enough():
    image = np.full((480, 640, 3), 170, dtype=np.uint8)
    cv2.circle(image, (112, 103), 28, (0, 0, 255), -1)
    cv2.circle(image, (112, 103), 20, (255, 255, 255), -1)
    assert find_signal_lamp(image) is None


def test_green_needs_three_new_confirmations():
    confirmation = ColorConfirmation()
    assert confirmation.update('Red') == 'Unknown'
    assert confirmation.update('Red') == 'Red'
    assert [confirmation.update('Green') for _ in range(3)] == ['Unknown', 'Unknown', 'Green']


def test_ros_detector_uses_fallback_without_yolo_and_rejects_old_or_repeated_frames():
    rclpy.init()
    node = TrafficLightDetector()
    outputs, presence = [], []
    from types import SimpleNamespace as NS
    node.publisher = NS(publish=outputs.append)
    node.presence_pub = NS(publish=presence.append)
    try:
        image = node.cv_bridge.cv2_to_imgmsg(signal_image((0, 0, 255)), encoding='bgr8')
        detections = DetectionArray()

        def process(age=0.0):
            image.header.stamp = Time(nanoseconds=node.get_clock().now().nanoseconds - int(age * 1e9)).to_msg()
            detections.header = image.header
            node.sync_callback(detections, image)

        process()
        assert outputs[-1].data == 'Unknown'
        assert presence[-1].data is True
        count = len(outputs)
        node.sync_callback(detections, image)
        assert len(outputs) == count
        assert len(presence) == count
        process()
        assert outputs[-1].data == 'Red'
        count = len(outputs)
        process(age=2.0)
        assert len(outputs) == count
        assert len(presence) == count
        image = node.cv_bridge.cv2_to_imgmsg(
            np.full((480, 640, 3), 170, dtype=np.uint8), encoding='bgr8')
        process()
        assert presence[-1].data is False
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


REFERENCE = Path('/home/donghyun/Camera Image_screenshot_08.11.2026.png')


@pytest.mark.skipif(not REFERENCE.exists(), reason='User reference photo not installed on this PC')
def test_user_reference_photo_is_red_even_without_a_yolo_signal_box():
    found = find_signal_lamp(cv2.imread(str(REFERENCE)))
    assert found is not None and found['color'] == 'Red'
    left, top, right, bottom = found['box']
    assert left <= 130 <= right and top <= 103 <= bottom
