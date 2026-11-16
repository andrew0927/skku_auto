"""Color classification must not invent a red light from an empty/black crop."""

import cv2
import numpy as np
import pytest
import rclpy
from interfaces_pkg.msg import Detection, DetectionArray
from types import SimpleNamespace as NS

from camera_perception_pkg.traffic_light_detector_node import TrafficLightDetector
from rclpy.parameter import Parameter


@pytest.mark.parametrize('color, expected', [
    ((0, 0, 255), 'Red'), ((0, 255, 0), 'Green'), ((0, 255, 255), 'Yellow'),
    ((0, 0, 0), 'Unknown'),
])
def test_signal_color_and_floor_filter(color, expected):
    rclpy.init()
    node = TrafficLightDetector(parameter_overrides=[
        Parameter('confirmation_frames', value=1), Parameter('green_confirmation_frames', value=1)])
    results = []
    node.publisher = NS(publish=results.append)
    try:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[40:80, 300:340] = color
        frame = node.cv_bridge.cv2_to_imgmsg(image, encoding='bgr8')
        detection = Detection(class_name='traffic_light', score=0.99)
        detection.bbox.center.position.x, detection.bbox.center.position.y = 320.0, 60.0
        detection.bbox.size.x, detection.bbox.size.y = 40.0, 40.0
        message = DetectionArray(detections=[detection])
        def process():
            frame.header.stamp = node.get_clock().now().to_msg()
            message.header = frame.header
            node.sync_callback(message, frame)
        process()
        assert results[-1].data == expected
        detection.bbox.center.position.y = 400.0
        process()
        assert results[-1].data == 'None'
        detection.bbox.center.position.y = 60.0
        detection.bbox.center.position.x = -100.0
        process()
        assert results[-1].data == 'Unknown'
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
