import cv2
import random
import numpy as np
from typing import Tuple
import sys, os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray, BoundingBox2D, Detection
from std_msgs.msg import Bool, String

from .lib import camera_perception_func_lib as CPFL
from .lamp_change import LampChangeDetector, find_lamp_blob, read_lamps

# ---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_IMAGE_TOPIC_NAME = "image_raw"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_traffic_light_info"

# ----------------------------------------------

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector_node')

        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_image_topic = self.declare_parameter('sub_image_topic', SUB_IMAGE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', False).value
        # 횡단보도 박스 아래쪽 끝이 이 y(픽셀, 640x480 기준)보다 아래로 오면 '가까움'
        self.crosswalk_stop_y = self.declare_parameter('crosswalk_stop_y', 440).value
        # A crosswalk at the stop line is >120 px tall; a white line across the road at the bottom
        # of the image was detected as a thin crosswalk and stopped the car on a straight.
        self.crosswalk_min_height = self.declare_parameter('crosswalk_min_height', 60).value
        # YOLO boxes below this score are ignored (dark red objects were reported as lights).
        self.min_score = self.declare_parameter('min_score', 0.5).value
        # Crops are saved whenever the reported colour changes, for tuning the colour ranges.
        self.sample_dir = self.declare_parameter('sample_dir', os.path.expanduser('~/traffic_light_samples')).value
        self.change = LampChangeDetector()
        self.stopped = False
        self.last_color = None
        self.last_near = None

        self.cv_bridge = CvBridge()

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.detection_sub = Subscriber(self, DetectionArray, self.sub_detection_topic, qos_profile=self.qos_profile)
        self.image_sub = Subscriber(self, Image, self.sub_image_topic, qos_profile=self.qos_profile)
        self.ts = ApproximateTimeSynchronizer([self.detection_sub, self.image_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

        self.publisher = self.create_publisher(String, self.pub_topic, self.qos_profile)
        self.crosswalk_pub = self.create_publisher(Bool, 'crosswalk_near', self.qos_profile)
        # Any crosswalk in view (seen ~4 s before the stop line): the planner slows down early.
        self.crosswalk_ahead_pub = self.create_publisher(Bool, 'crosswalk_ahead', self.qos_profile)
        self.state_sub = self.create_subscription(String, 'control_state', self.state_callback, 1)

    def state_callback(self, msg: String):
        self.stopped = msg.data.startswith('stopped at crosswalk')

    def sync_callback(self, detection_msg: DetectionArray, image_msg: Image):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        hsv_ranges = {
            'red1': (np.array([0, 100, 95]), np.array([15, 255, 255])),
            'red2': (np.array([160, 100, 95]), np.array([179, 255, 255])),
            'yellow': (np.array([20, 100, 95]), np.array([30, 255, 255])),
            'green': (np.array([40, 100, 95]), np.array([90, 255, 255]))
        }

        # 'None': 신호등 미검출, Red/Yellow/Green/Unknown: 가장 점수가 높은 신호등의 색
        lights = [d for d in detection_msg.detections
                  if d.class_name.casefold().replace('_', '') == 'trafficlight' and d.score >= self.min_score]
        light = max(lights, key=lambda d: d.score) if lights else None
        color, box = 'None', None
        if light is not None:
            cx, cy, w, h = light.bbox.center.position.x, light.bbox.center.position.y, light.bbox.size.x, light.bbox.size.y
            box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
            # Lit lamp read inside the box only (amber counts as its own colour).
            lamp, _ = read_lamps(cv_image, box)
            color = lamp if lamp is not None else CPFL.get_traffic_light_color(cv_image, light.bbox, hsv_ranges)
        elif any(d.class_name.casefold() == 'crosswalk' for d in detection_msg.detections):
            # YOLO misses the light at mid distance; near a crosswalk look for a lit red/amber lamp.
            lamp, lamp_box = find_lamp_blob(cv_image)
            if lamp is not None:
                color, box = lamp, lamp_box
        # While stopped at red, a dark lamp lighting up counts as the light changing.
        changed = self.change.update(cv_image, box, color, self.stopped, time.monotonic())
        note = ''
        if changed != color:
            note = (f' (lamp pixels red {self.change.last_read[0]:.0%} green {self.change.last_read[1]:.0%})'
                    if self.change.last_read else f' (lamp change: {self.change.last_blob} px lit up)')
        color = changed

        self.publisher.publish(String(data=color))
        if color != self.last_color:
            score = f' (score {light.score:.2f})' if light is not None else ''
            self.get_logger().info(f'Traffic light: {color}{score}{note}')
            self.last_color = color
            self.save_sample(cv_image, box, color)

        crosswalks = [d for d in detection_msg.detections if d.class_name.casefold() == 'crosswalk']
        bottoms = [d.bbox.center.position.y + d.bbox.size.y / 2 for d in crosswalks
                   if d.bbox.size.y >= self.crosswalk_min_height]
        near = bool(bottoms) and max(bottoms) >= self.crosswalk_stop_y
        self.crosswalk_pub.publish(Bool(data=near))
        self.crosswalk_ahead_pub.publish(Bool(data=bool(crosswalks)))
        if near != self.last_near:
            where = f' (bottom y={max(bottoms):.0f}, stop line {self.crosswalk_stop_y})' if bottoms else ''
            self.get_logger().info(f'Crosswalk near: {near}{where}')
            self.last_near = near

        if self.show_image:
            self.show(cv_image, color, light, crosswalks, near)

    def save_sample(self, image, box, color):
        if box is None or not self.sample_dir:
            return
        try:
            os.makedirs(self.sample_dir, exist_ok=True)
            x0, y0, x1, y1 = box
            pad = 40
            crop = image[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            if crop.size:
                cv2.imwrite(os.path.join(self.sample_dir, f'{time.strftime("%H%M%S")}_{color}.png'), crop)
        except OSError as error:
            self.get_logger().warning(f'Could not save traffic light sample: {error}')

    def show(self, image, color, light, crosswalks, near):
        line_color = (0, 0, 255) if near else (0, 255, 255)
        for d in crosswalks:
            cx, cy, w, h = d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size.x, d.bbox.size.y
            cv2.rectangle(image, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), (255, 128, 0), 2)
        y = int(self.crosswalk_stop_y)
        cv2.line(image, (0, y), (image.shape[1], y), line_color, 2)
        cv2.putText(image, 'crosswalk NEAR' if near else 'stop line', (10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
        bgr = {'Red': (0, 0, 255), 'Yellow': (0, 255, 255), 'Green': (0, 255, 0)}.get(color, (200, 200, 200))
        if light is not None:
            cx, cy = light.bbox.center.position.x, light.bbox.center.position.y
            w, h = light.bbox.size.x, light.bbox.size.y
            cv2.rectangle(image, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), bgr, 2)
        label = 'no traffic light' if light is None else f'{color} {light.score:.2f}'
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        cv2.imshow('traffic_light', image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
