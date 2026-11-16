"""YOLO signal colors with a geometric fallback for missed horizontal lamps."""

import json

import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray
from std_msgs.msg import Bool, String

from .lib import camera_perception_func_lib as CPFL
from .signal_geometry import HSV_RANGES, ColorConfirmation, find_signal_lamp


class TrafficLightDetector(Node):
    def __init__(self, **kwargs):
        super().__init__('traffic_light_detector_node', **kwargs)
        detection_topic = self.declare_parameter('sub_detection_topic', 'detections').value
        image_topic = self.declare_parameter('sub_image_topic', 'image_raw').value
        output_topic = self.declare_parameter('pub_topic', 'yolov8_traffic_light_info').value
        self.max_height_ratio = self.declare_parameter('max_height_ratio', 0.45).value
        self.fallback_enabled = self.declare_parameter('enable_lamp_fallback', True).value
        self.source_timeout = self.declare_parameter('source_timeout', 1.0).value
        self.confirmation = ColorConfirmation(
            self.declare_parameter('confirmation_frames', 2).value,
            self.declare_parameter('green_confirmation_frames', 3).value)
        if not 0.1 <= self.max_height_ratio <= 0.6 or self.source_timeout <= 0:
            raise ValueError('Invalid traffic light region or timeout')
        self.cv_bridge = CvBridge()
        self.detection_sub = Subscriber(self, DetectionArray, detection_topic, qos_profile=1)
        self.image_sub = Subscriber(self, Image, image_topic, qos_profile=1)
        self.ts = ApproximateTimeSynchronizer([self.detection_sub, self.image_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        self.publisher = self.create_publisher(String, output_topic, 1)
        self.presence_pub = self.create_publisher(Bool, 'traffic_light_presence', 1)
        self.debug_pub = self.create_publisher(String, 'traffic_light_debug', 1)
        self.preview_pub = self.create_publisher(Image, 'traffic_light_preview', 1)
        self.last_stamp = 0.0
        self.last_display = None

    def sync_callback(self, detection_msg, image_msg):
        stamps = [m.header.stamp.sec + m.header.stamp.nanosec / 1e9
                  for m in (detection_msg, image_msg)]
        stamp = min(stamps)
        now = self.get_clock().now().nanoseconds / 1e9
        if stamp <= self.last_stamp or not -0.05 <= now - stamp <= self.source_timeout:
            return
        self.last_stamp = stamp
        image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        candidates, model_count = [], 0
        for detection in detection_msg.detections:
            if detection.class_name.casefold().replace('_', '') != 'trafficlight':
                continue
            model_count += 1
            box = detection.bbox
            if box.center.position.y > image.shape[0] * self.max_height_ratio:
                continue
            color = CPFL.get_traffic_light_color(image, box, HSV_RANGES)
            candidates.append(dict(color=color, source='yolo', score=detection.score,
                                   box=[box.center.position.x - box.size.x / 2,
                                        box.center.position.y - box.size.y / 2,
                                        box.center.position.x + box.size.x / 2,
                                        box.center.position.y + box.size.y / 2]))
        if self.fallback_enabled:
            fallback = find_signal_lamp(image, self.max_height_ratio)
            if fallback is not None:
                candidates.append(fallback)
        priority = {'Red': 3, 'Yellow': 2, 'Green': 1, 'Unknown': 0}
        selected = max(candidates, key=lambda c: (priority[c['color']], c.get('score', 0))) if candidates else None
        raw = selected['color'] if selected else 'None'
        color = self.confirmation.update(raw)
        source = selected['source'] if selected else 'none'
        present = selected is not None
        debug = dict(raw=raw, confirmed=color, source=source, present=present, yolo_objects=model_count,
                     box=selected['box'] if selected else None, stamp=stamp)
        self.publisher.publish(String(data=color))
        self.presence_pub.publish(Bool(data=present))
        self.debug_pub.publish(String(data=json.dumps(debug)))
        display = (color, source, model_count)
        if display != self.last_display:
            self.get_logger().info(f'Traffic light: {color} ({source}, YOLO objects={model_count})')
            self.last_display = display
        if self.preview_pub.get_subscription_count():
            import cv2
            preview = image.copy()
            if selected is not None:
                x0, y0, x1, y1 = map(lambda v: int(round(v)), selected['box'])
                cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(preview, f'{color} ({source})', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            msg = self.cv_bridge.cv2_to_imgmsg(preview, encoding='bgr8')
            msg.header = image_msg.header
            self.preview_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
