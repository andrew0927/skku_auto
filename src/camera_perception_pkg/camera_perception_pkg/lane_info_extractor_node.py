"""Track lane2, with lane1 fallback only while the measured lane spacing is fresh."""

import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray
from .lane_geometry import LaneCenterTracker, TARGET_HEIGHTS


class Yolov8InfoExtractor(Node):
    def __init__(self, **kwargs):
        super().__init__('lane_info_extractor_node', **kwargs)
        self.target_lane = self.declare_parameter('target_lane', 'lane2').value
        self.show_image = self.declare_parameter('show_image', True).value
        pair_timeout = self.declare_parameter('pair_timeout', 2.0).value
        if self.target_lane not in ('lane1', 'lane2'):
            raise ValueError('target_lane must be lane1 or lane2')
        self.tracker = LaneCenterTracker(pair_timeout)
        self.cv_bridge = CvBridge()
        self.subscriber = self.create_subscription(DetectionArray, 'detections', self.detection_callback, 1)
        self.publisher = self.create_publisher(LaneInfo, 'yolov8_lane_info', 1)
        self.source_pub = self.create_publisher(String, 'lane_tracking_source', 1)
        self.roi_pub = self.create_publisher(Image, 'roi_image', 1)
        self.reset_sub = self.create_subscription(String, 'target_lane', self.target_callback, 1)

    def target_callback(self, msg):
        if msg.data in ('lane1', 'lane2') and msg.data != self.target_lane:
            self.target_lane = msg.data
            self.tracker.reset()

    def detection_callback(self, msg):
        centers, source, masks, roi = self.tracker.update(msg.detections, self.target_lane, time.monotonic())
        lane = LaneInfo()
        lane.header = msg.header
        lane.tracking_source = source
        if centers is not None:
            for x, y in zip(centers, TARGET_HEIGHTS):
                lane.target_points.append(TargetPoint(target_x=round(x), target_y=y))
        self.source_pub.publish(String(data=source))
        self.publisher.publish(lane)
        image_msg = self.cv_bridge.cv2_to_imgmsg(roi, encoding='mono8')
        image_msg.header = msg.header
        self.roi_pub.publish(image_msg)
        if self.show_image:
            edges = np.zeros((480, 640, 3), dtype=np.uint8)
            for name, color in [('lane1', (255, 180, 0)), ('lane2', (0, 255, 0))]:
                if name in masks:
                    contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(edges, contours, -1, color, 2)
                cv2.putText(edges, name, (10, 25 if name == 'lane1' else 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            preview = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            if centers is not None:
                for x, y in zip(centers, TARGET_HEIGHTS):
                    cv2.circle(preview, (round(x), y), 5, (0, 0, 255), -1)
            cv2.putText(preview, f'{self.target_lane}: {source}', (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('lane_edge_image', edges)
            cv2.imshow('lane_roi', preview)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
