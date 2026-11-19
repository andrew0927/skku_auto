"""Track lane2, with lane1 fallback only while the measured lane spacing is fresh."""

import time
from collections import Counter, deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String
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
        # Cars from vehicle_detector_node, with their lane read from the dashed centre line.
        # Box bottom below this = near. HybridNets sees the track's cars ~3 s ahead (bottom ~y 280 at
        # speed 150); deciding only at y 330 left ~1.3 m for the lane change (11-18 laps).
        self.near_y = self.declare_parameter('vehicle_near_y', 260).value
        # A car whose lane cannot be read counts as ours only closer and straight ahead.
        self.center_near_y = self.declare_parameter('vehicle_center_near_y', 330).value
        self.masks = {}
        # The lane the car is on, from the segmentation: lane1/lane2 mask right above the hood at the
        # image centre, majority of the last ~1 s (checked against the images of 11-18: correct
        # wherever it reads; unreadable only when a person or a car covers that spot).
        self.under_history = deque(maxlen=10)
        self.under_pub = self.create_publisher(String, 'lane_under_car', 1)
        self.vehicle_sub = self.create_subscription(DetectionArray, 'vehicle_detections', self.vehicle_callback, 1)
        self.own_pub = self.create_publisher(Bool, 'camera_obstacle_info', 1)     # near car in my lane
        self.other_pub = self.create_publisher(Bool, 'camera_obstacle_other', 1)  # near car in the other lane
        # [own x0 y0 x1 y1, nearest x0 y0 x1 y1] (NaN when none): the nearest car in my lane, and the
        # nearest car anywhere (followed by the planner while it passes, when its lane changes).
        self.box_pub = self.create_publisher(Float32MultiArray, 'camera_obstacle_box', 1)
        self.last_report = None

    def target_callback(self, msg):
        if msg.data in ('lane1', 'lane2') and msg.data != self.target_lane:
            self.target_lane = msg.data
            self.tracker.switch_target(time.monotonic())  # keep spacings: the new lane is estimated

    def detection_callback(self, msg):
        centers, source, masks, roi = self.tracker.update(msg.detections, self.target_lane, time.monotonic())
        self.masks = masks
        self.under_history.append(lane_under_car(masks))
        self.under_pub.publish(String(data=self.my_lane() or 'none'))
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
            for row, (name, color) in enumerate([('lane1', (255, 180, 0)), ('lane2', (0, 255, 0)),
                                                 ('crosswalk', (0, 0, 255))]):
                if name in masks:
                    contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(edges, contours, -1, color, 2)
                cv2.putText(edges, name, (10, 25 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            preview = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            if centers is not None:
                for x, y in zip(centers, TARGET_HEIGHTS):
                    cv2.circle(preview, (round(x), y), 5, (0, 0, 255), -1)
            cv2.putText(preview, f'{self.target_lane}: {source}', (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('lane_edge_image', edges)
            cv2.imshow('lane_roi', preview)
            cv2.waitKey(1)


    def my_lane(self):
        """Majority (70 %, 3+ readings) of the recent under-car readings, else None."""
        votes = Counter(v for v in self.under_history if v)
        if votes:
            lane, count = votes.most_common(1)[0]
            if count >= 3 and count >= 0.7 * sum(votes.values()):
                return lane
        return None

    def vehicle_callback(self, msg):
        """Near cars: in my lane (the followed lane, which the planner keeps) or in the other one.
        The dashed centre line beside the car decides (Detection.id from the vehicle detector); a
        car sitting on the line blocks us too. Without visible dashes only a close car straight
        ahead counts as ours."""
        own = other = False
        own_box = near_box = None
        mine_lane = self.my_lane() or self.target_lane
        for car in msg.detections:
            cx, cy = car.bbox.center.position.x, car.bbox.center.position.y
            w, h = car.bbox.size.x, car.bbox.size.y
            bottom = cy + h / 2
            box = (cx - w / 2, cy - h / 2, cx + w / 2, bottom)
            if bottom >= 240 and (near_box is None or bottom > near_box[3]):
                near_box = box
            if bottom < self.near_y:
                continue
            side = car.id
            if side not in ('lane1', 'lane2'):
                # Centre line says "on the line" or nothing: the lane mask under the car decides (on
                # 11-19 a lane1 car at the image edge read "line" and the car turned into it).
                side = lane_at(self.masks, box) or side
            if side in ('lane1', 'lane2'):
                mine = side == mine_lane
            elif side == 'line':
                mine = min(box[2], 440) - max(box[0], 200) >= 40   # only when in front of us
            else:
                # Close, straight ahead and on the road (a lane mask under it): a car parked on the
                # mat beside the road on 11-18 started a lane change right before the light.
                mine = (bottom >= self.center_near_y and cx - w / 2 < 320 < cx + w / 2
                        and lane_at(self.masks, box) is not None)
                if not mine:
                    continue          # lane unknown and not straight ahead: neither lane
            if mine and bottom >= 400 and min(box[2], 440) - max(box[0], 200) < 40:
                # Close but off to the side: a car we are passing (11-17 131 s, x 0-190 at y 470),
                # not one ahead. The manual demo turned for cars ahead at y 305-349.
                continue
            if mine:
                own = True
                if own_box is None or bottom > own_box[3]:
                    own_box = box
            else:
                other = True
        nan = float('nan')
        self.box_pub.publish(Float32MultiArray(data=[float(v) for v in (own_box or (nan,) * 4) + (near_box or (nan,) * 4)]))
        self.own_pub.publish(Bool(data=own))
        self.other_pub.publish(Bool(data=other))
        if (own, other) != self.last_report:
            self.get_logger().info(f'Car near: my lane ({mine_lane}) {own}, other lane {other}')
            self.last_report = (own, other)


def lane_under_car(masks, x0=270, x1=370, y0=400, y1=445):
    """lane1/lane2 covering the road just above the hood at the image centre, or None."""
    fractions = {name: float((masks[name][y0:y1, x0:x1] > 0).mean()) for name in ('lane1', 'lane2') if name in masks}
    if not fractions:
        return None
    best = max(fractions, key=fractions.get)
    if fractions[best] >= 0.5 and all(v < 0.2 for k, v in fractions.items() if k != best):
        return best
    return None


def lane_at(masks, box, band=15, gap=3, min_fraction=0.5):
    """lane1/lane2 of a car: the road strip just in front of its box (box bottom, car width) must
    lie at least min_fraction on one lane mask. None when it does not (car off the road or half out
    of view) or when the strip is below the image (car right in front: no road visible)."""
    x0, y0, x1, y1 = box
    best, best_fraction = None, min_fraction
    for name in ('lane1', 'lane2'):
        mask = masks.get(name)
        if mask is None:
            continue
        height, width = mask.shape[:2]
        top, bottom = int(y1) + gap, min(height, int(y1) + gap + band)
        left, right = max(0, int(x0)), min(width, int(x1))
        if bottom - top < 3 or right - left < 3:
            continue
        fraction = float(np.count_nonzero(mask[top:bottom, left:right])) / ((bottom - top) * (right - left))
        if fraction >= best_fraction:
            best, best_fraction = name, fraction
    return best


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
