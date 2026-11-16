# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import List, Dict
import json
import time
import numpy as np

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints
from torch import cuda

from sensor_msgs.msg import Image
from interfaces_pkg.msg import Point2D
from interfaces_pkg.msg import BoundingBox2D
from interfaces_pkg.msg import Mask
from interfaces_pkg.msg import KeyPoint2D
from interfaces_pkg.msg import KeyPoint2DArray
from interfaces_pkg.msg import Detection
from interfaces_pkg.msg import DetectionArray

from std_srvs.srv import SetBool
from std_msgs.msg import String
from .lane_inference import GreenLaneInference, merge_detection_groups


class Yolov8Node(LifecycleNode):

    def __init__(self, **kwargs) -> None:
        super().__init__("yolov8_node", **kwargs)
        
        #---------------Variable Setting---------------
        # 딥러닝 모델 pt 파일명 작성
        #self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("model", "best.pt")
        
        # 추론 하드웨어 선택 (cpu / gpu) 
        self.declare_parameter("device", "auto")
        #self.declare_parameter("device", "cuda:0")
        #----------------------------------------------
        
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("enable", True)
        self.declare_parameter('enable_green_grayscale_fallback', True)
        self.target_lane = self.declare_parameter('target_lane', 'lane2').value
        self.declare_parameter('green_context_hold', 1.0)
        self.declare_parameter('green_color_probe_interval', 0.4)
        self.declare_parameter('green_color_confirmations', 3)
        self.inference = None
        self.signal_color = None
        self.last_signal_time = None
        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.RELIABLE)

        self.get_logger().info('Yolov8Node created')
        self.last_inference_mode = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Configuring {self.get_name()}')

        self.model = self.get_parameter(
            "model").get_parameter_value().string_value

        self.device = self.get_parameter(
            "device").get_parameter_value().string_value
        if self.device == 'auto':
            self.device = 'cuda:0' if cuda.is_available() else 'cpu'

        self.threshold = self.get_parameter(
            "threshold").get_parameter_value().double_value

        self.enable = self.get_parameter(
            "enable").get_parameter_value().bool_value

        self.reliability = self.get_parameter(
            "image_reliability").get_parameter_value().integer_value

        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self._pub = self.create_lifecycle_publisher(
            DetectionArray, "detections", 1)
        self.debug_pub = self.create_publisher(String, 'yolo_inference_debug', 1)
        self._srv = self.create_service(
            SetBool, "enable", self.enable_cb
        )
        self.cv_bridge = CvBridge()
        self.target_sub = self.create_subscription(
            String, 'target_lane', self.target_callback,
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.signal_sub = self.create_subscription(
            String, 'yolov8_traffic_light_info', self.signal_callback, 1)

        return TransitionCallbackReturn.SUCCESS

    def enable_cb(self, request, response):
        self.enable = request.data
        if self.inference is not None:
            self.inference.reset()
        response.success = True
        return response

    def target_callback(self, msg):
        if msg.data in ('lane1', 'lane2'):
            self.target_lane = msg.data
            if self.inference is not None:
                self.inference.set_target_lane(msg.data)

    def signal_callback(self, msg):
        self.signal_color, self.last_signal_time = msg.data, time.monotonic()

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Activating {self.get_name()}')

        try:
            self.yolo = YOLO(self.model)  # 모델 로딩
            self.yolo.to(self.device)
            self.yolo.fuse()
            if 'lane2' not in self.yolo.names.values():
                raise ValueError(f'Model has no lane2 class: {self.yolo.names}')
            self.yolo.predict(np.zeros((480, 640, 3), dtype=np.uint8),
                              device=self.device, verbose=False)
            self.inference = GreenLaneInference(
                self.yolo, self.threshold, self.device,
                self.get_parameter('enable_green_grayscale_fallback').value,
                target_lane=self.target_lane,
                green_hold=self.get_parameter('green_context_hold').value,
                probe_interval=self.get_parameter('green_color_probe_interval').value,
                color_confirmations=self.get_parameter('green_color_confirmations').value)
        except FileNotFoundError:
            self.get_logger().error(f"Error: Model file '{self.model}' not found!")
            return TransitionCallbackReturn.FAILURE
        except Exception as e:
            self.get_logger().error(f"Error while loading model '{self.model}': {str(e)}")
            return TransitionCallbackReturn.FAILURE

        # subs
        self._sub = self.create_subscription(
            Image,
            "image_raw",
            self.image_cb,
            self.image_qos_profile
        )

        super().on_activate(state)
        self.get_logger().info(f'Model ready on {self.device}; classes: {self.yolo.names}')

        return TransitionCallbackReturn.SUCCESS


    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Deactivating {self.get_name()}')

        self.inference = None
        del self.yolo
        if 'cuda' in self.device:
            self.get_logger().info("Clearing CUDA cache")
            cuda.empty_cache()

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Cleaning up {self.get_name()}')

        self.destroy_publisher(self._pub)
        self.destroy_subscription(self.target_sub)
        self.destroy_subscription(self.signal_sub)

        del self.image_qos_profile

        return TransitionCallbackReturn.SUCCESS

    def parse_hypothesis(self, results: Results) -> List[Dict]:

        hypothesis_list = []

        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": self.yolo.names[int(box_data.cls)],
                "score": float(box_data.conf)
            }
            hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:

        boxes_list = []

        box_data: Boxes
        for box_data in results.boxes:

            msg = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [create_point2d(float(ele[0]), float(ele[1]))
                        for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def detections_from_result(self, results):
        hypothesis = self.parse_hypothesis(results) if results.boxes else []
        boxes = self.parse_boxes(results) if results.boxes else []
        masks = self.parse_masks(results) if results.masks else []
        keypoints = self.parse_keypoints(results) if results.keypoints else []
        detections = []
        for i in range(len(results)):
            detection = Detection()
            if results.boxes:
                detection.class_id = hypothesis[i]['class_id']
                detection.class_name = hypothesis[i]['class_name']
                detection.score = hypothesis[i]['score']
                detection.bbox = boxes[i]
            if results.masks:
                detection.mask = masks[i]
            if results.keypoints:
                detection.keypoints = keypoints[i]
            detections.append(detection)
        return detections

    def image_cb(self, msg: Image) -> None:
        output = DetectionArray()
        output.header = msg.header
        if self.enable:
            started = time.monotonic()
            try:
                image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                hint = (self.signal_color if self.last_signal_time is not None
                        and 0 <= started - self.last_signal_time <= 1.0 else None)
                groups, debug = self.inference.predict(
                    image, now=started, signal_color=hint,
                    frame_stamp=msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)
                output.detections = merge_detection_groups(
                    [self.detections_from_result(result) for result in groups],
                    prefer_recovered_lanes=debug['prefer_recovered_lanes'])
                debug.update(stamp=msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                             inference_ms=round((time.monotonic() - started) * 1000, 1))
                self.debug_pub.publish(String(data=json.dumps(debug)))
                if debug['mode'] != self.last_inference_mode:
                    self.get_logger().info(f"Lane inference mode: {debug['mode']}")
                    self.last_inference_mode = debug['mode']
            except Exception as error:
                self.get_logger().error(f'Image inference failed: {error}')
                self.inference.reset()
        else:
            self.inference.reset()
        self._pub.publish(output)


def main():
    rclpy.init()
    node = Yolov8Node()
    try:
        configured = node.trigger_configure()
        activated = node.trigger_activate() if configured == TransitionCallbackReturn.SUCCESS else None
        if activated != TransitionCallbackReturn.SUCCESS:
            raise RuntimeError('YOLO failed to activate')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
