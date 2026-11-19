"""Cars on the track from a second model run beside the lane model (best.pt has no car class).

backend 'hybridnets' (default): HybridNets (official weights converted to TorchScript, GPU when
available, about 26 ms/frame; CPU about 150 ms), as in skku_autonomous_driving where it detected the
track's ride-on cars well. hybridnets_model:=...onnx runs the ONNX model on CPU instead.
backend 'yolo': a COCO YOLO model (car/bus/truck).
It misses some frames, so a car counts only when seen in at least `min_hits` of the last `window`
frames, and boxes touching the bottom of the image are our own car's hood.

Publishes vehicle_detections (DetectionArray, header of the camera frame) with the cars kept.
"""

import os
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces_pkg.msg import BoundingBox2D, Detection, DetectionArray

from .center_line import car_side

VEHICLE_CLASSES = {'car', 'bus', 'truck'}


def keep_box(box, image_height=480, hood_y=450, hood_min_height=60):
    """A box reaching into the hood band and only a few pixels tall is the car's own bonnet."""
    x0, y0, x1, y1 = box
    return not (y1 >= hood_y and (y1 - y0) < hood_min_height)


class VehicleDetector(Node):
    def __init__(self):
        super().__init__('vehicle_detector_node')
        models = Path(__file__).resolve().parents[3] / 'models'  # ros2_ws/models (symlink install)
        self.backend = self.declare_parameter('backend', 'hybridnets').value
        model = self.declare_parameter('model', str(models / 'yolov8n.pt')).value
        hn_model = self.declare_parameter('hybridnets_model', str(models / 'hybridnets_384x512' / 'hybridnets_384x512.torchscript.pt')).value
        anchors = self.declare_parameter('hybridnets_anchors', str(models / 'hybridnets_384x512' / 'anchors_384x512.npy')).value
        self.conf = self.declare_parameter('conf', 0.5 if self.backend == 'hybridnets' else 0.3).value
        self.window = self.declare_parameter('window', 5).value
        self.min_hits = self.declare_parameter('min_hits', 2).value
        device = self.declare_parameter('device', 'auto').value
        if self.backend == 'hybridnets':
            from .hybridnets_vehicles import HybridNetsVehicles
            if not os.path.exists(hn_model):
                raise FileNotFoundError(f'HybridNets model missing: {hn_model} (see README) - or backend:=yolo')
            self.hybridnets = HybridNetsVehicles(hn_model, anchors, conf=self.conf, device=device)
            self.device, model = self.hybridnets.provider, hn_model
        else:
            from ultralytics import YOLO
            import torch
            self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
            self.model = YOLO(model)
            # Only the vehicle classes are computed; the other COCO classes are never reported.
            self.class_ids = [k for k, v in self.model.names.items() if v in VEHICLE_CLASSES]
        self.bridge = CvBridge()
        self.hits = deque(maxlen=self.window)
        self.last_cars = []
        self.publisher = self.create_publisher(DetectionArray, 'vehicle_detections', 1)
        self.create_subscription(Image, 'image_raw', self.image_callback, 1)
        self.get_logger().info(f'Vehicle model ready on {self.device}: {model}')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.backend == 'hybridnets':
            found = self.hybridnets.detect(frame)
        else:
            result = self.model(frame, conf=self.conf, classes=self.class_ids, device=self.device, verbose=False)[0]
            found = list(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()))
        cars = [(box, float(score)) for box, score in found if keep_box(box, frame.shape[0])]
        self.hits.append(bool(cars))
        if cars:
            self.last_cars = cars
        out = DetectionArray()
        out.header = msg.header
        # Seen often enough lately: report the latest boxes even on a frame the model missed.
        if sum(self.hits) >= self.min_hits:
            for (x0, y0, x1, y1), score in (cars or self.last_cars):
                # id: the car's lane from the dashed centre line beside it ('' when not seen).
                side = car_side(frame, (x0, y0, x1, y1)) if y1 >= 250 else None
                d = Detection(class_name='car', score=score, id=side or '')
                d.bbox = BoundingBox2D()
                d.bbox.center.position.x, d.bbox.center.position.y = (x0 + x1) / 2, (y0 + y1) / 2
                d.bbox.size.x, d.bbox.size.y = x1 - x0, y1 - y0
                out.detections.append(d)
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
