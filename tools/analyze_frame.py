"""Offline perception report for one camera image. No ROS nodes or ports opened."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import torch
from ultralytics import YOLO

from camera_perception_pkg.lane_geometry import LaneCenterTracker
from camera_perception_pkg.signal_geometry import find_signal_lamp
from camera_perception_pkg.lane_inference import GreenLaneInference, merge_detection_groups
from decision_making_pkg.crosswalk_stop import crosswalk_edge_ratio, RedLightStopController


def parse_result(result, names, shape):
    detections, objects = [], []
    for index, box in enumerate(result.boxes):
        x, y, w, h = map(float, box.xywh[0])
        name, score = names[int(box.cls)], float(box.conf)
        polygon = result.masks.xy[index] if result.masks is not None else []
        detection = NS(class_name=name, score=score,
                       bbox=NS(center=NS(position=NS(x=x, y=y)), size=NS(x=w, y=h)),
                       mask=NS(width=shape[1], height=shape[0],
                               data=[NS(x=float(px), y=float(py)) for px, py in polygon]))
        if score >= 0.5:
            detections.append(detection)
        objects.append(dict(name=name, confidence=score, xyxy=list(map(float, box.xyxy[0]))))
    return detections, objects


def lane_report(detections):
    lanes = {}
    for target in ('lane1', 'lane2'):
        tracker = LaneCenterTracker()
        centers, source, _, _ = tracker.update(detections, target, 0.0)
        lanes[target] = dict(source=source, centers=centers, details=tracker.debug)
    return lanes


def analyze(image_path, model_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Cannot read image: {image_path}')
    original_shape = list(image.shape)
    # Match image_publisher_node before YOLO, including its fixed aspect ratio.
    image = cv2.resize(image, (640, 480))
    torch.set_num_threads(4)
    model = YOLO(str(model_path))
    result = model.predict(source=image, conf=0.1, device='cpu', verbose=False)[0]
    detections, objects = parse_result(result, model.names, image.shape)
    recovery = GreenLaneInference(model)
    groups, debug = recovery.predict(image)
    effective = merge_detection_groups([parse_result(r, model.names, image.shape)[0] for r in groups],
                                       prefer_recovered_lanes=debug['prefer_recovered_lanes'])
    _, steady_debug = recovery.predict(image)
    position = crosswalk_edge_ratio(detections, 'far_edge')
    threshold = RedLightStopController().stop_ratio
    return dict(image=str(image_path), original_shape=original_shape,
                processing_shape=list(image.shape), model=str(model_path), classes=model.names,
                detections=objects, lane_acceptance_threshold=0.5, lanes=lane_report(detections),
                recovery_inference=debug, steady_inference=steady_debug,
                effective_detections=[dict(name=d.class_name, confidence=d.score) for d in effective],
                effective_lanes=lane_report(effective),
                effective_crosswalk_far_edge_ratio=crosswalk_edge_ratio(effective, 'far_edge'),
                signal_fallback=find_signal_lamp(image), crosswalk_far_edge_ratio=position,
                stop_y_ratio=threshold, stop_y_pixels=round(threshold * image.shape[0]),
                would_stop_on_red=position is not None and position >= threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'best.pt')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(analyze(args.image, args.model), indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
