#!/usr/bin/env python3
"""Run YOLO on every frame of a recorded lap and cache the detections (no ROS nodes needed).

usage: offline_detect.py <lap dir> [--conf 0.25]
Writes <lap>/detections.pkl: list of (time since first frame, [detection dicts]).
Each detection has class_name, score, bbox (x0, y0, x1, y1) and mask polygon (N x 2).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_lap import read_bag  # noqa: E402

WS = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lap')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    lap = Path(args.lap)
    images = read_bag(lap / 'bag', ['/image_raw'])['/image_raw']
    model = YOLO(str(WS / 'best.pt'))
    t0 = images[0][0]
    out = []
    for t, msg in images:
        frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        result = model(frame, conf=args.conf, verbose=False)[0]
        dets = []
        for k in range(len(result.boxes)):
            mask = result.masks.xy[k] if result.masks is not None else np.zeros((0, 2))
            dets.append({'class_name': result.names[int(result.boxes.cls[k])],
                         'score': float(result.boxes.conf[k]),
                         'bbox': tuple(float(v) for v in result.boxes.xyxy[k]),
                         'mask': np.asarray(mask, dtype=np.float32)})
        out.append((t - t0, dets))
    with open(lap / 'detections.pkl', 'wb') as f:
        pickle.dump(out, f)
    print(f'{len(out)} frames -> {lap / "detections.pkl"}')


if __name__ == '__main__':
    main()
