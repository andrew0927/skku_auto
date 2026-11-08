#!/usr/bin/env python3
"""Run the lane tracker on cached detections (offline_detect.py) and report how often it is lost.

usage: eval_lane_tracker.py <lap dir> [START END]
"""

import pickle
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from camera_perception_pkg.lane_geometry import LaneCenterTracker


def to_detection(d):
    x0, y0, x1, y1 = d['bbox']
    return NS(class_name=d['class_name'], score=d['score'],
              bbox=NS(center=NS(position=NS(x=(x0 + x1) / 2, y=(y0 + y1) / 2)), size=NS(x=x1 - x0, y=y1 - y0)),
              mask=NS(width=640, height=480, data=[NS(x=float(x), y=float(y)) for x, y in d['mask']]))


def main(lap, start=0.0, end=1e9):
    frames = pickle.load(open(Path(lap) / 'detections.pkl', 'rb'))
    tracker = LaneCenterTracker()
    sources, lost_runs, run = Counter(), [], 0
    for t, dets in frames:
        if not float(start) <= t <= float(end):
            continue
        dets = [to_detection(d) for d in dets if d['score'] >= 0.5]
        _, source, _, _ = tracker.update(dets, 'lane2', t)
        sources[source] += 1
        run = run + 1 if source == 'lost' else 0
        if run == 1:
            lost_runs.append([t, t])
        elif run > 1:
            lost_runs[-1][1] = t
    total = sum(sources.values())
    print(f'{total} frames: ' + ', '.join(f'{k} {100 * v / total:.1f}%' for k, v in sources.most_common()))
    long_runs = [(a, b) for a, b in lost_runs if b - a >= 0.4]
    print(f'lost stretches >= 0.4 s (would stop the car): {len(long_runs)}  '
          + ' '.join(f'{a:.1f}-{b:.1f}' for a, b in long_runs[:12]))


if __name__ == '__main__':
    main(*sys.argv[1:])
