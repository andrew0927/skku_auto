"""Lane-region centers using the skeleton's 640x480 perspective calibration."""

import cv2
import numpy as np


TARGET_HEIGHTS = (5, 55, 105)


def region_centers(roi):
    centers = []
    for y in TARGET_HEIGHTS:
        strip = roi[max(0, y - 5):min(roi.shape[0], y + 5)]
        if strip.size == 0:
            centers.append(None)
            continue
        xs = np.flatnonzero(np.count_nonzero(strip, axis=0) >= max(1, strip.shape[0] // 3))
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        run = max(runs, key=len)
        centers.append(float((run[0] + run[-1]) / 2) if len(run) >= 10 else None)
    return centers


class LaneCenterTracker:
    def __init__(self, pair_timeout=2.0):
        self.pair_timeout = pair_timeout
        self.pairs = {}

    def reset(self):
        self.pairs.clear()

    def update(self, detections, target_lane, now):
        masks = {}
        for detection in detections:
            name = detection.class_name.casefold()
            mask = detection.mask
            if name not in ('lane1', 'lane2') or len(mask.data) < 3:
                continue
            if not (0 < mask.width <= 4096 and 0 < mask.height <= 4096):
                continue
            points = np.array([(p.x, p.y) for p in mask.data], dtype=float)
            if not np.isfinite(points).all():
                continue
            canvas = np.zeros((mask.height, mask.width), dtype=np.uint8)
            points[:, 0] = np.clip(points[:, 0], 0, mask.width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, mask.height - 1)
            cv2.fillPoly(canvas, [points.astype(np.int32)], 255)
            canvas = cv2.resize(canvas, (640, 480), interpolation=cv2.INTER_NEAREST)
            masks[name] = cv2.bitwise_or(masks.get(name, np.zeros_like(canvas)), canvas)
        centers, rois = {}, {}
        src = np.float32([[238, 316], [402, 313], [501, 476], [155, 476]])
        dst = np.float32([[192, 0], [448, 0], [448, 480], [192, 480]])
        transform = cv2.getPerspectiveTransform(src, dst)
        for name, mask in masks.items():
            bird = cv2.warpPerspective(mask, transform, (640, 480), flags=cv2.INTER_NEAREST)
            rois[name] = bird[300:]
            centers[name] = region_centers(rois[name])
        one = centers.get('lane1', [None] * 3)
        two = centers.get('lane2', [None] * 3)
        for index, (left, right) in enumerate(zip(one, two)):
            if left is not None and right is not None and 80 <= abs(right - left) <= 450:
                self.pairs[index] = (right - left, now)
        other_lane = 'lane1' if target_lane == 'lane2' else 'lane2'
        chosen = list(centers.get(target_lane, [None] * 3))
        other = centers.get(other_lane, [None] * 3)
        estimated = False
        for index, x in enumerate(chosen):
            pair = self.pairs.get(index)
            if (x is None and other[index] is not None and pair is not None
                    and 0 <= now - pair[1] <= self.pair_timeout):
                x = other[index] + (pair[0] if target_lane == 'lane2' else -pair[0])
                chosen[index] = x if 0 <= x < 640 else None
                estimated = True
        valid = [i for i, x in enumerate(chosen) if x is not None]
        roi = rois.get(target_lane, np.zeros((180, 640), dtype=np.uint8))
        if len(valid) < 2:
            return None, 'lost', masks, roi
        for index, x in enumerate(chosen):
            if x is None:
                chosen[index] = float(np.interp(TARGET_HEIGHTS[index],
                                                [TARGET_HEIGHTS[i] for i in valid],
                                                [chosen[i] for i in valid]))
                estimated = True
        return chosen, 'estimated' if estimated else 'direct', masks, roi
