"""Lane-region centers using the skeleton's 640x480 perspective calibration."""

import cv2
import numpy as np


TARGET_HEIGHTS = (5, 55, 105)


def region_centers(roi, lane_width=300):
    centers = []
    for y in TARGET_HEIGHTS:
        strip = roi[max(0, y - 5):min(roi.shape[0], y + 5)]
        if strip.size == 0:
            centers.append(None)
            continue
        xs = np.flatnonzero(np.count_nonzero(strip, axis=0) >= max(1, strip.shape[0] // 3))
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        run = max(runs, key=len)
        if len(run) < 40:
            centers.append(None)
        elif run[0] == 0 and run[-1] < roi.shape[1] - 1:
            centers.append(max(0.0, float(run[-1]) - lane_width / 2))
        elif run[-1] == roi.shape[1] - 1 and run[0] > 0:
            centers.append(min(float(roi.shape[1] - 1), float(run[0]) + lane_width / 2))
        else:
            centers.append(float((run[0] + run[-1]) / 2))
    return centers


class LaneCenterTracker:
    def __init__(self, pair_timeout=2.0, lane_width=300, max_center_step=110,
                 max_accepted_step=55):
        self.pair_timeout = pair_timeout
        self.lane_width = lane_width
        self.max_center_step = max_center_step
        self.max_accepted_step = max_accepted_step
        self.pairs = {}
        self.last_centers = {}
        self.debug = {}

    def reset(self, keep_pairs=False):
        if not keep_pairs:
            self.pairs.clear()
        self.last_centers.clear()

    def update(self, detections, target_lane, now):
        self.debug = dict(target_lane=target_lane,
                          detected_lanes=[d.class_name for d in detections
                                          if d.class_name.casefold() in ('lane1', 'lane2')],
                          rejected=[], limited=[])
        masks, areas = {}, {}
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
            area = cv2.countNonZero(canvas)
            if area > areas.get(name, 0):
                masks[name], areas[name] = canvas, area
        centers, rois = {}, {}
        src = np.float32([[238, 316], [402, 313], [501, 476], [155, 476]])
        dst = np.float32([[192, 0], [448, 0], [448, 480], [192, 480]])
        transform = cv2.getPerspectiveTransform(src, dst)
        for name, mask in masks.items():
            bird = cv2.warpPerspective(mask, transform, (640, 480), flags=cv2.INTER_NEAREST)
            rois[name] = bird[300:]
            centers[name] = region_centers(rois[name], self.lane_width)
        one = centers.get('lane1', [None] * 3)
        two = centers.get('lane2', [None] * 3)
        self.debug['raw_centers'] = centers
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
            x = chosen[index]
            previous = self.last_centers.get(index)
            if x is not None and previous is not None and 0 <= now - previous[1] < self.pair_timeout:
                change = x - previous[0]
                if abs(change) > self.max_center_step:
                    chosen[index] = None
                    self.debug['rejected'].append(index)
                elif abs(change) > self.max_accepted_step:
                    chosen[index] = previous[0] + float(np.sign(change)) * self.max_accepted_step
                    estimated = True
                    self.debug['limited'].append(index)
        valid = [i for i, x in enumerate(chosen) if x is not None]
        roi = rois.get(target_lane, np.zeros((180, 640), dtype=np.uint8))
        if len(valid) < 2:
            self.debug['reason'] = 'no_lane_detection' if not masks else 'too_few_valid_target_points'
            return None, 'lost', masks, roi
        for index in valid:
            self.last_centers[index] = (chosen[index], now)
        for index, x in enumerate(chosen):
            if x is None:
                chosen[index] = float(np.interp(TARGET_HEIGHTS[index],
                                                [TARGET_HEIGHTS[i] for i in valid],
                                                [chosen[i] for i in valid]))
                estimated = True
        self.debug['reason'] = 'estimated' if estimated else 'direct'
        return chosen, 'estimated' if estimated else 'direct', masks, roi
