"""Lane-region centers using the skeleton's 640x480 perspective calibration."""

import cv2
import numpy as np


TARGET_HEIGHTS = (5, 55, 105)
ROI_TOP = 300  # bird's-eye row where the near ROI (and TARGET_HEIGHTS) start
# Farther bird's-eye rows, used only when the near rows lose the lane (mat creases, the car's
# own hood in tight curves, crosswalks): the far lane is extended straight down to the near rows.
FAR_HEIGHTS = (100, 160, 220, 280)
# The road occupies bird's-eye columns 192..448. A target height is treated as hidden by the
# crosswalk when at least half of those columns contain crosswalk pixels in its strip; counting
# only road columns keeps the slightly tilted calibration from spilling onto nearby heights.
ROAD_X = (192, 448)
CROSSWALK_MIN_COLUMNS = (ROAD_X[1] - ROAD_X[0]) // 2


def region_centers(roi, heights=TARGET_HEIGHTS):
    centers = []
    for y in heights:
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
    def __init__(self, pair_timeout=2.0, slope_timeout=1.0, switch_window=8.0):
        self.pair_timeout = pair_timeout
        self.slope_timeout = slope_timeout
        # After the followed lane changes (obstacle avoidance) the new lane is mostly outside the
        # bird's-eye ROI until the car is over it, so it is placed one lane spacing from the lane
        # still in view, using the last measured spacing however old, for switch_window seconds.
        self.switch_window = switch_window
        self.switch_time = None
        self.pairs = {}
        self.last_slope = None  # (dx per ROI row, time) of the last lane seen at >= 2 heights

    def reset(self):
        self.pairs.clear()
        self.last_slope = None
        self.switch_time = None

    def switch_target(self, now):
        """The followed lane changed: keep the measured spacings to place the new lane."""
        self.switch_time = now
        self.last_slope = None

    def spacing(self, index, now):
        pair = self.pairs.get(index)
        if pair is not None and 0 <= now - pair[1] <= self.pair_timeout:
            return pair[0]
        if self.switch_time is not None and 0 <= now - self.switch_time <= self.switch_window:
            if pair is not None:
                return pair[0]
            known = [p[0] for p in self.pairs.values()]
            return float(np.median(known)) if known else float(ROAD_X[1] - ROAD_X[0])
        return None

    def extend_straight(self, chosen, valid, now):
        """Fill missing heights by extending the visible lane centers in a straight line."""
        ys = [TARGET_HEIGHTS[i] for i in valid]
        xs = [chosen[i] for i in valid]
        if len(valid) >= 2:
            slope, intercept = np.polyfit(ys, xs, 1)
        elif self.last_slope is not None and 0 <= now - self.last_slope[1] <= self.slope_timeout:
            slope = self.last_slope[0]
            intercept = xs[0] - slope * ys[0]
        else:
            return chosen
        filled = list(chosen)
        for index, y in enumerate(TARGET_HEIGHTS):
            if filled[index] is None:
                x = float(slope * y + intercept)
                filled[index] = x if 0 <= x < 640 else None
        return filled

    def update(self, detections, target_lane, now):
        masks = {}
        crosswalk = np.zeros((480, 640), dtype=np.uint8)
        for detection in detections:
            name = detection.class_name.casefold()
            mask = detection.mask
            if name == 'crosswalk':
                # The box (not the possibly fragmented mask) marks the whole striped area.
                box = detection.bbox
                sx = 640.0 / mask.width if mask.width > 0 else 1.0
                sy = 480.0 / mask.height if mask.height > 0 else 1.0
                corners = [(box.center.position.x - box.size.x / 2) * sx, (box.center.position.y - box.size.y / 2) * sy,
                           (box.center.position.x + box.size.x / 2) * sx, (box.center.position.y + box.size.y / 2) * sy]
                if np.isfinite(corners).all() and corners[2] > corners[0] and corners[3] > corners[1]:
                    x0, y0, x1, y1 = (int(round(v)) for v in corners)
                    cv2.rectangle(crosswalk, (x0, y0), (x1, y1), 255, -1)
                continue
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
        centers, rois, far = {}, {}, {}
        src = np.float32([[238, 316], [402, 313], [501, 476], [155, 476]])
        dst = np.float32([[192, 0], [448, 0], [448, 480], [192, 480]])
        transform = cv2.getPerspectiveTransform(src, dst)
        for name, mask in masks.items():
            bird = cv2.warpPerspective(mask, transform, (640, 480), flags=cv2.INTER_NEAREST)
            rois[name] = bird[ROI_TOP:]
            centers[name] = region_centers(rois[name])
            far[name] = region_centers(bird, FAR_HEIGHTS)
        # Crosswalk stripes corrupt the lane masks. Heights the crosswalk covers are ignored
        # for both lanes and filled afterwards by extending the visible lane straight.
        covered = set()
        crosswalk_roi = None
        if crosswalk.any():
            masks['crosswalk'] = crosswalk
            crosswalk_roi = cv2.warpPerspective(crosswalk, transform, (640, 480), flags=cv2.INTER_NEAREST)[300:]
            for index, y in enumerate(TARGET_HEIGHTS):
                strip = crosswalk_roi[max(0, y - 5):y + 5, ROAD_X[0]:ROAD_X[1]]
                if strip.size and np.count_nonzero(strip.any(axis=0)) >= CROSSWALK_MIN_COLUMNS:
                    covered.add(index)
        one = list(centers.get('lane1', [None] * 3))
        two = list(centers.get('lane2', [None] * 3))
        for index in covered:
            one[index] = two[index] = None
        for index, (left, right) in enumerate(zip(one, two)):
            if left is not None and right is not None and 80 <= abs(right - left) <= 450:
                self.pairs[index] = (right - left, now)
        other_lane = 'lane1' if target_lane == 'lane2' else 'lane2'
        chosen = list(two if target_lane == 'lane2' else one)
        other = one if target_lane == 'lane2' else two
        estimated = False
        for index, x in enumerate(chosen):
            spacing = self.spacing(index, now) if x is None and other[index] is not None else None
            if spacing is not None:
                x = other[index] + (spacing if target_lane == 'lane2' else -spacing)
                chosen[index] = x if 0 <= x < 640 else None
                estimated = True
        valid = [i for i, x in enumerate(chosen) if x is not None]
        if len(valid) >= 2:
            self.last_slope = (float(np.polyfit([TARGET_HEIGHTS[i] for i in valid],
                                                [chosen[i] for i in valid], 1)[0]), now)
        if covered and valid and len(valid) < 3:
            chosen = self.extend_straight(chosen, valid, now)
            if [i for i, x in enumerate(chosen) if x is not None] != valid:
                estimated = True
            valid = [i for i, x in enumerate(chosen) if x is not None]
        if len(valid) < 2:
            filled = far_extend(chosen, far.get(target_lane, [None] * 3))
            if filled is not None:
                chosen, estimated = filled, True
                valid = [i for i, x in enumerate(chosen) if x is not None]
        roi = rois.get(target_lane, np.zeros((180, 640), dtype=np.uint8))
        if crosswalk_roi is not None:
            roi = np.maximum(roi, crosswalk_roi // 2)  # preview: crosswalk shown in grey
        if len(valid) < 2:
            return None, 'lost', masks, roi
        for index, x in enumerate(chosen):
            if x is None:
                chosen[index] = float(np.interp(TARGET_HEIGHTS[index],
                                                [TARGET_HEIGHTS[i] for i in valid],
                                                [chosen[i] for i in valid]))
                estimated = True
        return chosen, 'estimated' if estimated else 'direct', masks, roi


def far_extend(near, far_centers):
    """Fill missing near centers from a straight line through every visible center (near rows
    and far rows), if at least two far rows see the lane. Returns None when that is impossible."""
    points = [(ROI_TOP + TARGET_HEIGHTS[i], x) for i, x in enumerate(near) if x is not None]
    far_points = [(y, x) for y, x in zip(FAR_HEIGHTS, far_centers) if x is not None]
    if len(far_points) < 2:
        return None
    ys, xs = zip(*(points + far_points))
    slope, intercept = np.polyfit(ys, xs, 1)
    filled = list(near)
    for index, y in enumerate(TARGET_HEIGHTS):
        if filled[index] is None:
            x = float(slope * (ROI_TOP + y) + intercept)
            filled[index] = x if 0 <= x < 640 else None
    return filled if sum(x is not None for x in filled) >= 2 else None


