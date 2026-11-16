"""Image-position stopping calibrated against a crosswalk reference image."""

import math


def crosswalk_edge_ratio(detections, edge='far_edge'):
    if edge not in ('near_edge', 'far_edge'):
        raise ValueError('Crosswalk edge must be near_edge or far_edge')
    candidates = []
    for detection in detections:
        if detection.class_name.casefold() != 'crosswalk':
            continue
        box = detection.bbox
        x, y, w, h = (box.center.position.x, box.center.position.y, box.size.x, box.size.y)
        width, height = detection.mask.width or 640, detection.mask.height or 480
        if (not all(math.isfinite(v) for v in (x, y, w, h))
                or not 0 < width <= 4096 or not 0 < height <= 4096
                or w < width * 0.1 or h <= 0):
            continue
        if x + w / 2 <= 0 or x - w / 2 >= width or y + h / 2 <= 0 or y - h / 2 >= height:
            continue
        reference_y = y + h / 2 if edge == 'near_edge' else y - h / 2
        candidates.append(max(0.0, min(1.0, reference_y / height)))
    return max(candidates) if candidates else None


def crosswalk_bottom_ratio(detections):
    return crosswalk_edge_ratio(detections, 'near_edge')


class RedLightStopController:
    def __init__(self, stop_ratio=325.0 / 480.0, approach_timeout=3.0):
        if not 0.2 <= stop_ratio <= 0.95 or not 0 < approach_timeout <= 5:
            raise ValueError('Invalid crosswalk stop ratio or approach timeout')
        self.stop_ratio, self.approach_timeout = stop_ratio, approach_timeout
        self.red = self.stopped = False
        self.approach_started = None
        self.reason = 'no red light'

    def observe_light(self, color):
        if color.casefold() == 'red':
            self.red = True
        elif color.casefold() == 'green':
            self.red = self.stopped = False
            self.approach_started = None
            self.reason = 'green light'

    def stop(self, reason):
        self.stopped, self.reason = True, reason
        return 'stop'

    def evaluate(self, now, position, fresh, can_approach):
        if not self.red:
            return 'clear'
        if self.stopped:
            self.reason = 'waiting for green'
            return 'stop'
        if not fresh or position is None or not math.isfinite(position):
            return self.stop('crosswalk position missing or stale')
        if position >= self.stop_ratio:
            return self.stop('crosswalk reached stop line')
        if not can_approach:
            return self.stop('no valid moving lane command for approach')
        if self.approach_started is None:
            self.approach_started = now
        if now - self.approach_started >= self.approach_timeout:
            return self.stop('crosswalk approach timeout')
        self.reason = 'approaching crosswalk before stop line'
        return 'approach'
