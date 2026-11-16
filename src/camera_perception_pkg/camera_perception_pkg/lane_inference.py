"""Recover color-sensitive lane masks under a geometrically verified green lamp."""

import math
import time

import cv2

from .signal_geometry import find_signal_lamp


def merge_detection_groups(groups, prefer_recovered_lanes=False):
    if not groups:
        return []
    combined = list(groups[0])
    if prefer_recovered_lanes:
        recovered = {d.class_name.casefold() for group in groups[1:] for d in group
                     if d.class_name.casefold() in ('lane1', 'lane2')}
        combined = [d for d in combined if d.class_name.casefold() not in recovered]
    present = {d.class_name.casefold() for d in combined}
    for group in groups[1:]:
        allowed = {'lane1', 'lane2', 'crosswalk'} - present
        combined.extend(d for d in group if d.class_name.casefold() in allowed)
        present.update(d.class_name.casefold() for d in group if d.class_name.casefold() in allowed)
    return combined


class GreenLaneInference:
    def __init__(self, model, threshold=0.5, device='cpu', enabled=True,
                 target_lane='lane2', green_hold=1.0, probe_interval=0.4,
                 color_confirmations=3):
        if (target_lane not in ('lane1', 'lane2')
                or not all(math.isfinite(v) and v > 0 for v in (green_hold, probe_interval))
                or not 0.1 <= green_hold <= 3.0 or not 0.1 <= probe_interval <= 2.0
                or not 2 <= color_confirmations <= 10):
            raise ValueError('Invalid green lane recovery parameters')
        self.model, self.threshold, self.device, self.enabled = model, threshold, device, enabled
        self.target_lane = target_lane
        self.green_hold, self.probe_interval = green_hold, probe_interval
        self.color_confirmations = color_confirmations
        self.reset()

    def reset(self):
        self.prefer_gray = False
        self.green_until = self.retry_after = -math.inf
        self.next_probe = -math.inf
        self.color_streak = 0
        self.last_probe_stamp = 0.0
        self.last_predict_time = None
        self.previous_gray_target = False

    def set_target_lane(self, lane):
        if lane not in ('lane1', 'lane2'):
            return
        if lane != self.target_lane:
            self.target_lane = lane
            self.color_streak = 0
            self.last_probe_stamp = 0.0
            self.next_probe = self.retry_after = -math.inf

    def lane_names(self, result):
        return {self.model.names[int(box.cls)].casefold()
                for box in (result.boxes if result.boxes is not None else [])
                if self.model.names[int(box.cls)].casefold() in ('lane1', 'lane2')
                and float(box.conf) >= self.threshold}

    def lane_count(self, result):
        return len(self.lane_names(result))

    def predict_image(self, image):
        return self.model.predict(source=image, verbose=False, stream=False,
                                  conf=self.threshold, device=self.device)[0].cpu()

    def _color_probe(self, result, stamp):
        if stamp is not None:
            if not math.isfinite(stamp) or stamp <= self.last_probe_stamp:
                return
            self.last_probe_stamp = stamp
        self.color_streak = (self.color_streak + 1
                             if self.target_lane in self.lane_names(result) else 0)

    def predict(self, image, now=None, signal_color=None, frame_stamp=None):
        now = time.monotonic() if now is None else now
        if (self.last_predict_time is not None
                and (now < self.last_predict_time or now - self.last_predict_time > self.green_hold)):
            self.reset()
        self.last_predict_time = now
        lamp = find_signal_lamp(image) if self.enabled else None
        raw_color = lamp['color'].casefold() if lamp is not None else None
        hint = signal_color.casefold() if signal_color else None
        if raw_color in ('red', 'yellow') or hint in ('red', 'yellow') or not self.enabled:
            self.reset()
            self.last_predict_time = now
        elif raw_color == 'green' or hint == 'green':
            self.green_until = now + self.green_hold
        green_context = self.enabled and now < self.green_until

        def metadata(mode, passes, results, prefer=False):
            names = set().union(*(self.lane_names(r) for r in results))
            return dict(mode=mode, forward_passes=passes, green_lamp=raw_color == 'green',
                        green_context=green_context, target_lane=self.target_lane,
                        target_found=self.target_lane in names, output_lanes=len(names),
                        color_confirmations=self.color_streak, prefer_recovered_lanes=prefer)

        def grayscale():
            return self.predict_image(cv2.cvtColor(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))

        if self.prefer_gray:
            recovered = grayscale()
            target_found = self.target_lane in self.lane_names(recovered)
            probe = (now >= self.next_probe or
                     (not target_found and (self.previous_gray_target or self.color_streak > 0)))
            self.previous_gray_target = target_found
            if not probe:
                return [recovered], metadata('green_grayscale_cached', 1, [recovered])
            primary = self.predict_image(image)
            self.next_probe = now + self.probe_interval
            self._color_probe(primary, frame_stamp)
            if self.color_streak >= self.color_confirmations and (not green_context or not target_found):
                self.prefer_gray = False
                return [primary], metadata('color', 2, [primary])
            return [primary, recovered], metadata('green_grayscale_probe', 2,
                                                   [primary, recovered], prefer=target_found)

        primary = self.predict_image(image)
        if (self.target_lane in self.lane_names(primary) or not green_context
                or now < self.retry_after):
            return [primary], metadata('color', 1, [primary])
        recovered = grayscale()
        self.prefer_gray = self.target_lane in self.lane_names(recovered)
        self.previous_gray_target = self.prefer_gray
        self.next_probe = self.retry_after = now + self.probe_interval
        self.color_streak = 0
        groups = [primary, recovered] if self.lane_count(recovered) else [primary]
        return groups, metadata('green_grayscale' if self.prefer_gray else 'green_grayscale_failed', 2, groups)
