"""Report a traffic light change from pixel differences while the car is stopped.

The camera is still while the car waits at red, so the light region can be compared with a
reference taken shortly after the stop. A lamp switching on is a compact, lamp-sized blob of
pixels that became clearly brighter. Such a blob that is not red means the light has changed,
even when the colour check cannot name the new colour (washed-out or bluish green LEDs).
"""

import cv2
import numpy as np


# Lit-lamp colours measured on the track's light (C920, 2026-11-08): while red, 18-19 % of the
# light region is bright saturated red and 0 % green; while green, 11-17 % green and 0 % red.
LAMP_MIN_FRACTION = 0.03
BOX_MEMORY = 5.0  # s: a light box seen this long before the stop still locates the lamps


def lamp_fractions(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = value > 180
    red = bright & (sat > 100) & ((hue <= 12) | (hue >= 165))
    green = bright & (sat > 60) & (hue >= 40) & (hue <= 100)
    return float(red.mean()), float(green.mean())


class LampChangeDetector:
    """While stopped at red, YOLO often loses the (large, close) light: on the recorded lap it
    found it in only ~35 % of the stopped frames. The camera is still, so the lamps are read
    directly from the pixels where the light was last seen."""

    def __init__(self, settle=0.7, margin=0.5, diff=90, brighter=50,
                 min_fill=0.3, min_fraction=0.02, min_pixels=30):
        self.settle = settle            # seconds after the stop before the reference is taken
        self.margin = margin            # box enlargement so a jittering box still contains the lamps
        self.diff = diff                # summed BGR difference that counts as a changed pixel
        self.brighter = brighter        # brightness gain that counts as a lamp switching on
        self.min_fill = min_fill        # blob area / its bounding box: lamps are compact
        self.min_fraction = min_fraction  # blob area relative to the region
        self.min_pixels = min_pixels
        self.stop_time = None
        self.last_box = None
        self.reference = None
        self.last_blob = 0
        self.seen_box = None
        self.seen_time = None
        self.lamp_region = None
        self.last_read = None

    def reset(self):
        self.stop_time = None
        self.last_box = None
        self.reference = None
        self.lamp_region = None

    def update(self, bgr, box, color, stopped, now):
        """box = (x0, y0, x1, y1) of the detected light or None; returns the colour to report."""
        if box is not None:
            self.seen_box, self.seen_time = box, now
        if not stopped:
            self.reset()
            return color
        if self.stop_time is None:
            self.stop_time = now
            if self.seen_box is not None and now - self.seen_time <= BOX_MEMORY:
                self.lamp_region = self.expand(bgr, self.seen_box, 0.15, 20)
        if box is not None:
            self.last_box = box
        if self.lamp_region is not None:
            x0, y0, x1, y1 = self.lamp_region
            red, green = lamp_fractions(bgr[y0:y1, x0:x1])
            self.last_read = (red, green)
            if red >= LAMP_MIN_FRACTION and red > green:
                return 'Red'
            if green >= LAMP_MIN_FRACTION and green > red:
                return 'Green'
        if self.reference is None:
            if self.last_box is not None and now - self.stop_time >= self.settle:
                self.capture(bgr, self.last_box)
            return color
        if color == 'Red':
            return color
        x0, y0, x1, y1 = self.reference['box']
        region = bgr[y0:y1, x0:x1].astype(np.int16)
        reference = self.reference['bgr']
        changed = np.abs(region - reference).sum(axis=2) > self.diff
        if changed.mean() > 0.5:
            # Almost everything changed: the camera moved or exposure jumped. Start over.
            self.capture(bgr, self.reference['raw_box'])
            return color
        lit = changed & ((region.max(axis=2) - reference.max(axis=2)) > self.brighter)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(lit.astype(np.uint8), connectivity=8)
        if count < 2:
            return color
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[largest, cv2.CC_STAT_AREA])
        fill = area / float(stats[largest, cv2.CC_STAT_WIDTH] * stats[largest, cv2.CC_STAT_HEIGHT])
        if area < max(self.min_pixels, self.min_fraction * lit.size) or fill < self.min_fill:
            return color
        blob = labels == largest
        hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hue, sat = hsv[..., 0][blob], hsv[..., 1][blob]
        # White (washed-out) pixels have hue 0, so only saturated pixels can count as red.
        reddish = (sat > 60) & ((hue <= 15) | (hue >= 160))
        if reddish.mean() >= 0.3:
            return color
        self.last_blob = area
        return 'Green'

    @staticmethod
    def expand(bgr, box, fraction, pixels):
        height, width = bgr.shape[:2]
        x0, y0, x1, y1 = box
        dx, dy = (x1 - x0) * fraction + pixels, (y1 - y0) * fraction + pixels
        x0, x1 = max(0, int(x0 - dx)), min(width, int(x1 + dx))
        y0, y1 = max(0, int(y0 - dy)), min(height, int(y1 + dy))
        return (x0, y0, x1, y1) if x1 - x0 >= 2 and y1 - y0 >= 2 else None

    def capture(self, bgr, box):
        region = self.expand(bgr, box, self.margin, 0)
        if region is None:
            return
        x0, y0, x1, y1 = region
        self.reference = {'raw_box': box, 'box': (x0, y0, x1, y1),
                          'bgr': bgr[y0:y1, x0:x1].astype(np.int16)}
