"""Conservative lamp fallback for a horizontal signal missed by YOLO."""

import math

import cv2
import numpy as np


HSV_RANGES = {
    'red1': (np.array([0, 100, 95]), np.array([10, 255, 255])),
    'red2': (np.array([160, 100, 95]), np.array([179, 255, 255])),
    'yellow': (np.array([11, 100, 95]), np.array([35, 255, 255])),
    'green': (np.array([40, 100, 95]), np.array([90, 255, 255])),
}


def find_signal_lamp(image, max_height_ratio=0.45):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    limit = int(height * max_height_ratio)
    upper = hsv[:limit]
    masks = {
        'Red': cv2.inRange(upper, *HSV_RANGES['red1']) | cv2.inRange(upper, *HSV_RANGES['red2']),
        'Yellow': cv2.inRange(upper, *HSV_RANGES['yellow']),
        'Green': cv2.inRange(upper, *HSV_RANGES['green']),
    }
    white = cv2.inRange(upper, np.array([0, 0, 235]), np.array([179, 90, 255]))
    candidates = []
    for mask in [white, *masks.values()]:
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            if (area < max(25, width * height * 0.00008) or area > width * height * 0.03
                    or not 0.65 <= w / h <= 1.5 or perimeter <= 0
                    or 4 * math.pi * area / perimeter ** 2 < 0.82):
                continue
            cx, cy, diameter = x + w // 2, y + h // 2, max(w, h)
            radius = max(3, int(diameter * 0.8))
            left, right = max(0, cx - radius), min(width, cx + radius + 1)
            top, bottom = max(0, cy - radius), min(limit, cy + radius + 1)
            yy, xx = np.ogrid[top:bottom, left:right]
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            counts = {name: int(np.count_nonzero(color[top:bottom, left:right][disk]))
                      for name, color in masks.items()}
            color = max(counts, key=counts.get)
            if counts[color] < max(20, int(np.count_nonzero(disk) * 0.04)):
                continue
            # A colored reflection alone is not enough: require an adjacent dark housing.
            dark_context = []
            for a, b in ((cx - 3 * diameter, cx - int(0.65 * diameter)),
                         (cx + int(0.65 * diameter), cx + 3 * diameter)):
                a, b = max(0, a), min(width, b)
                t, end = max(0, cy - diameter // 2), min(limit, cy + diameter // 2)
                fraction = float(np.mean(hsv[t:end, a:b, 2] < 110)) if b > a and end > t else 0.0
                dark_context.append(fraction)
            if max(dark_context) < 0.45:
                continue
            candidates.append(dict(color=color, box=[left, top, right, bottom],
                                   pixels=counts[color], source='lamp_geometry'))
    priority = {'Red': 3, 'Yellow': 2, 'Green': 1}
    return max(candidates, key=lambda item: (priority[item['color']], item['pixels'])) if candidates else None


class ColorConfirmation:
    def __init__(self, frames=2, green_frames=3):
        if frames < 1 or green_frames < 1:
            raise ValueError('Color confirmation counts must be positive')
        self.frames, self.green_frames = frames, green_frames
        self.color, self.count = None, 0

    def update(self, color):
        self.count = self.count + 1 if color == self.color else 1
        self.color = color
        required = self.green_frames if color == 'Green' else self.frames
        return color if color in ('None', 'Unknown') or self.count >= required else 'Unknown'
