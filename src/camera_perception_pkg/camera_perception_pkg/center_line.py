"""Which lane a detected car is in, from the white dashed centre line painted between the lanes.

best.pt's lane1/lane2 masks cannot separate the two lanes reliably (lane2 often covers the whole
road), so a car in the other lane was taken for one in ours. The centre line is found directly:
white paint with dark road on both sides (edge lines have mat or floor on one side). Only the
dashes right beside the bottom of the car's box are used, so curves do not matter. lane1 is the
left lane. Used only to check that a car is really on our side before changing lanes.
"""

import cv2
import numpy as np


def center_points(bgr, y_top=200, y_bot=470, step=4):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    white = (v > 165) & (s < 60)
    road = (v < 120) & (s < 90)
    width = bgr.shape[1]
    points = []
    for y in range(y_top, y_bot, step):
        scale = (y - 160) / 310.0                      # markings grow towards the bottom
        widest, gap, reach = 4 + 34 * scale, int(4 + 8 * scale), int(8 + 30 * scale)
        xs = np.flatnonzero(white[y])
        if not len(xs):
            continue
        for run in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1):
            if not 2 <= run[-1] - run[0] + 1 <= widest:
                continue
            a, b, c, d = run[0] - gap - reach, run[0] - gap, run[-1] + gap, run[-1] + gap + reach
            if a < 0 or d >= width:
                continue
            if road[y, a:b].mean() > 0.7 and road[y, c:d].mean() > 0.7:
                points.append(((run[0] + run[-1]) / 2.0, y))
    return points


def ransac_lines(points, tries=200, tol=8.0, min_points=4, min_span=15, max_lines=2, seed=0):
    """Up to max_lines straight lines x = a*y + b (not horizontal markings): [(a, b, inlier rows)]."""
    rng = np.random.default_rng(seed)
    p = np.array(points, float)
    lines = []
    while len(p) >= min_points and len(lines) < max_lines:
        best = None
        for _ in range(tries):
            i, j = rng.choice(len(p), 2, replace=False)
            if abs(p[i, 1] - p[j, 1]) < 10:
                continue
            a = (p[j, 0] - p[i, 0]) / (p[j, 1] - p[i, 1])
            inliers = np.abs(a * p[:, 1] + p[i, 0] - a * p[i, 1] - p[:, 0]) < tol
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or best.sum() < min_points or np.ptp(p[best, 1]) < min_span:
            break
        a, b = np.polyfit(p[best, 1], p[best, 0], 1)
        if abs(a) < 2.5:
            lines.append((float(a), float(b), np.sort(p[best, 1])))
        p = p[~best]
    return lines


def car_side(bgr, box, above=15, below=110, on_line=0.2, beside=1.5):
    """'lane1', 'lane2', 'line' (sits on the centre line) or None when no centre-line paint is right
    beside it (within `beside` car widths: a parking-box line far from a car parked at the image
    edge was taken for it on 11-18 and started a lane change)."""
    x0, _, x1, y1 = box
    ya, yb = int(max(250, y1 - above)), int(min(472, y1 + below))
    if yb - ya < 30:
        return None
    points = [(x, y) for x, y in center_points(bgr, ya, yb, 2) if not (x0 - 5 <= x <= x1 + 5 and y <= y1)]
    lines = ransac_lines(points)
    if not lines:
        return None
    centre = (x0 + x1) / 2
    width = max(x1 - x0, 1.0)
    lines = [l for l in lines if abs(l[0] * y1 + l[1] - centre) <= beside * width]
    if not lines:
        return None
    a, b, _ = min(lines, key=lambda line: abs(line[0] * y1 + line[1] - centre))
    line_x = a * y1 + b
    if abs(centre - line_x) < on_line * (x1 - x0):
        return 'line'
    return 'lane1' if centre < line_x else 'lane2'
