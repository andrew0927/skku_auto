from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from camera_perception_pkg.lane_geometry import LaneCenterTracker, region_centers


def lane(name, x0, x1):
    src = np.float32([[238, 316], [402, 313], [501, 476], [155, 476]])
    dst = np.float32([[192, 0], [448, 0], [448, 480], [192, 480]])
    inverse = cv2.getPerspectiveTransform(dst, src)
    polygon = np.float32([[[x0, 0], [x1, 0], [x1, 479], [x0, 479]]])
    points = cv2.perspectiveTransform(polygon, inverse)[0]
    return NS(class_name=name, mask=NS(width=640, height=480,
                                      data=[NS(x=float(x), y=float(y)) for x, y in points]))


def test_no_lane_does_not_fabricate_a_center():
    assert region_centers(np.zeros((180, 640), dtype=np.uint8)) == [None] * 3
    centers, source, _, _ = LaneCenterTracker().update([], 'lane2', 0)
    assert centers is None and source == 'lost'


def test_fallback_uses_recent_measured_spacing_and_expires():
    tracker = LaneCenterTracker(pair_timeout=2.0)
    one, two = lane('lane1', 40, 180), lane('lane2', 220, 360)
    direct, source, _, _ = tracker.update([one, two], 'lane2', 0)
    assert source == 'direct'
    assert direct == pytest.approx([290] * 3, abs=4)
    estimated, source, _, _ = tracker.update([one], 'lane2', 0.1)
    assert source == 'estimated'
    assert estimated == pytest.approx(direct, abs=2)
    missing, source, _, _ = tracker.update([one], 'lane2', 2.1)
    assert missing is None and source == 'lost'


def test_fallback_needs_an_observed_pair():
    result, source, _, _ = LaneCenterTracker().update([lane('lane1', 40, 180)], 'lane2', 0)
    assert result is None and source == 'lost'
