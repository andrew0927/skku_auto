from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from camera_perception_pkg.lane_geometry import LaneCenterTracker, region_centers


SRC = np.float32([[238, 316], [402, 313], [501, 476], [155, 476]])
DST = np.float32([[192, 0], [448, 0], [448, 480], [192, 480]])
INVERSE = cv2.getPerspectiveTransform(DST, SRC)


def lane_polygon(name, bird_points):
    points = cv2.perspectiveTransform(np.float32([bird_points]), INVERSE)[0]
    return NS(class_name=name, mask=NS(width=640, height=480,
                                      data=[NS(x=float(x), y=float(y)) for x, y in points]))


def lane(name, x0, x1):
    return lane_polygon(name, [[x0, 0], [x1, 0], [x1, 479], [x0, 479]])


def crosswalk(bird_y0, bird_y1):
    """A crosswalk box covering bird's-eye rows bird_y0..bird_y1 across the road."""
    quad = np.float32([[[192, bird_y0], [448, bird_y0], [448, bird_y1], [192, bird_y1]]])
    ys = cv2.perspectiveTransform(quad, INVERSE)[0][:, 1]
    return NS(class_name='crosswalk', mask=NS(width=640, height=480, data=[]),
              bbox=NS(center=NS(position=NS(x=320.0, y=float((ys.min() + ys.max()) / 2))),
                      size=NS(x=640.0, y=float(ys.max() - ys.min()))))


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


def test_crosswalk_heights_are_ignored_and_extended_straight():
    # lane2 is straight at x 220..360 but a wide false blob appears beside it in the crosswalk.
    straight = lane('lane2', 220, 360)
    # The ROI is bird's-eye rows 300..480; this blob sits beside the lane at the far height.
    blob = lane_polygon('lane2', [[362, 300], [600, 300], [600, 345], [362, 345]])
    far_crosswalk = crosswalk(300, 345)
    wrong, source, _, _ = LaneCenterTracker().update([straight, blob], 'lane2', 0)
    assert source == 'direct' and wrong[0] > 400
    fixed, source, masks, _ = LaneCenterTracker().update([straight, blob, far_crosswalk], 'lane2', 0)
    assert source == 'estimated' and 'crosswalk' in masks
    assert fixed == pytest.approx([290] * 3, abs=4)


def test_one_visible_height_extends_with_the_remembered_slope():
    tracker = LaneCenterTracker()
    slanted = lane_polygon('lane2', [[180, 0], [320, 0], [400, 479], [260, 479]])
    before, source, _, _ = tracker.update([slanted], 'lane2', 0)
    assert source == 'direct'
    near_crosswalk = crosswalk(340, 480)
    after, source, _, _ = tracker.update([slanted, near_crosswalk], 'lane2', 0.1)
    assert source == 'estimated'
    assert after == pytest.approx(before, abs=3)
    # Without a remembered slope the lane beyond the crosswalk still gives the line.
    fresh, source, _, _ = LaneCenterTracker().update([slanted, near_crosswalk], 'lane2', 0)
    assert source == 'estimated' and fresh == pytest.approx(before, abs=8)
    # Near rows all hidden: the far lane is extended over the crosswalk.
    over, source, _, _ = tracker.update([slanted, crosswalk(300, 480)], 'lane2', 0.2)
    assert source == 'estimated' and over == pytest.approx(before, abs=8)
    # Nothing visible anywhere: lost, so the planner's crossing logic takes over.
    none, source, _, _ = LaneCenterTracker().update([crosswalk(300, 480)], 'lane2', 0.3)
    assert none is None and source == 'lost'


def test_far_lane_fills_near_rows_when_near_lane_is_hidden():
    from camera_perception_pkg.lane_geometry import FAR_HEIGHTS, ROI_TOP, far_extend
    # A straight lane at x = 300 seen only far away.
    assert far_extend([None, None, None], [300.0] * len(FAR_HEIGHTS)) == pytest.approx([300.0] * 3)
    # One far point is not enough to draw a line.
    assert far_extend([None, None, None], [300.0] + [None] * (len(FAR_HEIGHTS) - 1)) is None
    # Near-only lane from the tracker: hidden near rows get the far line.
    far_only = lane_polygon('lane2', [[220, 0], [360, 0], [360, ROI_TOP - 10], [220, ROI_TOP - 10]])
    centers, source, _, _ = LaneCenterTracker().update([far_only], 'lane2', 0)
    assert source == 'estimated' and centers == pytest.approx([290] * 3, abs=6)


def test_switching_to_lane1_places_it_one_spacing_left_of_lane2():
    tracker = LaneCenterTracker(pair_timeout=2.0)
    one, two = lane('lane1', 40, 180), lane('lane2', 220, 360)
    tracker.update([one, two], 'lane2', 0)                         # spacing measured here
    seen_lane1, _, _, _ = LaneCenterTracker().update([one], 'lane1', 0)
    # 10 s later an obstacle: switch to lane1, but only lane2 is in view.
    tracker.switch_target(10.0)
    est, source, _, _ = tracker.update([two], 'lane1', 10.1)
    assert source == 'estimated'
    assert est == pytest.approx(seen_lane1, abs=3)                 # where lane1 really was
    # Without the switch, a 10 s old spacing is not trusted.
    stale, source, _, _ = LaneCenterTracker().update([two], 'lane1', 10.1)
    assert stale is None and source == 'lost'


def test_switch_without_any_measured_spacing_uses_the_lane_width():
    tracker = LaneCenterTracker()
    tracker.switch_target(0.0)
    est, source, _, _ = tracker.update([lane('lane2', 220, 360)], 'lane1', 0.1)
    assert source == 'estimated' and est == pytest.approx([290 - 256] * 3, abs=6)
