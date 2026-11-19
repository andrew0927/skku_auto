import math

from lidar_perception_pkg.lidar_obstacle_detector_node import corridor_points


def scan(*objects):
    """360 one-degree ranges, 0 deg ahead, angles growing to the left; objects = (deg, metres)."""
    ranges = [float('inf')] * 360
    for degree, distance in objects:
        ranges[degree % 360] = distance
    return ranges


def points(ranges):
    return corridor_points(ranges, 0.0, 2 * math.pi / 360, 0.15, 1.5, 0.3)


def test_obstacle_straight_ahead_is_in_the_corridor():
    found = points(scan((0, 1.0), (1, 1.0), (359, 1.0)))
    assert len(found) == 3 and all(abs(y) < 0.05 for _, y in found)


def test_car_in_the_next_lane_is_ignored():
    # About 0.55-0.6 m to the side, 1-1.4 m ahead: what the old +-15 deg sector still caught.
    assert points(scan((25, 1.4), (30, 1.1), (335, 1.3))) == []


def test_offset_sign_left_is_positive():
    (x, y), = points(scan((10, 1.0)))
    assert y > 0.15 and 0.9 < x < 1.0
