"""Read LaserScan sectors using angle metadata, including sectors across zero."""

import math


def stamp_seconds(scan):
    return scan.header.stamp.sec + scan.header.stamp.nanosec / 1e9


def scan_is_fresh(scan, ros_now, timeout=1.0):
    stamp = stamp_seconds(scan)
    return stamp > 0 and -0.05 <= ros_now - stamp <= timeout


def scan_is_valid(scan):
    return (bool(scan.ranges) and math.isfinite(scan.angle_min)
            and math.isfinite(scan.angle_increment) and scan.angle_increment != 0
            and any(math.isfinite(d) and max(0.05, scan.range_min) <= d <= scan.range_max
                    for d in scan.ranges))


def sector_distance(scan, start_degrees, end_degrees):
    span = (end_degrees - start_degrees) % 360
    values = []
    clear = False
    for index, distance in enumerate(scan.ranges):
        angle = math.degrees(scan.angle_min + index * scan.angle_increment) % 360
        if (angle - start_degrees) % 360 > span + 1e-4:
            continue
        if math.isfinite(distance) and max(0.05, scan.range_min) <= distance <= scan.range_max:
            values.append(distance)
        elif distance == math.inf:
            clear = True
    if values:
        return min(values)
    return math.inf if clear else None
