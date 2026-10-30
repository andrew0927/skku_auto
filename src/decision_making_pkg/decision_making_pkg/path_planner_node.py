"""Interpolate valid lane centers and carry the original camera timestamp."""

import numpy as np
from scipy.interpolate import CubicSpline
import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import LaneInfo, PathPlanningResult


class PathPlannerNode(Node):
    def __init__(self, **kwargs):
        super().__init__('path_planner_node', **kwargs)
        self.car_center_point = self.declare_parameter('car_center_point', [320.0, 179.0]).value
        self.lane_sub = self.create_subscription(LaneInfo, 'yolov8_lane_info', self.lane_callback, 1)
        self.publisher = self.create_publisher(PathPlanningResult, 'path_planning_result', 1)

    def lane_callback(self, msg):
        result = PathPlanningResult()
        result.header = msg.header
        result.tracking_source = msg.tracking_source
        points = [(float(p.target_y), float(p.target_x)) for p in msg.target_points]
        if len(points) >= 2 and all(np.isfinite(v) for point in points for v in point):
            # The vehicle anchor takes precedence if a detection shares its y coordinate.
            heights = dict(points)
            heights[float(self.car_center_point[1])] = float(self.car_center_point[0])
            ys = np.array(sorted(heights))
            xs = np.array([heights[y] for y in ys])
            if len(ys) >= 3:
                spline = CubicSpline(ys, xs, bc_type='natural')
                y_new = np.linspace(ys[0], ys[-1], 100)
                x_new = spline(y_new)
                if np.isfinite(x_new).all():
                    result.x_points = x_new.tolist()
                    result.y_points = y_new.tolist()
        # Empty paths explicitly invalidate the previous path on detection loss.
        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
