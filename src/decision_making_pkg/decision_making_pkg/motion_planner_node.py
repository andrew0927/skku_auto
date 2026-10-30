"""Lane following with reduced speed and bounded steering hold on brief lane loss."""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from interfaces_pkg.msg import MotionCommand, PathPlanningResult
from std_msgs.msg import Bool, String

from .control_policy import MAX_DRIVING_SPEED, clamp, path_steering


class MotionPlanningNode(Node):
    def __init__(self, **kwargs):
        super().__init__('motion_planner_node', **kwargs)
        self.driving_speed = int(clamp(self.declare_parameter('driving_speed', 255).value,
                                       0, MAX_DRIVING_SPEED))
        self.corner_speed = int(clamp(self.declare_parameter('corner_speed', self.driving_speed).value,
                                      0, self.driving_speed))
        self.estimated_speed = int(clamp(self.declare_parameter('estimated_speed', 200).value,
                                         0, self.driving_speed))
        self.lane_loss_steering = int(clamp(self.declare_parameter('lane_loss_steering', 4).value,
                                            0, 7))
        self.lane_loss_hold = self.declare_parameter('lane_loss_hold', 0.4).value
        self.reacquire_hold = self.declare_parameter('reacquire_hold', 0.5).value
        self.path_timeout = self.declare_parameter('path_timeout', 0.6).value
        self.source_timeout = self.declare_parameter('source_timeout', 1.0).value
        self.lookahead = self.declare_parameter('lookahead', 100.0).value
        self.gain = self.declare_parameter('steering_gain', 0.12).value
        self.offset = self.declare_parameter('steering_offset', 0.0).value
        self.sign = self.declare_parameter('steering_sign', 1.0).value
        self.alpha = self.declare_parameter('steering_alpha', 0.4).value
        self.rate_limit = self.declare_parameter('steering_rate_limit', 10.0).value
        self.car_center = self.declare_parameter('car_center_point', [320.0, 179.0]).value
        self.obey_lights = self.declare_parameter('obey_traffic_lights', False).value
        if (self.path_timeout <= 0 or self.source_timeout <= 0 or self.lookahead <= 0
                or not 0 < self.alpha <= 1 or self.rate_limit <= 0
                or self.sign not in (-1.0, 1.0) or len(self.car_center) != 2
                or not 0 <= self.lane_loss_hold <= 1.0 or self.reacquire_hold < 0):
            raise ValueError('Invalid controller parameters')
        self.path = None
        self.empty_path = False
        self.last_path_time = None
        self.source_stamp = 0.0
        self.lane_source = 'lost'
        self.auto_active = False
        self.light = None
        self.last_light_time = None
        self.previous_steering = 0.0
        self.last_tick = time.monotonic()
        self.state = None
        self.clear_recovery()
        self.publisher = self.create_publisher(MotionCommand, 'autonomous_control_signal', 1)
        self.state_pub = self.create_publisher(String, 'control_state', 1)
        self.path_sub = self.create_subscription(PathPlanningResult, 'path_planning_result',
                                                  self.path_callback, 1)
        self.reset_sub = self.create_subscription(Bool, 'driving_mode_reset', self.reset_callback, 1)
        self.light_sub = self.create_subscription(String, 'yolov8_traffic_light_info',
                                                   self.light_callback, 1)
        mode_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_sub = self.create_subscription(String, 'driving_mode', self.mode_callback, mode_qos)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def clear_recovery(self, keep_slow=False):
        self.last_follow_time = None
        self.last_follow_stamp = 0.0
        if not keep_slow:
            self.slow_until = 0.0
            self.reacquire_pending = False

    def path_callback(self, msg):
        points = list(zip(msg.x_points, msg.y_points))
        self.empty_path = not msg.x_points and not msg.y_points
        self.path = points if (len(msg.x_points) == len(msg.y_points) and len(points) >= 2
                               and all(math.isfinite(v) for p in points for v in p)) else None
        self.last_path_time = time.monotonic()
        self.source_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        # Source and path belong to the same camera frame; independent topics can race.
        self.lane_source = msg.tracking_source

    def mode_callback(self, msg):
        active = msg.data == 'autonomous_active'
        if not active or not self.auto_active:
            self.clear_recovery()
        self.auto_active = active

    def reset_callback(self, msg):
        if msg.data:
            self.previous_steering = 0.0
            self.last_tick = time.monotonic()
            self.clear_recovery()

    def light_callback(self, msg):
        self.light = msg.data.casefold()
        self.last_light_time = time.monotonic()

    def can_recover_lane_loss(self, now, ros_now):
        return (self.auto_active and self.lane_loss_hold > 0
                and self.last_follow_time is not None
                and 0 <= now - self.last_follow_time <= self.lane_loss_hold
                and -0.05 <= ros_now - self.last_follow_stamp <= self.source_timeout)

    def timer_callback(self):
        now = time.monotonic()
        dt = clamp(now - self.last_tick, 0.0, 0.2)
        self.last_tick = now
        ros_now = self.get_clock().now().nanoseconds / 1e9
        source_age = ros_now - self.source_stamp
        state = 'lane following'
        command = MotionCommand()
        recovering = False
        if self.last_path_time is None:
            state = 'path missing or invalid'
        elif now - self.last_path_time > self.path_timeout:
            state = 'path receipt stale'
        elif self.source_stamp <= 0 or not -0.05 <= source_age <= self.source_timeout:
            state = 'path source frame stale or invalid'
        elif self.obey_lights and (self.last_light_time is None
                                  or now - self.last_light_time > self.source_timeout):
            state = 'traffic light input stale or missing'
        elif self.obey_lights and self.light == 'red':
            state = 'red traffic light'
        elif self.path is None:
            if self.empty_path and self.lane_source == 'lost':
                # Both-lane loss has no speed reduction or post-loss speed hold.
                self.slow_until = 0.0
                self.reacquire_pending = False
            if self.empty_path and self.lane_source == 'lost' and self.can_recover_lane_loss(now, ros_now):
                state = 'lane lost (right steering at cruise speed)'
                command.steering = self.lane_loss_steering
                # Resume the normal steering filter from the command actually sent.
                self.previous_steering = float(command.steering)
                command.left_speed = command.right_speed = self.driving_speed
                recovering = True
                # Do not refresh last_follow_time: repeated missing frames must time out.
            else:
                state = 'lane lost timeout or no previous valid command'
        elif self.lane_source not in ('direct', 'estimated'):
            state = 'invalid lane tracking source'
        else:
            try:
                desired = path_steering(self.path, *self.car_center, self.lookahead,
                                        self.gain, self.offset, self.sign)
            except ValueError:
                state = 'invalid path geometry'
            else:
                smoothed = self.previous_steering + self.alpha * (desired - self.previous_steering)
                max_change = self.rate_limit * dt
                self.previous_steering = clamp(smoothed, self.previous_steering - max_change,
                                               self.previous_steering + max_change)
                command.steering = int(clamp(round(self.previous_steering), -7, 7))
                speed = self.driving_speed - (self.driving_speed - self.corner_speed) * abs(command.steering) / 7
                if self.lane_source == 'estimated':
                    speed = min(speed, self.estimated_speed)
                    self.slow_until = now + self.reacquire_hold
                    self.reacquire_pending = True
                    state = 'lane following (estimated from other lane at reduced speed)'
                else:
                    if self.reacquire_pending:
                        self.slow_until = now + self.reacquire_hold
                        self.reacquire_pending = False
                    if now < self.slow_until:
                        speed = min(speed, self.estimated_speed)
                        state = 'lane following (reacquired, holding reduced speed)'
                command.left_speed = command.right_speed = int(round(speed))
                if self.auto_active:
                    self.last_follow_time = now
                    self.last_follow_stamp = self.source_stamp
        if not recovering and not state.startswith('lane following'):
            self.previous_steering = 0.0
            self.clear_recovery(keep_slow=self.reacquire_pending)
        if state != self.state:
            self.get_logger().info(f'Control state: {state}')
            self.state = state
        self.state_pub.publish(String(data=state))
        self.publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
