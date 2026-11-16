"""Lane tracking, crosswalk steering guard, red lights and lidar avoidance."""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from interfaces_pkg.msg import MotionCommand, PathPlanningResult, DetectionArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from lidar_perception_pkg.scan_geometry import scan_is_fresh, scan_is_valid, stamp_seconds

from .control_policy import MAX_DRIVING_SPEED, clamp, path_steering
from .crosswalk_stop import RedLightStopController, crosswalk_edge_ratio


class MotionPlanningNode(Node):
    def __init__(self, **kwargs):
        super().__init__('motion_planner_node', **kwargs)
        def setting(name, default):
            return self.declare_parameter(name, default).value
        self.driving_speed = int(clamp(setting('driving_speed', 255), 0, MAX_DRIVING_SPEED))
        self.corner_speed = int(clamp(setting('corner_speed', self.driving_speed), 0, self.driving_speed))
        self.estimated_speed = int(clamp(setting('estimated_speed', 200), 0, self.driving_speed))
        self.lane_loss_hold = setting('lane_loss_hold', 3.0)
        self.recovery_speed = int(clamp(setting('recovery_speed', 70), 0, self.driving_speed))
        self.recovery_steering = int(clamp(setting('recovery_steering', 2), 0, 7))
        self.sweep_interval = setting('recovery_sweep_interval', 1.0)
        self.recovery_exit_step = setting('recovery_exit_speed_step', 20)
        self.reacquire_hold = setting('reacquire_hold', 0.5)
        self.path_timeout = setting('path_timeout', 0.6)
        self.path_gap_grace = setting('path_gap_grace', 0.3)
        self.source_timeout = setting('source_timeout', 1.0)
        self.lookahead = setting('lookahead', 100.0)
        self.gain = setting('steering_gain', 0.12)
        self.offset = setting('steering_offset', 0.0)
        self.sign = setting('steering_sign', 1.0)
        self.alpha = setting('steering_alpha', 0.4)
        self.rate_limit = setting('steering_rate_limit', 10.0)
        self.left_curve_offset = setting('left_curve_offset', 0.0)
        self.right_curve_offset = setting('right_curve_offset', 0.0)
        self.crosswalk_guard = setting('enable_crosswalk_guard', True)
        self.crosswalk_rate = setting('crosswalk_steering_rate', 5.0)
        self.car_center = setting('car_center_point', [320.0, 179.0])
        self.obey_lights = setting('obey_traffic_lights', True)
        self.red_stop_at_crosswalk = setting('red_stop_at_crosswalk', True)
        self.crosswalk_stop = RedLightStopController(
            setting('crosswalk_stop_y_ratio', 325.0 / 480.0), setting('crosswalk_approach_timeout', 3.0))
        self.crosswalk_stop_edge = setting('crosswalk_stop_edge', 'far_edge')
        if self.crosswalk_stop_edge not in ('near_edge', 'far_edge'):
            raise ValueError('crosswalk_stop_edge must be near_edge or far_edge')
        self.crosswalk_approach_speed = int(clamp(setting('crosswalk_approach_speed', 70), 0, self.driving_speed))
        self.crosswalk_position_timeout = setting('crosswalk_position_timeout', 0.3)
        self.avoidance_enabled = setting('enable_obstacle_avoidance', True)
        self.lidar_timeout = setting('lidar_timeout', 0.6)
        self.swerve_speed = int(clamp(setting('swerve_speed', 80), 0, self.driving_speed))
        self.swerve_angle = int(clamp(setting('swerve_angle', 7), 0, 7))
        self.swerve_duration = setting('swerve_duration', 4.0)
        self.swerve_cooldown = setting('swerve_cooldown', 4.5)
        self.preferred_lane = setting('target_lane', 'lane2')
        if (min(self.path_timeout, self.source_timeout, self.lookahead, self.rate_limit,
                self.crosswalk_rate, self.lidar_timeout, self.sweep_interval,
                self.crosswalk_position_timeout) <= 0
                or not 0 < self.alpha <= 1 or self.sign not in (-1.0, 1.0)
                or len(self.car_center) != 2 or not 0 <= self.lane_loss_hold <= 5.0
                or self.reacquire_hold < 0 or self.recovery_exit_step <= 0
                or not 0 <= self.path_gap_grace <= 0.5
                or self.swerve_duration <= 0 or self.swerve_cooldown <= self.swerve_duration
                or self.preferred_lane not in ('lane1', 'lane2')):
            raise ValueError('Invalid controller parameters')
        self.current_lane = self.preferred_lane
        self.path = None
        self.path_lane = ''
        self.empty_path = False
        self.last_path_time = None
        self.source_stamp = 0.0
        self.lane_source = 'lost'
        self.auto_active = False
        self.light = self.last_light_time = None
        self.signal_present = False
        self.last_signal_presence_time = None
        self.crosswalk = False
        self.detection_stamp = 0.0
        self.detection_received = None
        self.crosswalk_position = None
        self.scan = self.last_scan_time = self.last_obstacle_time = None
        self.obstacle = False
        self.avoidance_ready = True
        self.swerve_until = 0.0
        self.last_swerve = -math.inf
        self.swerve_steering = 0
        self.previous_steering = 0.0
        self.last_speed = 0
        self.last_tick = time.monotonic()
        self.state = None
        self.clear_recovery()
        self.publisher = self.create_publisher(MotionCommand, 'autonomous_control_signal', 1)
        self.state_pub = self.create_publisher(String, 'control_state', 1)
        self.crosswalk_debug_pub = self.create_publisher(String, 'crosswalk_stop_debug', 1)
        mode_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.lane_pub = self.create_publisher(String, 'target_lane', mode_qos)
        self.path_sub = self.create_subscription(PathPlanningResult, 'path_planning_result', self.path_callback, 1)
        self.reset_sub = self.create_subscription(Bool, 'driving_mode_reset', self.reset_callback, 1)
        self.light_sub = self.create_subscription(String, 'yolov8_traffic_light_info', self.light_callback, 1)
        self.light_presence_sub = self.create_subscription(
            Bool, 'traffic_light_presence', self.light_presence_callback, 1)
        self.detection_sub = self.create_subscription(DetectionArray, 'detections', self.detection_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, 'lidar_processed', self.scan_callback, 1)
        self.obstacle_sub = self.create_subscription(Bool, 'lidar_obstacle_info', self.obstacle_callback, 1)
        self.mode_sub = self.create_subscription(String, 'driving_mode', self.mode_callback, mode_qos)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def clear_recovery(self, keep_slow=False):
        self.last_follow_time = None
        self.last_follow_lane = None
        self.last_follow_steering = 0
        self.last_follow_path_time = None
        self.loss_started = None
        self.was_recovering = False
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
        self.lane_source, self.path_lane = msg.tracking_source, msg.target_lane

    def mode_callback(self, msg):
        active = msg.data == 'autonomous_active'
        if not active or not self.auto_active:
            self.clear_recovery()
            self.swerve_until = 0.0
        self.auto_active = active

    def reset_callback(self, msg):
        if msg.data:
            self.previous_steering = 0.0
            self.last_tick = time.monotonic()
            self.current_lane = self.preferred_lane
            self.swerve_until = 0.0
            self.avoidance_ready = True
            self.last_swerve = -math.inf
            self.clear_recovery()
            self.lane_pub.publish(String(data=self.current_lane))

    def light_callback(self, msg):
        self.light, self.last_light_time = msg.data.casefold(), time.monotonic()
        self.crosswalk_stop.observe_light(msg.data)

    def light_presence_callback(self, msg):
        self.signal_present = msg.data
        self.last_signal_presence_time = time.monotonic()

    def has_signal_context(self, now):
        return (self.signal_present and self.last_signal_presence_time is not None
                and 0 <= now - self.last_signal_presence_time <= self.source_timeout)

    def crosswalk_is_active(self, now, ros_now):
        return (self.crosswalk_guard and self.crosswalk and self.has_signal_context(now)
                and self.detection_stamp > 0
                and -0.05 <= ros_now - self.detection_stamp <= self.source_timeout)

    def detection_callback(self, msg):
        self.crosswalk = any(d.class_name.casefold() == 'crosswalk' for d in msg.detections)
        self.detection_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.detection_received = time.monotonic()
        self.crosswalk_position = crosswalk_edge_ratio(msg.detections, self.crosswalk_stop_edge)

    def red_light_command(self, now, ros_now, dt):
        fresh = (self.detection_received is not None and self.detection_stamp > 0
                 and now - self.detection_received <= self.crosswalk_position_timeout
                 and -0.05 <= ros_now - self.detection_stamp <= self.crosswalk_position_timeout)
        can_approach = (self.red_stop_at_crosswalk and self.auto_active
                        and (self.has_signal_context(now) or self.crosswalk_stop.approach_started is not None)
                        and self.last_follow_lane == self.current_lane
                        and self.last_follow_time is not None and self.last_speed > 0
                        and self.path is not None and self.path_lane == self.current_lane
                        and self.lane_source in ('direct', 'estimated')
                        and self.last_path_time is not None and now - self.last_path_time <= self.path_timeout
                        and self.source_stamp > 0 and -0.05 <= ros_now - self.source_stamp <= self.source_timeout
                        and not self.swerve_until
                        and (not self.avoidance_enabled or
                             (self.lidar_is_fresh(now, ros_now) and not self.obstacle)))
        action = self.crosswalk_stop.evaluate(now, self.crosswalk_position, fresh, can_approach)
        if action == 'approach':
            command, state, _ = self.lane_command(now, ros_now, dt)
            if not state.startswith('lane following') or command.left_speed <= 0:
                self.crosswalk_stop.stop('invalid lane geometry during red approach')
            else:
                speed = min(command.left_speed, self.crosswalk_approach_speed, self.last_speed)
                command.left_speed = command.right_speed = speed
                self.was_recovering = True
                return command, 'red light approach to crosswalk', True
        return MotionCommand(), 'red traffic light', False

    def scan_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if (scan_is_valid(msg) and scan_is_fresh(msg, now, self.source_timeout)
                and (self.scan is None or stamp_seconds(msg) > stamp_seconds(self.scan))):
            self.scan, self.last_scan_time = msg, time.monotonic()

    def obstacle_callback(self, msg):
        self.obstacle, self.last_obstacle_time = msg.data, time.monotonic()
        if not msg.data:
            self.avoidance_ready = True

    def lidar_is_fresh(self, now, ros_now):
        return (self.scan is not None and self.last_scan_time is not None
                and self.last_obstacle_time is not None
                and now - self.last_scan_time <= self.lidar_timeout
                and now - self.last_obstacle_time <= self.lidar_timeout
                and scan_is_fresh(self.scan, ros_now, self.source_timeout))

    def avoidance_command(self, now):
        if self.swerve_until and now >= self.swerve_until:
            self.swerve_until = 0.0
            self.was_recovering = True
        if not self.swerve_until and self.obstacle:
            if not self.avoidance_ready or now - self.last_swerve < self.swerve_cooldown:
                return MotionCommand(), 'obstacle remains or avoidance cooldown'
            if self.path is None or self.path_lane != self.current_lane:
                return MotionCommand(), 'obstacle detected without a valid target lane'
            self.current_lane = 'lane1' if self.current_lane == 'lane2' else 'lane2'
            self.swerve_steering = -self.swerve_angle if self.current_lane == 'lane1' else self.swerve_angle
            self.last_swerve, self.swerve_until = now, now + self.swerve_duration
            self.avoidance_ready = False
            self.path = None
            self.empty_path = False
            self.lane_pub.publish(String(data=self.current_lane))
        if self.swerve_until:
            self.previous_steering = float(self.swerve_steering)
            return MotionCommand(steering=self.swerve_steering, left_speed=self.swerve_speed,
                                 right_speed=self.swerve_speed), 'lidar obstacle avoidance'
        return None, None

    def path_gap_command(self, now):
        # Bridge only a brief planner delay, never a lost-lane or mission command.
        if (not self.auto_active or self.last_follow_time is None or self.last_speed <= 0
                or self.last_follow_lane != self.current_lane or self.path_lane != self.current_lane
                or self.last_follow_path_time != self.last_path_time
                or self.path is None or self.lane_source not in ('direct', 'estimated')
                or self.swerve_until or (self.avoidance_enabled and self.obstacle)
                or self.path_gap_grace <= 0
                or now - self.last_path_time > self.path_timeout + self.path_gap_grace):
            return None
        self.was_recovering = True
        self.previous_steering = float(self.last_follow_steering)
        speed = min(self.recovery_speed, self.last_speed)
        return MotionCommand(steering=self.last_follow_steering,
                             left_speed=speed, right_speed=speed)

    def lane_command(self, now, ros_now, dt):
        if self.path_lane != self.current_lane:
            return MotionCommand(), 'waiting for the selected lane', False
        if self.path is None:
            if (self.empty_path and self.lane_source == 'lost' and self.auto_active
                    and self.last_follow_time is not None and self.lane_loss_hold > 0):
                if self.loss_started is None:
                    self.loss_started = now
                elapsed = now - self.loss_started
                if elapsed < self.lane_loss_hold:
                    self.was_recovering = True
                    steering = self.previous_steering
                    if elapsed >= self.sweep_interval:
                        phase = int((elapsed - self.sweep_interval) / self.sweep_interval)
                        steering = self.offset + (1 if phase % 2 == 0 else -1) * self.recovery_steering
                    steering = int(clamp(round(steering), -7, 7))
                    self.previous_steering = float(steering)
                    return MotionCommand(steering=steering, left_speed=self.recovery_speed,
                                         right_speed=self.recovery_speed), 'lane lost (low-speed search)', True
            return MotionCommand(), 'lane lost timeout or no previous valid command', False
        if self.lane_source not in ('direct', 'estimated'):
            return MotionCommand(), 'invalid lane tracking source', False
        try:
            desired = path_steering(self.path, *self.car_center, self.lookahead,
                                    self.gain, self.offset, self.sign)
        except ValueError:
            return MotionCommand(), 'invalid path geometry', False
        if desired < 0:
            desired = min(0, desired + self.left_curve_offset)
        elif desired > 0:
            desired = max(0, desired - self.right_curve_offset)
        crosswalk = self.crosswalk_is_active(now, ros_now)
        rate = min(self.rate_limit, self.crosswalk_rate) if crosswalk else self.rate_limit
        smoothed = self.previous_steering + self.alpha * (desired - self.previous_steering)
        self.previous_steering = clamp(smoothed, self.previous_steering - rate * dt,
                                       self.previous_steering + rate * dt)
        steering = int(clamp(round(self.previous_steering), -7, 7))
        speed = self.driving_speed - (self.driving_speed - self.corner_speed) * abs(steering) / 7
        state = 'lane following'
        if self.lane_source == 'estimated':
            speed = min(speed, self.estimated_speed)
            self.slow_until, self.reacquire_pending = now + self.reacquire_hold, True
            state = 'lane following (estimated from other lane at reduced speed)'
        else:
            if self.reacquire_pending:
                self.slow_until, self.reacquire_pending = now + self.reacquire_hold, False
            if now < self.slow_until:
                speed = min(speed, self.estimated_speed)
                state = 'lane following (reacquired, holding reduced speed)'
        if self.was_recovering:
            target_speed = speed
            speed = min(speed, self.last_speed + self.recovery_exit_step)
            if speed >= target_speed:
                self.was_recovering = False
        if self.auto_active:
            self.last_follow_time = now
            self.last_follow_lane = self.current_lane
            self.last_follow_steering = steering
            self.last_follow_path_time = self.last_path_time
        self.loss_started = None
        if crosswalk:
            state += ' (crosswalk steering guard)'
        return MotionCommand(steering=steering, left_speed=int(round(speed)),
                             right_speed=int(round(speed))), state, False

    def timer_callback(self):
        now = time.monotonic()
        dt = clamp(now - self.last_tick, 0.0, 0.2)
        self.last_tick = now
        ros_now = self.get_clock().now().nanoseconds / 1e9
        command, state, recovering = MotionCommand(), None, False
        self.lane_pub.publish(String(data=self.current_lane))
        if self.obey_lights and (self.last_light_time is None or now - self.last_light_time > self.source_timeout):
            state = 'traffic light input stale or missing'
        elif self.obey_lights and self.crosswalk_stop.red:
            command, state, recovering = self.red_light_command(now, ros_now, dt)
        elif self.avoidance_enabled and not self.lidar_is_fresh(now, ros_now):
            state = 'lidar input stale or missing'
        elif self.last_path_time is None:
            state = 'path missing or invalid'
        elif self.source_stamp <= 0 or not -0.05 <= ros_now - self.source_stamp <= self.source_timeout:
            state = 'path source frame stale or invalid'
        elif now - self.last_path_time > self.path_timeout:
            bridged = self.path_gap_command(now)
            if bridged is not None:
                command, recovering = bridged, True
                state = 'brief path gap (low-speed hold)'
            else:
                state = 'path receipt stale'
        else:
            if self.avoidance_enabled and self.auto_active:
                forced, state = self.avoidance_command(now)
                if forced is not None:
                    command, recovering = forced, state == 'lidar obstacle avoidance'
            if state is None:
                command, state, recovering = self.lane_command(now, ros_now, dt)
        if not recovering and not state.startswith('lane following'):
            self.previous_steering = 0.0
            self.swerve_until = 0.0
            self.clear_recovery(keep_slow=self.reacquire_pending)
        self.last_speed = command.left_speed
        if state != self.state:
            self.get_logger().info(f'Control state: {state}')
            self.state = state
        self.state_pub.publish(String(data=state))
        position = 'missing' if self.crosswalk_position is None else f'{self.crosswalk_position:.3f}'
        self.crosswalk_debug_pub.publish(String(data=(
            f'signal_present={self.has_signal_context(now)} '
            f'crosswalk_active={self.crosswalk_is_active(now, ros_now)} '
            f'edge={self.crosswalk_stop_edge} position={position} stop_y={self.crosswalk_stop.stop_ratio:.3f} '
            f'red={self.crosswalk_stop.red} stopped={self.crosswalk_stop.stopped} '
            f'reason={self.crosswalk_stop.reason}')))
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
