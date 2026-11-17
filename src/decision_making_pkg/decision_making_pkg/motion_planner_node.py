"""Lane following with reduced speed and bounded steering hold on brief lane loss."""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from interfaces_pkg.msg import MotionCommand, PathPlanningResult
from std_msgs.msg import Bool, String

from .control_policy import MAX_DRIVING_SPEED, clamp, path_steering


# Stops caused by a late or unusable inference, as opposed to deliberate ones (red light, pause).
HOLDABLE = {
    'path receipt stale',
    'path source frame stale or invalid',
    'traffic light input stale or missing',
    'lane lost timeout or no previous valid command',
    'invalid lane tracking source',
    'invalid path geometry',
}


class MotionPlanningNode(Node):
    def __init__(self, **kwargs):
        super().__init__('motion_planner_node', **kwargs)
        self.driving_speed = int(clamp(self.declare_parameter('driving_speed', 255).value,
                                       0, MAX_DRIVING_SPEED))
        # Speed at full steering; in between it falls linearly with the steering. The steering
        # motor is slow, so S-curves need less speed than straights. -1 = no slowdown.
        corner = self.declare_parameter('corner_speed', -1).value
        self.corner_speed = self.driving_speed if corner < 0 else int(clamp(corner, 0, self.driving_speed))
        # A negative value means "same as driving_speed" (no slowdown), the launch default.
        estimated = self.declare_parameter('estimated_speed', 200).value
        self.estimated_speed = self.driving_speed if estimated < 0 else int(clamp(estimated, 0, self.driving_speed))
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
        # Red light: slow down, keep following the lane, stop once the crosswalk is near,
        # and wait there until green. red_memory covers frames where the light is missed.
        self.red_memory = self.declare_parameter('red_memory', 3.0).value
        approach = self.declare_parameter('red_approach_speed', 150).value
        self.red_approach_speed = self.driving_speed if approach < 0 else int(clamp(approach, 0, self.driving_speed))
        # The crosswalk hides the lane lines. For crossing_time seconds after red turns green
        # (or while the crosswalk is near), drive straight at red_approach_speed whenever no
        # usable path exists instead of stopping; lane following resumes as soon as one does.
        self.crossing_time = self.declare_parameter('crossing_time', 2.5).value  # at hold_reference_speed
        # Lanes are detected well while stopped at red, but the green lamp tints the mat and the
        # lane estimate drifts once it lights. So the steering of the last good lane-following
        # command before the stop is kept from the stop through the crossing, and any estimate
        # may move it by at most crossing_max_change steps, until a lane is detected directly.
        self.crossing_max_change = self.declare_parameter('crossing_max_change', 1).value
        self.crossing_max_heading = self.declare_parameter('crossing_max_heading', 1).value
        # Green must be reported this many times in a row before the car leaves a red stop.
        self.green_confirm = self.declare_parameter('green_confirm', 3).value
        self.green_streak = 0
        # Between inferences (late path, lost lane, odd detection) keep driving with the last
        # moving command for up to hold_time instead of stopping: continuous motion, not jerks.
        # hold_time is the hold at hold_reference_speed; faster commands are held proportionally
        # shorter, so the car covers the same distance blind whatever the cruise speed.
        self.hold_time = self.declare_parameter('hold_time', 0.5).value
        self.hold_reference_speed = self.declare_parameter('hold_reference_speed', 150).value
        # Green may be missed by the colour check: start when red has not been reported for
        # this long (0 disables). Red seen again while the crosswalk is near stops the car again.
        self.red_release_time = self.declare_parameter('red_release_time', 3.0).value
        self.crosswalk_memory = self.declare_parameter('crosswalk_memory', 1.0).value
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
        self.last_red_time = None
        self.crosswalk_near = False
        self.last_crosswalk_time = None
        self.last_crosswalk_near_time = None
        self.crossing_since = None
        self.cross_until = None
        self.stopped_for_red = False
        self.previous_steering = 0.0
        # Steering of the last frame with a valid lane, and the value frozen when stopping at
        # red. Near the crosswalk the lanes disappear; holding these keeps the car's heading
        # instead of the right-turn lane-loss recovery, which aimed it at the light's pole.
        self.lane_steering = 0
        self.stop_steering = 0
        self.last_move = None
        self.last_move_time = None
        self.path_count = 0
        self.last_move_path = -1
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
        self.last_crosswalk_ahead_time = None
        self.ahead_sub = self.create_subscription(Bool, 'crosswalk_ahead', self.crosswalk_ahead_callback, 1)
        self.crosswalk_sub = self.create_subscription(Bool, 'crosswalk_near',
                                                       self.crosswalk_callback, 1)
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
        self.path_count += 1
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
        if not active or not self.auto_active:
            self.last_move = None
        if not active:
            self.cross_until = None
        self.auto_active = active

    def reset_callback(self, msg):
        if msg.data:
            self.previous_steering = 0.0
            self.last_tick = time.monotonic()
            self.clear_recovery()
            # Space restart is also the manual override for a red-light stop.
            self.last_red_time = None
            self.stopped_for_red = False
            self.cross_until = None
            self.lane_steering = self.stop_steering = 0
            self.last_move = None

    def light_callback(self, msg):
        self.light = msg.data.casefold()
        self.last_light_time = time.monotonic()
        self.green_streak = self.green_streak + 1 if self.light == 'green' else 0
        if self.light in ('red', 'yellow'):  # amber means stop as well
            self.last_red_time = self.last_light_time
        elif self.light == 'green' and self.green_streak >= self.green_confirm:
            if self.stopped_for_red or self.red_recent(self.last_light_time):
                # Red -> green: the crosswalk ahead hides the lanes; allow a straight crossing.
                self.cross_until = self.last_light_time + self.crossing_limit()
            self.last_red_time = None
            self.stopped_for_red = False

    def crosswalk_ahead_callback(self, msg):
        if msg.data:
            self.last_crosswalk_ahead_time = time.monotonic()

    def crosswalk_ahead(self, now):
        # A crosswalk in view: arrive at red_approach_speed whatever the cruise speed, so the stop
        # line works the same at 150, 200 or 250 without retuning.
        return (self.obey_lights and self.last_crosswalk_ahead_time is not None
                and now - self.last_crosswalk_ahead_time <= 0.5)

    def crosswalk_callback(self, msg):
        self.crosswalk_near = msg.data
        self.last_crosswalk_time = time.monotonic()
        if msg.data:
            self.last_crosswalk_near_time = self.last_crosswalk_time

    def can_cross_crosswalk(self, now):
        if not (self.obey_lights and self.auto_active):
            self.crossing_since = None
            return False
        if self.cross_until is not None and now <= self.cross_until:
            return True
        if not (self.last_crosswalk_near_time is not None
                and now - self.last_crosswalk_near_time <= self.crosswalk_memory):
            self.crossing_since = None
            return False
        if self.crossing_since is None:
            self.crossing_since = now
        return now - self.crossing_since <= self.crossing_limit()

    def straight(self, now=None):
        """Steering to keep around the light: the last good lane steering (frozen at the stop)."""
        after_stop = now is not None and self.cross_until is not None and now <= self.cross_until
        return int(self.stop_steering if after_stop or self.stopped_for_red else self.lane_steering)

    def crossing_limit(self):
        """crossing_time is for hold_reference_speed; faster crossings last proportionally less."""
        speed = max(self.red_approach_speed, 1)
        return self.crossing_time * min(1.0, self.hold_reference_speed / speed)

    def crossing_active(self, now):
        after_green = self.cross_until is not None and now <= self.cross_until
        near = (self.last_crosswalk_near_time is not None
                and now - self.last_crosswalk_near_time <= self.crosswalk_memory)
        return self.obey_lights and self.auto_active and (after_green or near)

    def hold_limit(self, move):
        speed = max(abs(move.left_speed), abs(move.right_speed), 1)
        return self.hold_time * min(1.0, self.hold_reference_speed / speed)

    def red_recent(self, now):
        return self.last_red_time is not None and now - self.last_red_time <= self.red_memory

    def must_stop_for_red(self, now):
        if (self.stopped_for_red and self.red_release_time > 0 and self.last_red_time is not None
                and now - self.last_red_time >= self.red_release_time):
            self.get_logger().info(f'Red not seen for {self.red_release_time:.1f}s: starting')
            self.cross_until = now + self.crossing_limit()
            self.last_red_time = None
            self.stopped_for_red = False
        if (not self.stopped_for_red and self.red_recent(now) and self.crosswalk_near
                and self.last_crosswalk_time is not None
                and now - self.last_crosswalk_time <= self.source_timeout):
            self.stopped_for_red = True
            # The road is straight around the light: only a small heading correction is kept, not
            # the steering of the curve the car was still leaving (that turned it off the road).
            self.stop_steering = int(clamp(self.lane_steering, -self.crossing_max_heading, self.crossing_max_heading))
        return self.stopped_for_red

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
        elif self.obey_lights and self.must_stop_for_red(now):
            state = 'stopped at crosswalk for red light'
            command.steering = self.straight(now)  # wheels set for the start
            # Keep the steering filter at the frozen value so the start does not ramp from 0.
            self.previous_steering = float(command.steering)
            recovering = True
        elif self.path is None:
            if self.empty_path and self.lane_source == 'lost':
                # Both-lane loss has no speed reduction or post-loss speed hold.
                self.slow_until = 0.0
                self.reacquire_pending = False
            if self.can_cross_crosswalk(now):
                state = 'crossing crosswalk (lanes hidden, keeping pre-green steering)'
                after_stop = self.cross_until is not None and now <= self.cross_until
                command.steering = self.straight(now)
                self.previous_steering = float(command.steering)
                command.left_speed = command.right_speed = self.red_approach_speed
                recovering = True
            elif (self.obey_lights and self.red_recent(now) and self.empty_path
                    and self.lane_source == 'lost' and self.can_recover_lane_loss(now, ros_now)):
                state = 'lane lost approaching red light (keeping lane steering)'
                command.steering = self.straight(now)
                self.previous_steering = float(command.steering)
                command.left_speed = command.right_speed = self.red_approach_speed
                recovering = True
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
                reference = self.straight(now)
                unstable = self.lane_source == 'estimated' and self.crossing_active(now)
                if unstable:
                    # Only a small move away from the good pre-green steering is allowed.
                    desired = clamp(desired, reference - self.crossing_max_change,
                                    reference + self.crossing_max_change)
                smoothed = self.previous_steering + self.alpha * (desired - self.previous_steering)
                max_change = self.rate_limit * dt
                self.previous_steering = clamp(smoothed, self.previous_steering - max_change,
                                               self.previous_steering + max_change)
                command.steering = int(clamp(round(self.previous_steering), -7, 7))
                # Slow for the curve ahead, not only the one the wheels already follow: the target
                # steering from the path leads the filtered command into a curve.
                bend = max(abs(command.steering), min(7.0, abs(desired)))
                speed = self.driving_speed - (self.driving_speed - self.corner_speed) * bend / 7
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
                if self.crosswalk_ahead(now) and speed > self.red_approach_speed:
                    speed = self.red_approach_speed
                    state = 'lane following (crosswalk ahead, slowing)'
                if self.obey_lights and self.red_recent(now):
                    speed = min(speed, self.red_approach_speed)
                    state = 'lane following (red light ahead, approaching crosswalk)'
                if unstable:
                    speed = min(speed, self.red_approach_speed)
                    state = 'crossing crosswalk (estimate limited to the pre-green steering)'
                    recovering = True  # keep the steering filter (state is not 'lane following')
                command.left_speed = command.right_speed = int(round(speed))
                if self.auto_active:
                    self.last_follow_time = now
                    self.last_follow_stamp = self.source_stamp
                    self.lane_steering = command.steering
        if command.left_speed != 0 or command.right_speed != 0:
            # Only commands that actually drove the car may be held (not ones computed while
            # waiting or in manual mode).
            if self.auto_active:
                # The hold is timed from the last command based on a new frame, so driving on
                # an old path (up to path_timeout) does not lengthen it.
                if self.path_count != self.last_move_path or self.last_move is None:
                    self.last_move_time, self.last_move_path = now, self.path_count
                self.last_move = command
        elif (state in HOLDABLE and self.auto_active and self.last_move is not None
                and 0 <= now - self.last_move_time <= self.hold_limit(self.last_move)):
            state = f'holding last command ({state})'
            command = MotionCommand(steering=self.last_move.steering, left_speed=self.last_move.left_speed,
                                    right_speed=self.last_move.right_speed)
            self.previous_steering = float(command.steering)
            recovering = True
        elif state.startswith('stopped at crosswalk'):
            self.last_move = None
        if state.startswith('lane following'):
            self.crossing_since = None
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
