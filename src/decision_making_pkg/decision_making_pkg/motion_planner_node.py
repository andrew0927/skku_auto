"""Lane following with reduced speed and bounded steering hold on brief lane loss."""

import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from interfaces_pkg.msg import MotionCommand, PathPlanningResult
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

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
        # Red/amber must be reported this many times (frames without a light do not count or
        # reset) before the car treats it as red: one false amber frame stopped it on a straight.
        self.red_confirm = self.declare_parameter('red_confirm', 2).value
        self.red_streak = 0
        # The car follows start_lane (given at launch). With avoid_obstacles:=true an obstacle ahead
        # (lidar) moves it to the other lane, where it stays; the next obstacle moves it back.
        # For avoid_switch_time (at hold_reference_speed) after a move the same obstacle is ignored
        # and the speed is red_approach_speed; if the obstacle never disappeared in that time both
        # lanes are blocked and the car stops.
        self.avoid_obstacles = self.declare_parameter('avoid_obstacles', False).value
        self.avoid_switch_time = self.declare_parameter('avoid_switch_time', 3.0).value
        start_lane = self.declare_parameter('start_lane', 'lane2').value
        self.avoid_lane = start_lane if start_lane in ('lane1', 'lane2') else 'lane2'
        self.obstacle = False
        self.obstacle_time = None
        self.switch_time = None
        self.cleared_since_switch = True
        self.lidar_warned = False
        self.obstacle_offset = float('nan')  # m to the left of the nearest obstacle point
        self.camera_obstacle = self.camera_other = False   # cars seen by vehicle_detector_node
        self.camera_time = None
        self.side_margin = self.declare_parameter('avoid_side_margin', 0.1).value
        # Lane change state machine: FOLLOW(lane) -> OUT -> BACK -> FOLLOW(other lane).
        # * My lane is the state itself: start_lane, swapped when a change starts (lane1 = left).
        # * Start: a car in my lane (dashed centre line check) whose box bottom passes avoid_start_y
        #   (manual demo run_04 at PWM 140-150: the human turned at y 305-322). Not while a crosswalk
        #   is in view or a red light was seen (cars wait there; the turn hid the light on 11-18).
        # * The change is a computed out-and-back, so it works even when no lane can be seen:
        #   OUT: full lock (avoid_steer) towards the new lane, ramped at avoid_steer_rate steps/s
        #   like the human in run_04, until the heading change (integral of the steering, step*s,
        #   ~1.4 deg/s per step at 150) reaches avoid_heading; BACK: full lock the other way until
        #   the heading is about to return to where it started; UNWIND: back to straight, then
        #   lane following. avoid_heading = 15: the human of run_04 (PWM 140-150) had turned 6 and
        #   23 step*s when the segmentation showed the new lane (the obstacles stand on the
        #   S-curve, so the sideways distance relative to the road differed 22 vs 54 and could not
        #   be used). Integrals use distance (time x speed/150): the same path at any speed.
        # * Whenever the segmentation shows the car on the new lane, lane following of that lane
        #   takes over at once (the human eased off right then in run_04).
        self.avoid_speed = self.declare_parameter('avoid_speed', 150).value
        # Image row of the car ahead (box bottom) vs the gap between the two cars: gap ~ k/(y-169) - c
        # with the horizon at y 169. Gap 0 at y ~465 (our hood edge; the 11-19 hits ended at box
        # bottom 468-474) and 2 m at y 310 (user, 11-19) give k 538, c 1.82: 1 m at y ~360, 0.5 m at
        # y ~400. (y 440 was taken as 1 m from the camera, not between the cars: the car hit at once.)
        # The change starts while driving at ~1.8 m (y 315): where the reverse of 11-19 lap 210352 ended
        # (box bottom 309-316) - the user: turn from there or further back, so the road stays in view.
        self.avoid_start_y = self.declare_parameter('avoid_start_y', 315.0).value
        # A car in my lane (or straight ahead whatever its label) at ~1.8 m without a change under way stops
        # the car (braking takes ~0.3 s / 10-15 px). 212007 26 s: the user stopped it at y 339 (~1.4 m) as
        # too close; 210352: at y 374.
        self.avoid_stop_y = self.declare_parameter('avoid_stop_y', 315.0).value
        # A change starts only for a car confirmed in my lane: avoid_confirm consecutive reports (user,
        # 11-19: move only when sure). Around the traffic light it starts as soon as it is confirmed,
        # whatever the distance; elsewhere from avoid_start_y.
        self.avoid_confirm = self.declare_parameter('avoid_confirm', 3).value
        self.avoid_arrive_turn = self.declare_parameter('avoid_arrive_turn', 5.0).value
        self.own_streak = 0
        self.light_zone_time = self.declare_parameter('light_zone_time', 5.0).value
        # Any car in our path (box overlapping x 220-420 by 60 px) at front_stop_y (~0.5 m) stops the car.
        # The code never reverses (user, 11-20): the operator watching the drive backs off by hand.
        self.front_stop_y = self.declare_parameter('front_stop_y', 400.0).value
        # Standing still while the wheels turn to the out lock before a change from a stop.
        self.avoid_settle = self.declare_parameter('avoid_settle', 0.3).value
        self.wait_since = None
        self.held_box = None
        self.held_box_y = self.declare_parameter('held_box_y', 380.0).value
        self.held_box_time = self.declare_parameter('held_box_time', 3.0).value
        self.avoid_steer = self.declare_parameter('avoid_steer', 7).value
        self.avoid_steer_rate = self.declare_parameter('avoid_steer_rate', 6.5).value
        self.avoid_heading = self.declare_parameter('avoid_heading', 15.0).value
        self.avoid_heading_max = self.declare_parameter('avoid_heading_max', 25.0).value  # human max 23
        # OUT longer than this (150-seconds) turns back anyway; the way back always completes, so the
        # heading is restored (a cut-off change once handed over with a 25-30 step*s turn).
        self.avoid_out_max = self.declare_parameter('avoid_out_max', 4.0).value
        self.avoid_total_max = self.declare_parameter('avoid_total_max', 12.0).value
        self.avoid_release_x = self.declare_parameter('avoid_release_x', 470).value
        self.avoid_hold_max = self.declare_parameter('avoid_hold_max', 5.0).value
        self.maneuver = None
        self.approaching = False
        self.own_box = self.near_box = None   # (x0, y0, x1, y1) from the lane extractor
        self.box_time = None
        # The obstacle counts as passed only when unseen for this long (at hold_reference_speed):
        # a one-frame miss made the car switch straight back into the lane of the same car.
        self.avoid_clear_time = self.declare_parameter('avoid_clear_time', 1.0).value
        self.obstacle_seen_time = None
        self.blocked = False
        self.moving = False  # last published command drove the car
        # The lane the car is on according to the segmentation (lane extractor): it decides the
        # direction of a change and corrects the followed lane when they disagree for lane_sync_time.
        self.lane_under = None          # (lane, time)
        self.lane_disagree_since = None
        self.lane_sync_time = self.declare_parameter('lane_sync_time', 1.0).value
        # Lane lost (white line across the road, glare, a car hiding the lane): keep driving with the
        # road steering of the last frames at red_approach_speed for lost_hold_time (at
        # hold_reference_speed, scaled like hold_time) before stopping.
        self.lost_hold_time = self.declare_parameter('lost_hold_time', 1.5).value
        self.road_window = self.declare_parameter('road_window', 1.0).value
        self.road_history = deque(maxlen=60)  # (time, steering) of guided (non-blind) commands
        self.last_guided_time = None
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
        self.lane_pub = self.create_publisher(String, 'target_lane', mode_qos)
        self.lidar_sub = self.create_subscription(Bool, 'lidar_obstacle_info', self.lidar_callback, 1)
        self.offset_sub = self.create_subscription(Float32, 'lidar_obstacle_offset', self.offset_callback, 1)
        self.cam_sub = self.create_subscription(Bool, 'camera_obstacle_info', self.camera_callback, 1)
        self.cam_other_sub = self.create_subscription(Bool, 'camera_obstacle_other', self.camera_other_callback, 1)
        self.box_sub = self.create_subscription(Float32MultiArray, 'camera_obstacle_box', self.box_callback, 1)
        self.under_sub = self.create_subscription(String, 'lane_under_car', self.lane_under_callback, 1)
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

    def guided(self, now, steering):
        self.road_history.append((now, steering))
        self.last_guided_time = now

    def mode_callback(self, msg):
        active = msg.data == 'autonomous_active'
        if not active or not self.auto_active:
            self.clear_recovery()
        if not active or not self.auto_active:
            self.last_move = None
            self.last_guided_time = None
            self.maneuver = None
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
            self.maneuver = None
            self.cross_until = None
            self.lane_steering = self.stop_steering = 0
            self.last_move = None

    def light_callback(self, msg):
        self.light = msg.data.casefold()
        self.last_light_time = time.monotonic()
        # Frames without a readable light ('none', 'unknown') carry no information: they do not
        # break the green streak (YOLO misses the close light in most frames at the stop line).
        if self.light == 'green':
            self.green_streak += 1
        elif self.light not in ('none', 'unknown'):
            self.green_streak = 0
        if self.light in ('red', 'yellow'):
            self.red_streak += 1
        elif self.light not in ('none', 'unknown'):
            self.red_streak = 0
        if self.light in ('red', 'yellow') and self.red_streak >= self.red_confirm:  # amber = stop too
            self.last_red_time = self.last_light_time
        elif self.light == 'green' and self.green_streak >= self.green_confirm:
            if self.stopped_for_red or self.red_recent(self.last_light_time):
                # Red -> green: the crosswalk ahead hides the lanes; allow a straight crossing.
                self.cross_until = self.last_light_time + self.crossing_limit()
            self.last_red_time = None
            self.stopped_for_red = False

    def lidar_callback(self, msg):
        self.obstacle = bool(msg.data)
        self.obstacle_time = time.monotonic()

    def camera_callback(self, msg):
        self.camera_obstacle, self.camera_time = bool(msg.data), time.monotonic()
        self.own_streak = self.own_streak + 1 if msg.data else 0

    def camera_other_callback(self, msg):
        self.camera_other = bool(msg.data)

    def lane_under_callback(self, msg):
        self.lane_under = (msg.data, time.monotonic()) if msg.data in ('lane1', 'lane2') else None

    def seen_lane(self, now):
        if self.lane_under is not None and now - self.lane_under[1] <= self.source_timeout:
            return self.lane_under[0]
        return None

    def sync_lane(self, now):
        """Follow the lane the segmentation sees the car on, when it disagrees long enough and no
        lane change is in progress (a false change on 11-18 left the state on the wrong lane)."""
        seen = self.seen_lane(now)
        if seen is None or seen == self.avoid_lane or self.maneuver is not None or self.switching(now):
            self.lane_disagree_since = None
            return
        if self.lane_disagree_since is None:
            self.lane_disagree_since = now
        elif now - self.lane_disagree_since >= self.lane_sync_time:
            self.set_avoid_lane(seen, 'segmentation sees the car on it')
            self.lane_disagree_since = None

    def box_callback(self, msg):
        values = list(msg.data)
        if len(values) != 8:
            return
        own, near = values[:4], values[4:]
        now = time.monotonic()
        self.own_box = tuple(own) if all(math.isfinite(v) for v in own) else None
        near = tuple(near) if all(math.isfinite(v) for v in near) else None
        # A car close in our path that the detector stops seeing is still there: at ~0.5 m it lost the car
        # (11-19 210352 49 s) and the car drove into it. Keep its last box while nothing covers that spot,
        # until we back off (reversing brings it back into view) or held_box_time passes.
        held = self.held_box
        if near is not None and near[3] >= self.held_box_y and self.in_path(now, near):
            self.held_box = (near, now)
        elif held is not None:
            covered = near is not None and min(near[2], held[0][2]) - max(near[0], held[0][0]) > 0 \
                and near[3] >= held[0][1]
            if covered or now - held[1] > self.held_box_time:
                self.held_box = None
            else:
                near = held[0]
        self.near_box = near
        self.box_time = now

    def offset_callback(self, msg):
        self.obstacle_offset = float(msg.data)

    def set_avoid_lane(self, lane, reason):
        if lane != self.avoid_lane:
            self.avoid_lane = lane
            self.get_logger().info(f'Avoidance: following {lane} ({reason})')
        self.lane_pub.publish(String(data=lane))

    def avoidance(self, now):
        """Update the followed lane; returns 'blocked' when the car must stop, else None."""
        if not self.avoid_obstacles or not self.auto_active:
            return None
        lidar_ok = self.obstacle_time is not None and now - self.obstacle_time <= self.source_timeout
        camera_ok = self.camera_time is not None and now - self.camera_time <= self.source_timeout
        if not (lidar_ok or camera_ok):
            if not self.lidar_warned:
                self.get_logger().warning('Avoidance on but no obstacle input (camera or lidar): driving without it')
                self.lidar_warned = True
        else:
            self.lidar_warned = False
        obstacle = (lidar_ok and self.obstacle) or (camera_ok and self.camera_obstacle)
        other_blocked = camera_ok and self.camera_other
        if obstacle:
            self.obstacle_seen_time = now
        elif (self.obstacle_seen_time is None
                or now - self.obstacle_seen_time >= self.avoid_clear_time * self.speed_scale()):
            self.cleared_since_switch = True
        if self.blocked:
            if self.cleared_since_switch:
                self.blocked = False  # unseen for avoid_clear_time: free again
            else:
                return 'blocked'      # a one-frame miss does not restart the car
        if self.switching(now):
            return None
        if obstacle and not self.avoid_allowed(now):
            return None  # red light: cars wait there
        if obstacle and not self.moving and (self.state or '').startswith(('lane lost timeout', 'stopped at')):
            return None  # stopped (lane lost, red light): a car seen now does not start a lane change
        own_box = self.own_box if camera_ok and self.box_time is not None and now - self.box_time <= self.source_timeout else None
        self.approaching = bool(camera_ok and self.camera_obstacle and own_box is not None)
        if camera_ok and self.camera_obstacle and not (lidar_ok and self.obstacle):
            # Only a car surely in my lane moves us (user, 11-19): avoid_confirm reports in a row.
            confirmed = self.own_streak >= self.avoid_confirm
            if own_box is None or not confirmed or (own_box[3] < self.avoid_start_y and not self.light_zone(now)):
                return None   # in my lane but still far or not yet sure: slow down (approaching), not yet OUT
        if obstacle:
            if self.switch_time is not None and not self.cleared_since_switch:
                self.blocked = True
                return 'blocked'  # still the same obstacle after moving over: both lanes blocked
            # My lane is the state (the user: we know which lane we are in; the segmentation corrects it
            # through sync_lane after 1 s). 212007 128.5 s: one lane2 reading started a "change" to the
            # lane we were already on.
            current = self.avoid_lane
            other = 'lane1' if current == 'lane2' else 'lane2'
            # lane1 is the left lane: do not steer towards the side the obstacle sits on.
            towards = self.obstacle_offset if other == 'lane1' else -self.obstacle_offset
            if other_blocked or (lidar_ok and math.isfinite(towards) and towards > self.side_margin):
                return 'blocked'
            # Turn out while driving, from ~1.8 m (or at once at the light): the road stays in view.
            # Starting at 1 m (205159 186 s) hit the car with the front corner. Reverse only when too
            # close; a car that got to 1.8 m without a change, or to 0.5 m, stops the car (timer).
            # Standing (stopped for it before it was confirmed): set the wheels first, then out.
            self.start_maneuver(now, current, 'out' if self.moving else 'stop', 'obstacle ahead'
                                + (f', car box bottom y {own_box[3]:.0f}' if own_box else ''))
        return None

    def avoid_allowed(self, now):
        """A lane change may start: not while a red light is pending or we stand at it (cars wait there)."""
        return not (self.red_recent(now) or self.stopped_for_red)

    def light_zone(self, now):
        """Just past the traffic light: crossing it (green) or the crosswalk passed within light_zone_time.
        Not before it: 11-19 lap 212007 8 s changed lanes before the stop line for a car ~4 m beyond."""
        if self.stopped_for_red or self.red_recent(now) or self.crosswalk_ahead(now):
            return False
        crossing = self.cross_until is not None and now <= self.cross_until
        passed = self.last_crosswalk_near_time is not None and now - self.last_crosswalk_near_time <= self.light_zone_time
        return crossing or passed

    def start_maneuver(self, now, current, phase, reason):
        """Lane change from current to the other lane; phase 'out' turns at once, 'stop' sets the wheels first."""
        other = 'lane1' if current == 'lane2' else 'lane2'
        self.set_avoid_lane(other, reason)
        self.switch_time, self.cleared_since_switch = now, False
        direction = -1 if other == 'lane1' else 1
        # The road's own curve: the lane-following steering of the last second. The change is computed
        # relative to it (heading and sideways offset relative to the road), so on a curve the car keeps
        # following the curve while it moves over. 11-19 lap 205159 80 s and 186 s: on a left curve
        # (steering -5) the change added its full turn to the curve and drove off the road.
        recent = self.last_guided_time is not None and now - self.last_guided_time <= self.road_window
        base = float(self.road_steering()) if recent else 0.0
        self.maneuver = {'start': now, 'direction': direction, 'target': other, 'from': current, 'phase': phase,
                         'base': base, 'steer': float(self.lane_steering), 'H': 0.0, 'Y': 0.0,
                         'phase_start': now}
        self.get_logger().info(
            f"Lane change {current} -> {other} ({'left' if direction < 0 else 'right'}, {phase}): "
            f"{direction * self.avoid_steer:+d} then {-direction * self.avoid_steer:+d} for "
            f"a {self.avoid_heading:.0f} step*s turn each way, or until the segmentation shows {other}; {reason}")

    def switching(self, now):
        scale = min(1.0, self.hold_reference_speed / max(self.red_approach_speed, 1))
        window = self.avoid_switch_time * scale
        if self.maneuver is not None:
            return True
        return self.switch_time is not None and now - self.switch_time < window

    def speed_scale(self):
        return min(1.0, self.hold_reference_speed / max(min(self.driving_speed, self.red_approach_speed), 1))

    def road_steering(self):
        """Median steering of the last road_window seconds of guided driving (0 if none)."""
        if not self.road_history:
            return 0
        latest = self.road_history[-1][0]
        recent = sorted(v for t, v in self.road_history if t >= latest - self.road_window)
        return int(recent[len(recent) // 2])

    def maneuver_command(self, now, dt):
        """(steering, speed, phase) of the lane change in progress, or None when it is over."""
        m = self.maneuver
        if m is None:
            return None
        speed = self.change_speed()
        scale = speed / float(self.hold_reference_speed)        # time -> distance (150-seconds)
        rate = self.avoid_steer_rate / max(scale, 1e-3)          # ramp per distance unit
        d, base, peak = m['direction'], m['base'], self.avoid_steer
        if m['phase'] == 'stop':
            # Standing still while the wheels go to the out lock (the servo is slow), then OUT.
            m['steer'] = float(clamp(base + d * peak, -peak, peak))
            if now - m['phase_start'] >= self.avoid_settle:
                m['phase'], m['start'] = 'out', now
            return int(round(m['steer'])), 0, 'stop'
        # Arrived only after some turn: a lane reading of the target before moving over was a flicker
        # (212007 128.5-129.5 s ended a change at once).
        arrived = self.seen_lane(now) == m['target'] and d * m['H'] >= self.avoid_arrive_turn
        rel = m['steer'] - base
        # The turn keeps growing while the steering ramps back through zero: count it in.
        turn = d * m['H'] + max(d * rel, 0.0) ** 2 / (2 * rate)
        # Do not turn back while the segmentation still clearly shows the old lane (run_04 103-106 s:
        # the human held full lock until it changed); without a reading use the computed turn.
        still_old = self.seen_lane(now) == m['from']
        out_long = now - m['start'] > self.avoid_out_max / max(scale, 1e-3)
        # Never turn back towards the car while it is still in front on that side: the human of
        # run_04 held full lock until its near edge passed x 470 (passing left) / 170 (passing
        # right). 11-19 104 s: turning back at the computed 15 steered at the car beside us.
        clear = self.passed_car(now, m)
        if m['phase'] in ('out', 'hold') and clear and ((turn >= self.avoid_heading and not still_old)
                                                        or turn >= self.avoid_heading_max or out_long):
            m['phase'] = 'back'
        elif m['phase'] == 'out' and (turn >= self.avoid_heading_max or out_long
                                      or (m.get('backed') and turn >= self.avoid_heading)):
            # After backing off the nose already points the computed turn: no more turning, drive on
            # until the car is beside us (205159 83 s: turning on to 25 left the road).
            m['phase'], m['hold_start'] = 'hold', now   # turned enough: keep the heading until beside
        elif m['phase'] == 'hold' and now - m['hold_start'] > self.avoid_hold_max / max(scale, 1e-3):
            self.get_logger().warning('Lane change: the car never got beside us, stopping')
            self.maneuver = None
            self.blocked = True
            self.cleared_since_switch = False
            self.switch_time = now
            return None
        if m['phase'] == 'back' and d * m['H'] <= rel * rel / (2 * rate) and d * rel < 0:
            m['phase'] = 'unwind'
        done = m['phase'] == 'unwind' and abs(rel) < 0.5
        if arrived or done or now - m['start'] > self.avoid_total_max:
            why = 'segmentation shows ' + m['target'] if arrived else ('done' if done else 'time limit')
            self.get_logger().info(f"Lane change {why} after {now - m['start']:.1f}s: "
                                   f"{m['Y']:+.1f} step*s^2 sideways, heading {m['H']:+.1f}")
            self.maneuver = None
            self.switch_time = now       # the same car is still ignored for avoid_switch_time
            return None
        target = clamp({'out': base + d * peak, 'hold': base, 'back': base - d * peak, 'unwind': base}[m['phase']],
                       -peak, peak)
        step = self.avoid_steer_rate * dt
        m['steer'] = clamp(target, m['steer'] - step, m['steer'] + step)
        m['H'] += (m['steer'] - base) * dt * scale
        m['Y'] += m['H'] * dt * scale
        return int(round(m['steer'])), speed, m['phase']

    def passed_car(self, now, m):
        """The car we go around is beside/behind us on the side we will turn back to."""
        box = self.near_box if self.box_time is not None and now - self.box_time <= self.source_timeout else None
        if box is None:
            if m.get('last_seen') is None:
                return True
            return now - m['last_seen'] >= 0.5
        m['last_seen'] = now
        return box[0] >= self.avoid_release_x if m['direction'] < 0 else box[2] <= 640 - self.avoid_release_x

    def in_path(self, now, box, need=60):
        """The box overlaps our path straight ahead (x 220-420) by need px."""
        return min(box[2], 420) - max(box[0], 220) >= need

    def car_ahead(self, now, row=None):
        """The nearest car straight ahead at the stop row: box bottom >= avoid_stop_y and its centre within
        our path x 220-420 (other-lane cars on curves had their centre at x 433-450), else None."""
        box = self.near_box if self.box_time is not None and now - self.box_time <= self.source_timeout else None
        row = self.avoid_stop_y if row is None else row
        if box is None or box[3] < row or not 220 <= (box[0] + box[2]) / 2 <= 420:
            return None
        return box

    def car_in_path(self, now, row):
        """The nearest car's box bottom is at or below row and it overlaps our path (x 220-420) by 60 px.
        The manual passes of run_04 kept it outside (left edge >= 399 when close); the 11-18/19 hits
        covered x 8-336 and 117-412."""
        box = self.near_box if self.box_time is not None and now - self.box_time <= self.source_timeout else None
        if box is None or box[3] < row:
            return False
        return self.in_path(now, box)

    def change_speed(self):
        return int(min(self.driving_speed, self.avoid_speed))

    def lost_limit(self):
        speed = max(min(self.driving_speed, self.red_approach_speed), 1)
        return self.lost_hold_time * min(1.0, self.hold_reference_speed / speed)

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
        recovering = blind = False
        if self.avoid_obstacles and self.auto_active:
            self.sync_lane(now)
        blocked = self.avoidance(now) == 'blocked'
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
                command.steering = self.straight(now)
                self.previous_steering = float(command.steering)
                command.left_speed = command.right_speed = self.red_approach_speed
                recovering = True
                if self.auto_active:
                    self.guided(now, command.steering)
            elif (self.obey_lights and self.red_recent(now) and self.empty_path
                    and self.lane_source == 'lost' and self.can_recover_lane_loss(now, ros_now)):
                state = 'lane lost approaching red light (keeping lane steering)'
                command.steering = self.straight(now)
                self.previous_steering = float(command.steering)
                command.left_speed = command.right_speed = self.red_approach_speed
                recovering = True
            elif (self.auto_active and self.last_guided_time is not None
                    and 0 <= now - self.last_guided_time <= self.lost_limit()):
                state = 'lane lost (keeping road steering at reduced speed)'
                command.steering = self.road_steering()
                self.previous_steering = float(command.steering)
                command.left_speed = command.right_speed = min(self.driving_speed, self.red_approach_speed)
                recovering = blind = True
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
                    if self.maneuver is None:
                        self.guided(now, command.steering)
        stop_reason = None
        if self.auto_active and self.avoid_obstacles and not blocked:
            # Never drive into a car (11-19: 4 hits). A car confirmed in my lane starts a lane change
            # (avoidance); one at ~1.2 m without a change under way, or straight ahead with its lane not
            # sure, stops the car. Any car in our path at ~0.5 m stops it. We move only when sure (user,
            # 11-19) and never reverse (user, 11-20).
            m = self.maneuver
            own_box = self.own_box if self.camera_obstacle and self.box_time is not None \
                and now - self.box_time <= self.source_timeout else None
            own_close = own_box is not None and own_box[3] >= self.avoid_stop_y and self.in_path(now, own_box)
            ahead = self.car_ahead(now)
            if m is None and ahead is not None and not own_close:
                # Straight ahead at ~1.2 m but labelled the other lane or unknown (210352 47.5 s: 'lane1' at
                # x 326-507 while the segmentation put us on lane2; we hit it). Not sure it is in my lane,
                # so no lane change: stop and wait (it starts once the car is confirmed in my lane).
                stop_reason = f'car straight ahead at y {ahead[3]:.0f}, lane not sure: stopped'
            if not self.car_in_path(now, self.front_stop_y):
                self.wait_since = None
            if stop_reason is not None:
                pass
            elif m is None and own_close:
                # avoidance() ran just before: it did not start a change (traffic light, right after a
                # change, ...), so the car is not turning away at 1 m: stop.
                stop_reason = f'car in my lane at y {own_box[3]:.0f}, not turning: stopped'
            elif self.car_in_path(now, self.front_stop_y):
                where = f'car in our path at y {self.near_box[3]:.0f}'
                # Too close: stop. The code never reverses (user, 11-20): the operator watching the drive
                # backs off by hand when needed. A change under way is dropped.
                stop_reason = f'{where}: stopped'
                self.maneuver = None
        profile = self.maneuver_command(now, dt) if self.auto_active else None
        if stop_reason is not None:
            state = stop_reason
            command = MotionCommand(steering=command.steering)
            profile = None
        elif blocked and self.auto_active:
            state = 'obstacle in both lanes: stopped'
            command = MotionCommand(steering=command.steering)
            self.maneuver = None
        elif profile is not None and not state.startswith('stopped'):
            steering, speed, phase = profile
            state = f'lane change to {self.avoid_lane}: {phase}'
            command = MotionCommand(steering=steering, left_speed=speed, right_speed=speed)
            self.previous_steering = float(steering)  # the ramp continues from what was sent
            self.last_guided_time = now
            recovering = True
        elif self.approaching and self.auto_active and command.left_speed > self.avoid_speed:
            command.left_speed = command.right_speed = self.avoid_speed
            if state.startswith('lane following'):
                state = 'lane following (car ahead in my lane, slowing)'
        elif self.switching(now) and self.auto_active:
            if command.left_speed > self.red_approach_speed:
                command.left_speed = command.right_speed = self.red_approach_speed
            if state.startswith('lane following'):
                state = f'lane following (moving to {self.avoid_lane} around an obstacle)'
        if command.left_speed != 0 or command.right_speed != 0:
            # Only commands that actually drove the car may be held (not ones computed while
            # waiting or in manual mode).
            # The blind lane-lost drive is itself the hold: it is not held again when it ends.
            if self.auto_active and not blind and command.left_speed > 0:  # never hold a reverse
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
        self.moving = command.left_speed != 0 or command.right_speed != 0
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
