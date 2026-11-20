"""Deterministic tests for short lane loss; no camera or vehicle serial is opened."""

from types import SimpleNamespace

import pytest
import rclpy
from rclpy.time import Time
from interfaces_pkg.msg import PathPlanningResult
from std_msgs.msg import Bool, String

from decision_making_pkg import motion_planner_node as module


@pytest.fixture
def controller(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    rclpy.init()
    node = module.MotionPlanningNode()
    node.timer.cancel()
    node.lost_hold_time = 0.0  # the blind lane-lost drive has its own tests below
    node.red_confirm = 1  # one report is enough here; the confirmation has its own test
    node.avoid_confirm = 1  # same for the car in my lane
    captured = []
    node.publisher = SimpleNamespace(publish=captured.append)
    node.mode_callback(String(data='autonomous_active'))

    def frame(source='direct', advance=0.1, age=0.0, malformed=False, left=False):
        clock.now += advance
        msg = PathPlanningResult(tracking_source=source)
        msg.header.stamp = Time(nanoseconds=node.get_clock().now().nanoseconds
                               - int(age * 1e9)).to_msg()
        if source != 'lost':
            msg.x_points = [170.0, 190.0, 320.0] if left else [470.0, 450.0, 320.0]
            msg.y_points = [5.0, 80.0, 179.0]
        if malformed:
            msg.x_points, msg.y_points = [float('nan'), 320.0], [5.0, 179.0]
        node.path_callback(msg)
        node.timer_callback()
        return captured[-1]

    yield node, clock, captured, frame
    node.destroy_node()
    rclpy.try_shutdown()


def test_direct_estimated_and_reacquired_speed(controller):
    node, _, _, frame = controller
    assert frame().left_speed == 255
    assert frame('estimated').left_speed == 200
    assert frame('direct').left_speed == 200
    assert frame('direct', advance=0.6).left_speed == 255


def test_lane_loss_holds_last_command_then_stops(controller):
    node, _, _, frame = controller
    node.hold_reference_speed = 255  # tests drive at 255: hold_time applies as is
    previous = frame(left=True)
    assert previous.steering < 0
    held = frame('lost')
    assert (held.steering, held.left_speed) == (previous.steering, previous.left_speed)
    assert node.state == 'holding last command (lane lost timeout or no previous valid command)'
    assert frame('lost', advance=0.3).left_speed == 255
    # New missing frames must not keep extending the hold (0.5 s since the last real command).
    assert frame('lost', advance=0.2).left_speed == 0
    assert frame('lost').left_speed == 0


def test_reacquired_steering_continues_from_the_held_command(controller):
    node, _, _, frame = controller
    frame(left=True)
    held = frame('lost')
    assert node.previous_steering == held.steering
    reacquired = frame('direct', left=True)
    assert abs(reacquired.steering - held.steering) <= 1
    assert reacquired.left_speed == 255


def test_reacquisition_after_timeout_restores_cruise_speed(controller):
    _, _, _, frame = controller
    frame()
    assert frame('lost').left_speed == 255
    assert frame('lost', advance=1.1).left_speed == 0
    assert frame('direct').left_speed == 255
    assert frame('direct', advance=0.6).left_speed == 255


def test_both_lane_loss_clears_previous_estimated_speed_hold(controller):
    _, _, _, frame = controller
    estimated = frame('estimated')
    assert estimated.left_speed == 200
    lost = frame('lost')
    assert (lost.left_speed, lost.steering) == (200, estimated.steering)
    assert frame('direct').left_speed == 255


@pytest.mark.parametrize('context', ['startup', 'manual', 'waiting', 'reset', 'start_again'])
def test_lane_loss_cannot_start_motion_without_active_following(controller, context):
    node, _, _, frame = controller
    if context != 'startup':
        frame()
    if context == 'manual':
        node.mode_callback(String(data='manual'))
        frame()  # Computed commands during manual mode do not authorize a hold.
    elif context in ('waiting', 'start_again'):
        node.mode_callback(String(data='autonomous_waiting'))
        frame()
        if context == 'start_again':
            node.mode_callback(String(data='autonomous_active'))
    elif context == 'reset':
        node.reset_callback(Bool(data=True))
    assert frame('lost').left_speed == 0


def test_camera_freeze_holds_briefly_then_stops(controller):
    node, clock, captured, frame = controller
    node.hold_reference_speed = 255  # tests drive at 255: hold_time applies as is
    node.hold_time = 1.0  # longer than path_timeout, so the hold itself is visible
    frame()
    clock.now += 0.7
    node.timer_callback()
    assert captured[-1].left_speed == 255
    assert node.state == 'holding last command (path receipt stale)'
    clock.now += 0.4
    node.timer_callback()
    assert captured[-1].left_speed == 0
    assert node.state == 'path receipt stale'
    assert frame('lost').left_speed == 0


@pytest.mark.parametrize('age', [2.0, -0.3])
def test_stale_or_future_source_frame_stops_after_the_hold(controller, age):
    node, _, _, frame = controller
    frame()
    assert frame('lost', age=age).left_speed == 255
    assert node.state == 'holding last command (path source frame stale or invalid)'
    assert frame('lost', age=age, advance=1.0).left_speed == 0
    assert node.state == 'path source frame stale or invalid'


def test_malformed_path_holds_then_stops(controller):
    _, _, _, frame = controller
    frame()
    assert frame('lost', malformed=True).left_speed == 255
    assert frame('lost', malformed=True, advance=1.1).left_speed == 0


def test_red_light_has_priority_over_lane_loss(controller):
    node, _, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='Green'))
    frame()
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame('lost').left_speed == 0
    assert node.state == 'stopped at crosswalk for red light'


def test_red_light_slows_until_crosswalk_then_waits_for_green(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    node.red_release_time = 0.0  # release on timeout disabled: only green starts the car
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=False))
    approach = frame()
    assert 0 < approach.left_speed <= 150
    assert node.state == 'lane following (red light ahead, approaching crosswalk)'
    # The light may leave the frame while approaching; the red is remembered.
    node.light_callback(String(data='None'))
    assert frame().left_speed == approach.left_speed
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    # Stays stopped even after the crosswalk/light leave the view, until green.
    node.crosswalk_callback(Bool(data=False))
    node.light_callback(String(data='None'))
    assert frame(advance=5.0).left_speed == 0
    for _ in range(3):  # green must be confirmed three times in a row
        node.light_callback(String(data='Green'))
    assert frame().left_speed == 255


def test_red_light_far_from_crosswalk_expires(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=False))
    assert frame().left_speed == 150
    clock.now += 3.5
    node.light_callback(String(data='None'))
    node.crosswalk_callback(Bool(data=False))
    assert frame().left_speed == 255


def test_space_restart_clears_red_stop(controller):
    node, _, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    node.reset_callback(Bool(data=True))
    node.light_callback(String(data='None'))
    assert frame().left_speed == 255


def test_green_after_red_crosses_straight_while_lanes_are_missing(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    clock.now += 5.0
    node.light_callback(String(data='Green'))
    # The light leaves the view and the crosswalk box is gone while the lanes are still hidden.
    node.light_callback(String(data='None'))
    node.crosswalk_callback(Bool(data=False))
    crossing = frame('lost')
    assert crossing.left_speed == 150 and crossing.steering == 0
    assert node.state == 'crossing crosswalk (lanes hidden, keeping pre-green steering)'
    # A partial lane detection without a usable path must not stop the car either.
    assert frame('direct', malformed=True).left_speed == 150
    # The detector keeps reporting 'no light' every frame while the car moves.
    clock.now += 1.8
    node.light_callback(String(data='None'))
    assert frame('lost').left_speed == 150
    # Lanes found again: normal lane following.
    assert frame().left_speed == 255
    # Window over (4 s after green): the usual lane-loss stop applies.
    clock.now += 1.0
    node.light_callback(String(data='None'))
    assert frame('lost').left_speed == 0


def test_green_without_red_does_not_open_crossing_window(controller):
    node, _, _, frame = controller
    node.hold_reference_speed = 255  # tests drive at 255: hold_time applies as is
    node.hold_time = 1.0
    node.obey_lights = True
    node.light_callback(String(data='Green'))
    node.crosswalk_callback(Bool(data=False))
    frame()
    node.light_callback(String(data='Green'))
    node.crosswalk_callback(Bool(data=False))
    held = frame('lost', advance=0.6)
    assert not node.state.startswith('crossing') and node.state.startswith('holding')
    node.light_callback(String(data='Green'))
    node.crosswalk_callback(Bool(data=False))
    assert frame('lost', advance=0.5).left_speed == 0


def test_crosswalk_crossing_is_time_limited(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    for _ in range(40):
        node.light_callback(String(data='None'))
        node.crosswalk_callback(Bool(data=True))
        last = frame('lost', advance=0.15)
    assert last.left_speed == 0


def test_red_not_seen_releases_the_stop_and_red_again_stops(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    node.red_release_time = 2.0
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    # The colour check keeps missing green: only 'Unknown' arrives for 1.6 s.
    for _ in range(5):
        node.light_callback(String(data='Unknown'))
        node.crosswalk_callback(Bool(data=True))
        assert frame(advance=0.3).left_speed == 0
    node.light_callback(String(data='Unknown'))
    node.crosswalk_callback(Bool(data=True))
    moving = frame('lost', advance=0.6)
    assert moving.left_speed == 150 and node.state.startswith('crossing')
    # Red reported again while the crosswalk is still near: stop again.
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0


def test_red_light_section_keeps_the_last_good_lane_steering(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    for _ in range(10):  # follow the lane while approaching
        node.light_callback(String(data='Red'))
        node.crosswalk_callback(Bool(data=False))
        approach = frame(left=True)
    assert approach.steering < -1
    good = max(-1, approach.steering)  # the road is straight here: at most a 1-step heading fix
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=False))
    lost = frame('lost')
    assert lost.steering == approach.steering and 0 < lost.left_speed <= 150
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    stopped = frame('lost')
    assert stopped.left_speed == 0 and stopped.steering == good
    clock.now += 5.0
    node.light_callback(String(data='Green'))
    node.crosswalk_callback(Bool(data=True))
    go = frame('lost')
    assert go.left_speed == 150 and go.steering == good



def test_default_hold_stops_a_frozen_camera_at_the_path_timeout(controller):
    node, clock, captured, frame = controller
    frame()
    clock.now += 0.7  # beyond path_timeout and beyond the 0.5 s hold
    node.timer_callback()
    assert captured[-1].left_speed == 0 and node.state == 'path receipt stale'



def test_hold_covers_the_same_distance_at_any_speed(controller):
    node, _, _, _ = controller
    from interfaces_pkg.msg import MotionCommand
    assert node.hold_limit(MotionCommand(left_speed=150, right_speed=150)) == pytest.approx(0.5)
    assert node.hold_limit(MotionCommand(left_speed=200, right_speed=200)) == pytest.approx(0.375)
    assert node.hold_limit(MotionCommand(left_speed=255, right_speed=255)) == pytest.approx(0.294, abs=0.001)
    assert node.hold_limit(MotionCommand(left_speed=100, right_speed=100)) == pytest.approx(0.5)


def test_estimate_after_green_moves_steering_by_at_most_one_step(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    for _ in range(10):
        node.light_callback(String(data='Red'))
        node.crosswalk_callback(Bool(data=False))
        before = frame().steering  # lane bends right: positive steering before the stop
    good = min(1, before)  # kept as a heading correction of at most one step
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    clock.now += 5.0
    for _ in range(5):
        # The green lamp tints the mat: the estimate swings far left; steering moves <= 1 step.
        node.light_callback(String(data='Green'))
        node.crosswalk_callback(Bool(data=True))
        drift = frame('estimated', left=True)
        assert good - 1 <= drift.steering <= good + 1 and drift.left_speed <= 150
    assert node.state.startswith('crossing crosswalk (estimate limited')
    # A lane detected directly is followed again.
    node.light_callback(String(data='None'))
    node.crosswalk_callback(Bool(data=False))
    for _ in range(5):
        direct = frame('direct', left=True)
    assert direct.steering < good - 1 or direct.steering < 0


def test_crossing_window_is_distance_based(controller):
    node, _, _, _ = controller
    node.red_approach_speed = 150
    assert node.crossing_limit() == pytest.approx(2.5)
    node.red_approach_speed = 255
    assert node.crossing_limit() == pytest.approx(2.5 * 150 / 255)



def test_yellow_keeps_the_red_stop_and_green_must_repeat(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    for _ in range(40):  # amber for 4 s: still stopped, the release timer never fires
        node.light_callback(String(data='Yellow'))
        node.crosswalk_callback(Bool(data=True))
        assert frame(advance=0.1).left_speed == 0
    node.light_callback(String(data='Green'))   # a single green frame (e.g. glare) is not enough
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    node.light_callback(String(data='Yellow'))
    node.light_callback(String(data='Green'))
    node.light_callback(String(data='Green'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    node.light_callback(String(data='Green'))
    node.crosswalk_callback(Bool(data=True))
    assert frame('lost').left_speed > 0



def test_curves_slow_down_with_steering(controller):
    node, _, _, frame = controller
    node.corner_speed = 150
    for _ in range(10):
        sharp = frame(left=True)
    assert sharp.steering <= -3
    by_wheels = round(255 - (255 - 150) * abs(sharp.steering) / 7)
    assert 150 <= sharp.left_speed <= by_wheels  # the curve ahead may slow it a little more

def test_crosswalk_in_view_slows_to_approach_speed(controller):
    node, _, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='None'))
    assert frame().left_speed == 255
    node.light_callback(String(data='None'))
    node.crosswalk_ahead_callback(Bool(data=True))
    slow = frame()
    assert slow.left_speed == 150 and node.state == 'lane following (crosswalk ahead, slowing)'
    node.light_callback(String(data='None'))
    assert frame(advance=0.6).left_speed == 255  # crosswalk out of view again


def test_speed_drops_before_the_wheels_turn_into_a_curve(controller):
    node, _, _, frame = controller
    node.corner_speed = 150
    frame()
    entering = frame(left=True)  # filter has barely moved the wheels yet
    assert abs(entering.steering) <= 2
    assert entering.left_speed < 255 - (255 - 150) * abs(entering.steering) / 7



def test_green_streak_survives_frames_without_a_light(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    clock.now += 2.0
    # Green, then the light is missed, then green again: three greens without red/amber.
    for color in ('Green', 'None', 'Green', 'Unknown', 'None', 'Green'):
        node.light_callback(String(data=color))
    node.crosswalk_callback(Bool(data=True))
    assert frame('lost').left_speed > 0
    # Amber in between still restarts the count.
    node.reset_callback(Bool(data=True))
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0
    for color in ('Green', 'Green', 'Yellow', 'Green'):
        node.light_callback(String(data=color))
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0


def cam(node, own=None, near=None, other=False):
    """Camera input as the lane extractor publishes it (boxes as x0, y0, x1, y1)."""
    from std_msgs.msg import Float32MultiArray
    nan = float('nan')
    node.camera_callback(Bool(data=own is not None))
    node.camera_other_callback(Bool(data=other))
    node.box_callback(Float32MultiArray(data=list(own or (nan,) * 4) + list(near or own or (nan,) * 4)))



def avoid_setup(node, start='lane2'):
    node.avoid_obstacles = True
    node.avoid_lane = start
    lanes = []
    node.lane_pub = type('P', (), {'publish': lambda self, m: lanes.append(m.data)})()
    return lanes


def drive(node, frame, obstacle, steps):
    for _ in range(steps):
        node.lidar_callback(Bool(data=obstacle))
        last = frame()
    return last


def test_each_obstacle_moves_the_car_to_the_other_lane_and_it_stays(controller):
    node, _, _, frame = controller
    lanes = avoid_setup(node)
    assert drive(node, frame, False, 3).left_speed == 255 and node.avoid_lane == 'lane2'
    moving = drive(node, frame, True, 1)
    assert node.avoid_lane == 'lane1' and lanes[-1] == 'lane1'
    assert 0 < moving.left_speed <= 150 and node.state == 'lane change to lane1: out'  # turns while driving
    drive(node, frame, True, 10)   # the same car is still ahead while moving over: ignored
    after = drive(node, frame, False, 200)
    assert node.avoid_lane == 'lane1' and after.left_speed == 255  # stays in lane1, full speed
    drive(node, frame, True, 1)    # next obstacle, now in lane1: back to lane2
    assert node.avoid_lane == 'lane2' and lanes[-1] == 'lane2'


def test_start_in_lane1(controller):
    node, _, _, frame = controller
    lanes = avoid_setup(node, 'lane1')
    drive(node, frame, False, 3)
    assert node.avoid_lane == 'lane1' and lanes == []
    drive(node, frame, True, 1)
    assert node.avoid_lane == 'lane2' and lanes[-1] == 'lane2'


def test_missing_lidar_drives_without_avoidance(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    assert frame().left_speed == 255 and node.avoid_lane == 'lane2' and node.lidar_warned

def test_obstacle_leaning_towards_the_target_lane_stops_instead(controller):
    from std_msgs.msg import Float32
    node, _, _, frame = controller
    avoid_setup(node)                      # in lane2, would move left to lane1
    frame()
    node.offset_callback(Float32(data=0.2))  # obstacle sits on the left of the corridor
    last = drive(node, frame, True, 2)
    assert node.avoid_lane == 'lane2' and last.left_speed == 0
    node.offset_callback(Float32(data=-0.2))  # on the right: moving left is fine
    drive(node, frame, True, 1)
    assert node.avoid_lane == 'lane1'


def test_camera_car_in_my_lane_moves_over_and_other_lane_car_blocks(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    frame()
    cam(node, own=(260, 260, 400, 365))
    frame()
    assert node.avoid_lane == 'lane1'
    # Back in a fresh setup: a car in my lane and one in the other lane -> stop, no swerve.
    node.avoid_lane, node.switch_time, node.maneuver = 'lane2', None, None
    cam(node, own=(260, 260, 400, 365), other=True)
    assert frame().left_speed == 0 and node.avoid_lane == 'lane2'


def test_lost_lane_keeps_road_steering_at_reduced_speed_then_stops(controller):
    node, _, _, frame = controller
    node.lost_hold_time = 1.5
    node.red_approach_speed = 150
    for _ in range(5):
        previous = frame(left=True)
    blind = frame('lost')
    assert node.state == 'lane lost (keeping road steering at reduced speed)'
    assert blind.steering == node.road_steering() and blind.steering < 0 and blind.left_speed == 150
    assert frame('lost', advance=1.3).left_speed == 150
    assert frame('lost', advance=0.3).left_speed == 0


def test_one_false_amber_frame_does_not_stop_the_car(controller):
    node, _, _, frame = controller
    node.obey_lights, node.red_confirm = True, 2
    node.crosswalk_callback(Bool(data=True))
    node.light_callback(String(data='Yellow'))   # an orange bench read as amber, once
    node.light_callback(String(data='None'))
    assert frame().left_speed > 0 and not node.stopped_for_red
    node.light_callback(String(data='Red'))      # a second report (frames without a light between)
    node.crosswalk_callback(Bool(data=True))
    assert frame().left_speed == 0 and node.stopped_for_red


def test_short_obstacle_gap_does_not_switch_back(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    drive(node, frame, False, 3)
    drive(node, frame, True, 1)
    assert node.avoid_lane == 'lane1'
    drive(node, frame, True, 30)          # same car still in view during the change
    drive(node, frame, False, 3)          # missed for 0.3 s
    drive(node, frame, True, 1)           # seen again: still the same car, not a new one
    assert node.avoid_lane == 'lane1'     # never switched back towards it


def test_car_seen_while_stopped_does_not_start_a_lane_change(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    frame('lost', advance=3.0)            # lane lost long ago: stopped
    assert node.state.startswith('lane lost timeout') and not node.moving
    node.lidar_callback(Bool(data=True))
    assert frame('lost').left_speed == 0
    assert node.avoid_lane == 'lane2' and node.switch_time is None


def test_car_in_my_lane_still_far_does_not_start_the_change(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    frame()
    cam(node, own=(260, 260, 380, 300))          # bottom 300: before avoid_start_y (315, ~1.8 m)
    frame()
    assert node.avoid_lane == 'lane2' and node.maneuver is None


def run_frames(node, frame, n, own=None):
    out = []
    for _ in range(n):
        cam(node, own=own); out.append(frame())
    return out


def test_no_lane_change_while_a_red_light_is_pending(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    node.obey_lights = True
    frame()
    node.light_callback(String(data='red'))
    cam(node, own=(250, 260, 400, 330))
    frame()
    assert node.avoid_lane == 'lane2' and node.maneuver is None


def test_at_the_light_the_change_starts_once_the_car_is_confirmed_in_my_lane(controller):
    # User 11-19: in front of the traffic light, turn as soon as the car is surely in my lane.
    node, _, _, frame = controller
    avoid_setup(node)
    node.obey_lights = True
    node.avoid_confirm = 3
    node.light_callback(String(data='green'))
    frame()
    node.crosswalk_callback(Bool(data=True))                   # crosswalk just passed
    for i in range(3):
        node.light_callback(String(data='green'))
        cam(node, own=(280, 230, 360, 285)); frame()          # far (y 285), but in my lane
        assert (node.maneuver is None) == (i < 2)              # third report: confirmed
    assert node.avoid_lane == 'lane1'


def test_same_car_still_ahead_after_the_change_stops(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    frame()
    cam(node, own=(250, 260, 400, 365)); frame()
    last = run_frames(node, frame, 130, own=(250, 260, 400, 365))  # never goes away
    assert last[-1].left_speed == 0 and node.state == 'obstacle in both lanes: stopped'


def test_car_passing_close_on_its_side_does_not_stop(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    frame()
    cam(node, near=(443, 300, 639, 467))             # run_04 107 s: passing it close on its left
    assert frame().left_speed > 0 and node.maneuver is None


def test_car_in_my_lane_at_a_red_light_stops_without_reversing(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    node.obey_lights = True
    frame()
    cams = []
    for _ in range(10):
        node.light_callback(String(data='red'))
        cam(node, own=(250, 260, 400, 365)); cams.append(frame())
    assert all(c.left_speed == 0 for c in cams) and node.maneuver is None


def test_a_reverse_is_never_held(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    frame()
    cam(node, near=(122, 229, 336, 410)); frame()
    for _ in range(5):
        cam(node, near=(122, 229, 336, 410)); frame()
    assert node.last_move is None or node.last_move.left_speed > 0


def test_segmentation_lane_sets_the_direction_and_corrects_the_state(controller):
    node, _, _, frame = controller
    avoid_setup(node, 'lane1')                       # state says lane1 ...
    for _ in range(12):
        node.lane_under_callback(String(data='lane2'))  # ... but the car is on lane2 for 1.2 s
        cam(node); frame()
    assert node.avoid_lane == 'lane2'                # state corrected
    node.lane_under_callback(String(data='lane2'))
    cam(node, own=(250, 260, 400, 365))
    frame()
    assert node.avoid_lane == 'lane1' and node.maneuver['direction'] == -1   # from lane2: left


def start_change(node, frame, start):
    avoid_setup(node, start)
    frame()
    node.road_history.clear()                       # straight road: the change is relative to 0
    node.last_guided_time = None
    node.lane_under_callback(String(data=start))
    cam(node, own=(250, 260, 400, 365))
    return frame()


def heading(cmds):
    """Heading (step*s) the commands give: forward adds the steering, reverse subtracts it."""
    H = 0.0; trace = []
    for c in cmds:
        H += c.steering * 0.1 * (1 if c.left_speed > 0 else -1 if c.left_speed < 0 else 0); trace.append(H)
    return H, trace


def test_change_turns_out_while_driving_without_stopping(controller):
    # User 11-19: turn from ~1.8 m right away (road stays in view); no stop, no reverse.
    node, _, _, frame = controller
    first = start_change(node, frame, 'lane2')
    assert node.avoid_lane == 'lane1' and first.left_speed > 0
    cmds = [first]
    for _ in range(10):
        node.lane_under_callback(String(data='lane2')); cam(node); cmds.append(frame('lost'))
    assert all(c.left_speed > 0 for c in cmds) and cmds[-1].steering <= -6


def test_change_from_lane1_turns_the_wheels_right(controller):
    node, _, _, frame = controller
    start_change(node, frame, 'lane1')
    cmds = []
    for _ in range(10):
        node.lane_under_callback(String(data='lane1')); cam(node); cmds.append(frame('lost'))
    assert node.avoid_lane == 'lane2' and cmds[-1].left_speed > 0 and cmds[-1].steering >= 6


def test_change_ends_when_the_segmentation_shows_the_new_lane(controller):
    node, _, _, frame = controller
    start_change(node, frame, 'lane2')
    node.lane_under_callback(String(data='lane1')); cam(node); frame('lost')
    assert node.maneuver is not None                         # a reading before turning is a flicker
    for _ in range(40):
        node.lane_under_callback(String(data='lane2')); cam(node); frame('lost')
        if node.maneuver['direction'] * node.maneuver['H'] >= 5:
            break
    node.lane_under_callback(String(data='lane1')); cam(node); frame()
    assert node.maneuver is None and node.avoid_lane == 'lane1'


def test_with_no_lane_visible_the_change_comes_back_to_the_road_heading(controller):
    node, _, _, frame = controller
    start_change(node, frame, 'lane2')
    cmds = []
    while node.maneuver is not None and len(cmds) < 200:
        cam(node); cmds.append(frame('lost'))                   # no lane, no segmentation, car gone
    H, trace = heading(cmds)
    assert min(trace) <= -13 and abs(H) <= 1.5                  # turned ~15 left, heading restored
    assert any(c.left_speed > 0 and c.steering >= 4 for c in cmds)   # turned back right
    assert node.maneuver is None and node.avoid_lane == 'lane1'


def test_segmentation_stuck_on_the_old_lane_still_ends(controller):
    node, _, _, frame = controller
    start_change(node, frame, 'lane2')
    cmds = []
    while node.maneuver is not None and len(cmds) < 250:
        node.lane_under_callback(String(data='lane2')); cam(node); cmds.append(frame('lost'))
    H, trace = heading(cmds)
    assert node.maneuver is None and min(trace) >= -27 and abs(H) <= 2.0


def test_does_not_turn_back_while_the_car_is_still_beside_on_that_side(controller):
    # 11-19 104-108 s: turning back while the car was still at our front right steered at it.
    node, _, _, frame = controller
    start_change(node, frame, 'lane2')
    for _ in range(40):
        cam(node, near=(240, 300, 456, 375)); frame('lost')
        if node.maneuver['phase'] not in ('stop', 'settle', 'reverse'):
            break
    steer = []
    for _ in range(25):                                         # car still front-right
        cam(node, near=(240, 300, 456, 375)); steer.append(frame('lost').steering)
    assert max(steer) <= 0 and node.maneuver is not None        # never steered back right
    for _ in range(40):                                         # now beside us (left edge past 470)
        cam(node, near=(480, 300, 640, 440)); steer.append(frame('lost').steering)
    assert max(steer) > 0                                        # then turned back


def test_on_a_curve_the_change_is_relative_to_the_curve(controller):
    # 205159 80 s: on a left curve (-5) the change turned a full extra 15 and left the road.
    node, _, _, frame = controller
    avoid_setup(node, 'lane2')
    for _ in range(12):
        frame(left=True)                                        # lane following a left curve
    base = node.road_steering()
    assert base <= -3
    node.lane_under_callback(String(data='lane2'))
    cam(node, own=(250, 260, 400, 365)); frame(left=True)
    assert node.maneuver['base'] == base
    cmds = []
    while node.maneuver is not None and len(cmds) < 200:
        cam(node); cmds.append(frame('lost'))
    rel = 0.0; low = 0.0
    for c in cmds:
        sgn = 1 if c.left_speed > 0 else -1 if c.left_speed < 0 else 0
        rel += (c.steering - base) * 0.1 * sgn; low = min(low, rel)
    # Relative to the curve the turn is limited by the lock (-7 on a -5 curve is only -2 more), but it
    # never goes past 15 and the heading is restored to the curve.
    assert -18 <= low <= -3 and abs(rel) <= 2.0



def test_other_lane_car_on_a_curve_is_not_a_reason_to_stop_at_1m(controller):
    # 11-19 lap 193158 247 s: in lane1, a lane2 car at x 323-576 y 365 overlapped our straight path.
    node, _, _, frame = controller
    avoid_setup(node, 'lane1')
    frame()
    cam(node, near=(323, 260, 576, 365), other=True)
    assert frame().left_speed > 0 and node.maneuver is None


def test_car_labelled_other_lane_but_straight_ahead_stops_until_sure(controller):
    # 11-19 lap 210352 47.5 s: 'lane1' at x 326-507 y 368 while the segmentation put us on lane2: hit.
    node, _, _, frame = controller
    avoid_setup(node, 'lane2')
    node.lane_under_callback(String(data='lane2'))
    frame()
    cam(node, near=(326, 260, 507, 368), other=True)
    c = frame()
    assert c.left_speed == 0 and node.maneuver is None and 'not sure' in node.state   # stop, no turn
    node.lane_under_callback(String(data='lane2'))
    cam(node, own=(326, 260, 507, 368)); c = frame()                 # now confirmed in my lane
    assert node.maneuver is not None and node.maneuver['phase'] == 'stop' and node.avoid_lane == 'lane1'


def test_other_lane_car_beside_our_path_close_is_waited_for(controller):
    node, _, _, frame = controller
    avoid_setup(node, 'lane1')
    frame()
    for _ in range(5):
        cam(node, near=(330, 260, 560, 420), other=True); c = frame()   # centre x 445: not ahead
    assert c.left_speed == 0 and node.maneuver is None
    for _ in range(30):                                   # never sure it is in my lane: keeps waiting
        cam(node, near=(330, 260, 560, 420), other=True); c = frame()
    assert c.left_speed == 0 and node.maneuver is None


def test_car_lost_by_the_detector_when_close_is_still_there(controller):
    # 11-19 lap 210352 48.5-49 s: in our path at y 372, then gone from the detections; the car hit it.
    node, _, _, frame = controller
    avoid_setup(node)
    frame()
    cam(node, near=(328, 260, 508, 390), other=True); frame()
    cam(node, near=(70, 230, 257, 313)); c = frame()            # the detector now sees only a far car
    assert c.left_speed == 0


def test_car_in_my_lane_at_1m_right_after_a_change_stops(controller):
    # Right after a lane change no new change starts (same-car window): not turning at 1 m -> stop.
    node, _, _, frame = controller
    avoid_setup(node)
    frame()
    cam(node, own=(250, 260, 400, 365)); frame()
    node.maneuver = None                                        # change over, window still running
    cam(node, own=(250, 260, 400, 370))
    c = frame()
    assert c.left_speed == 0 and 'not turning' in node.state



def test_change_needs_the_car_confirmed_in_my_lane(controller):
    node, _, _, frame = controller
    avoid_setup(node)
    node.avoid_confirm = 3
    frame()
    for i in range(3):
        cam(node, own=(250, 260, 400, 320)); frame()
        assert (node.maneuver is None) == (i < 2)


def test_no_early_change_before_the_crosswalk(controller):
    # 212007 8 s: before the stop line, a car ~4 m beyond the light started a change.
    node, _, _, frame = controller
    avoid_setup(node)
    node.obey_lights = True
    frame()
    for _ in range(5):
        node.crosswalk_ahead_callback(Bool(data=True))
        node.light_callback(String(data='none'))
        cam(node, own=(280, 230, 360, 285)); frame()
    assert node.maneuver is None


def test_my_lane_is_the_state_not_one_segmentation_reading(controller):
    # 212007 128.5 s: one lane2 reading while in lane1 started a change "to lane1".
    node, _, _, frame = controller
    avoid_setup(node, 'lane1')
    frame()
    node.lane_under_callback(String(data='lane2'))
    cam(node, own=(250, 260, 400, 330)); frame()
    assert node.maneuver is not None and node.avoid_lane == 'lane2'   # from lane1 to lane2


def test_car_confirmed_in_my_lane_starts_the_change_wherever_it_is_in_the_image(controller):
    # 11-19 lap 213025 9-14 s: left curve, my-lane car at x 1-227: a "must block x 220-420" rule kept the
    # change from starting and the car drove into it.
    node, _, _, frame = controller
    avoid_setup(node)
    frame()
    cam(node, own=(11, 230, 227, 320)); frame()
    assert node.maneuver is not None and node.avoid_lane == 'lane1'


def test_too_close_only_stops_never_reverses(controller):
    # User 11-20: the code never reverses; the operator backs off by hand when needed.
    node, _, _, frame = controller
    avoid_setup(node)
    node.lane_under_callback(String(data='lane2'))
    frame()
    cmds = []
    for _ in range(30):
        cam(node, own=(220, 300, 420, 420)); cmds.append(frame('lost'))
    assert all(c.left_speed >= 0 for c in cmds) and cmds[-1].left_speed == 0
    cam(node, own=(250, 260, 400, 365)); frame()               # change under way, then too close
    for _ in range(5):
        cam(node, near=(200, 300, 420, 420)); c = frame('lost')
        assert c.left_speed >= 0
    assert c.left_speed == 0
