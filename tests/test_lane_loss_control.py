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
