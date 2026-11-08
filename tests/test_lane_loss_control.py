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


def test_both_lanes_missing_steer_right_without_slowing_then_stop(controller):
    node, _, _, frame = controller
    previous = frame(left=True)
    assert previous.steering < 0
    short_loss = frame('lost')
    assert short_loss.left_speed == short_loss.right_speed == 255
    assert short_loss.steering == 4
    assert frame('lost', advance=0.15).left_speed == 255
    # New missing frames must not keep extending the hold interval.
    assert frame('lost', advance=0.2).left_speed == 0
    assert frame('lost').left_speed == 0
    assert node.last_follow_time is None


def test_reacquired_steering_resumes_from_the_right_recovery_command(controller):
    node, _, _, frame = controller
    frame(left=True)
    recovery = frame('lost')
    assert recovery.steering == node.previous_steering == 4
    reacquired = frame('direct', left=True)
    assert abs(reacquired.steering - recovery.steering) <= 1
    assert reacquired.left_speed == 255


def test_reacquisition_after_timeout_restores_cruise_speed(controller):
    _, _, _, frame = controller
    frame()
    assert frame('lost').left_speed == 255
    assert frame('lost', advance=0.5).left_speed == 0
    assert frame('direct').left_speed == 255
    assert frame('direct', advance=0.6).left_speed == 255


def test_both_lane_loss_clears_previous_estimated_speed_hold(controller):
    _, _, _, frame = controller
    assert frame('estimated').left_speed == 200
    lost = frame('lost')
    assert lost.left_speed == lost.right_speed == 255
    assert lost.steering == 4
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


def test_camera_freeze_stops_instead_of_holding(controller):
    node, clock, captured, frame = controller
    frame()
    clock.now += 0.7
    node.timer_callback()
    assert captured[-1].left_speed == 0
    assert node.state == 'path receipt stale'
    assert frame('lost').left_speed == 0


@pytest.mark.parametrize('age', [2.0, -0.3])
def test_stale_or_future_source_frame_cannot_trigger_hold(controller, age):
    node, _, _, frame = controller
    frame()
    assert frame('lost', age=age).left_speed == 0
    assert node.state == 'path source frame stale or invalid'


def test_malformed_path_is_not_treated_as_temporary_lane_loss(controller):
    _, _, _, frame = controller
    frame()
    assert frame('lost', malformed=True).left_speed == 0


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
    assert node.state == 'crossing crosswalk (lanes hidden, holding lane heading)'
    # A partial lane detection without a usable path must not stop the car either.
    assert frame('direct', malformed=True).left_speed == 150
    # The detector keeps reporting 'no light' every frame while the car moves.
    clock.now += 3.0
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
    node.obey_lights = True
    node.light_callback(String(data='Green'))
    node.crosswalk_callback(Bool(data=False))
    frame()
    assert frame('lost', advance=0.6).left_speed == 0


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


def test_red_approach_keeps_lane_heading_through_stop_and_green(controller):
    node, clock, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=False))
    for _ in range(10):  # follow a left-curving lane while approaching
        node.light_callback(String(data='Red'))
        node.crosswalk_callback(Bool(data=False))
        approach = frame(left=True)
    held = approach.steering
    assert held < 0
    # Lanes vanish at the crosswalk: hold the lane heading, never the +4 right recovery.
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=False))
    lost = frame('lost')
    assert lost.steering == held and 0 < lost.left_speed <= 150
    node.light_callback(String(data='Red'))
    node.crosswalk_callback(Bool(data=True))
    stopped = frame('lost')
    assert stopped.left_speed == 0 and stopped.steering == held
    clock.now += 5.0
    node.light_callback(String(data='Green'))
    node.crosswalk_callback(Bool(data=True))
    go = frame('lost')
    assert go.left_speed == 150 and go.steering == held
