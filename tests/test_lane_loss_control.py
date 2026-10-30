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
    assert frame('lost').left_speed == 0
    assert node.state == 'red traffic light'
