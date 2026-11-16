"""Use a deterministic clock to check recovery without vehicle hardware."""

from types import SimpleNamespace

import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.time import Time
from interfaces_pkg.msg import PathPlanningResult
from std_msgs.msg import Bool, String

from decision_making_pkg import motion_planner_node as module


@pytest.fixture
def controller(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    rclpy.init()
    node = module.MotionPlanningNode(parameter_overrides=[
        Parameter('obey_traffic_lights', value=False),
        Parameter('enable_obstacle_avoidance', value=False),
    ])
    node.timer.cancel()
    captured = []
    node.publisher = SimpleNamespace(publish=captured.append)
    node.mode_callback(String(data='autonomous_active'))

    def frame(source='direct', advance=0.1, age=0.0, malformed=False, left=False):
        clock.now += advance
        msg = PathPlanningResult(tracking_source=source, target_lane=node.current_lane)
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


def test_normal_and_estimated_speeds_are_preserved(controller):
    _, _, _, frame = controller
    assert frame().left_speed == 255
    assert frame('estimated').left_speed == 200
    assert frame('direct').left_speed == 200
    assert frame('direct', advance=0.6).left_speed == 255


def test_missing_lanes_hold_then_sweep_at_70_for_at_most_three_seconds(controller):
    node, _, _, frame = controller
    previous = frame(left=True)
    first = frame('lost')
    assert first.left_speed == first.right_speed == 70
    assert first.steering == previous.steering
    assert frame('lost', advance=1.1).steering == 2
    assert frame('lost', advance=1.0).steering == -2
    assert frame('lost', advance=1.0).left_speed == 0
    assert frame('lost').left_speed == 0
    assert node.last_follow_time is None


def test_reacquisition_ramps_back_to_the_same_cruise_speed(controller):
    node, _, _, frame = controller
    frame()
    frame('lost')
    assert frame().left_speed == 90
    speeds = [frame().left_speed for _ in range(10)]
    assert speeds == sorted(speeds)
    assert max(b - a for a, b in zip(speeds, speeds[1:])) <= 20
    assert speeds[-1] == 255
    assert not node.was_recovering


@pytest.mark.parametrize('context', ['startup', 'manual', 'waiting', 'reset', 'start_again'])
def test_lane_loss_never_starts_a_stopped_vehicle(controller, context):
    node, _, _, frame = controller
    if context != 'startup':
        frame()
    if context == 'manual':
        node.mode_callback(String(data='manual'))
        frame()
    elif context in ('waiting', 'start_again'):
        node.mode_callback(String(data='autonomous_waiting'))
        frame()
        if context == 'start_again':
            node.mode_callback(String(data='autonomous_active'))
    elif context == 'reset':
        node.reset_callback(Bool(data=True))
    assert frame('lost').left_speed == 0


def test_camera_freeze_stops_without_searching(controller):
    node, clock, captured, frame = controller
    frame()
    clock.now += 0.95
    node.timer_callback()
    assert captured[-1].left_speed == 0
    assert node.state == 'path receipt stale'
    assert frame('lost').left_speed == 0


def test_brief_path_gap_holds_last_valid_steering_at_low_speed(controller):
    node, clock, captured, frame = controller
    last_valid = frame(left=True)
    clock.now += 0.7
    node.timer_callback()
    held = captured[-1]
    assert held.left_speed == held.right_speed == 70
    assert held.steering == last_valid.steering
    assert node.state == 'brief path gap (low-speed hold)'
    clock.now += 0.1
    node.timer_callback()
    assert captured[-1].left_speed == 70
    assert frame().left_speed == 90


@pytest.mark.parametrize('context', ['startup', 'waiting', 'lost', 'malformed', 'switched_lane', 'disabled'])
def test_brief_gap_never_uses_invalid_or_wrong_lane_history(controller, context):
    node, clock, captured, frame = controller
    if context != 'startup':
        frame()
    if context == 'waiting':
        node.mode_callback(String(data='autonomous_waiting'))
    elif context == 'lost':
        frame('lost')
    elif context == 'malformed':
        frame(malformed=True)
    elif context == 'switched_lane':
        node.current_lane = 'lane1'
    elif context == 'disabled':
        node.path_gap_grace = 0.0
    clock.now += 0.7
    node.timer_callback()
    assert captured[-1].left_speed == 0


def test_source_age_still_limits_brief_gap(controller):
    node, clock, captured, frame = controller
    frame(age=1.1)
    clock.now += 0.7
    node.timer_callback()
    assert captured[-1].left_speed == 0
    assert node.state == 'path source frame stale or invalid'


def test_new_unvalidated_geometry_cannot_extend_last_good_command(controller):
    node, clock, captured, frame = controller
    frame()
    msg = PathPlanningResult(tracking_source='direct', target_lane=node.current_lane,
                             x_points=[320.0, 320.0], y_points=[200.0, 300.0])
    msg.header.stamp = node.get_clock().now().to_msg()
    clock.now += 0.1
    node.path_callback(msg)
    clock.now += 0.7
    node.timer_callback()
    assert captured[-1].left_speed == 0


@pytest.mark.parametrize('age', [2.0, -0.3])
def test_bad_frame_timestamps_cannot_trigger_recovery(controller, age):
    node, _, _, frame = controller
    frame()
    assert frame('lost', age=age).left_speed == 0
    assert node.state == 'path source frame stale or invalid'


def test_malformed_path_is_not_lane_loss(controller):
    _, _, _, frame = controller
    frame()
    assert frame('lost', malformed=True).left_speed == 0


def test_red_light_precedes_recovery(controller):
    node, _, _, frame = controller
    node.obey_lights = True
    node.light_callback(String(data='Green'))
    frame()
    node.light_callback(String(data='Red'))
    assert frame('lost').left_speed == 0
    assert node.state == 'red traffic light'
