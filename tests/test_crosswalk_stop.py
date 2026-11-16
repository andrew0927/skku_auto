"""Crosswalk stop lines, missing detections and red-to-green state transitions."""

from types import SimpleNamespace as NS

import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import Bool, String
from interfaces_pkg.msg import Detection, DetectionArray, PathPlanningResult

from decision_making_pkg.crosswalk_stop import RedLightStopController, crosswalk_bottom_ratio, crosswalk_edge_ratio
from decision_making_pkg import motion_planner_node as module


def crosswalk(position=0.5, width=640, height=480, edge='far_edge'):
    detection = Detection(class_name='crosswalk', score=0.95)
    detection.mask.width, detection.mask.height = width, height
    detection.bbox.center.position.x = width / 2
    detection.bbox.center.position.y = position * height + (20 if edge == 'far_edge' else -20)
    detection.bbox.size.x, detection.bbox.size.y = width / 2, 40.0
    return detection


def test_nearest_crosswalk_edge_is_normalized_for_image_resolution():
    assert crosswalk_bottom_ratio([crosswalk(0.4, edge='near_edge'), crosswalk(0.6, edge='near_edge')]) == pytest.approx(0.6)
    assert crosswalk_bottom_ratio([crosswalk(0.6, 1280, 960, edge='near_edge')]) == pytest.approx(0.6)
    assert crosswalk_bottom_ratio([Detection(class_name='lane1')]) is None
    assert crosswalk_bottom_ratio([Detection(class_name='crosswalk')]) is None
    broken = crosswalk()
    broken.bbox.center.position.y = float('nan')
    assert crosswalk_bottom_ratio([broken]) is None


def test_stop_line_is_position_based_not_a_fixed_delay():
    policy = RedLightStopController(stop_ratio=0.5)
    policy.observe_light('Red')
    assert policy.evaluate(0.0, 0.4, True, True) == 'approach'
    assert policy.evaluate(0.1, 0.5, True, True) == 'stop'
    policy.observe_light('None')
    assert policy.evaluate(0.2, 0.4, True, True) == 'stop'
    policy.observe_light('Unknown')
    assert policy.evaluate(0.3, 0.4, True, True) == 'stop'
    policy.observe_light('Green')
    assert policy.evaluate(0.4, 0.4, True, True) == 'clear'


@pytest.mark.parametrize('position, fresh, moving', [
    (None, True, True), (0.4, False, True), (0.4, True, False), (0.8, True, True)])
def test_missing_stale_stopped_or_already_near_crosswalk_never_starts_an_approach(position, fresh, moving):
    policy = RedLightStopController()
    policy.observe_light('Red')
    assert policy.evaluate(0.0, position, fresh, moving) == 'stop'


def test_approach_timeout_cannot_be_extended_by_new_red_frames():
    policy = RedLightStopController(approach_timeout=3.0)
    policy.observe_light('Red')
    assert policy.evaluate(10.0, 0.4, True, True) == 'approach'
    policy.observe_light('Red')
    assert policy.evaluate(13.0, 0.4, True, True) == 'stop'
    assert policy.reason == 'crosswalk approach timeout'


@pytest.fixture
def controller(monkeypatch):
    clock = NS(now=100.0)
    monkeypatch.setattr(module, 'time', NS(monotonic=lambda: clock.now))
    rclpy.init()
    node = module.MotionPlanningNode(parameter_overrides=[
        Parameter('enable_obstacle_avoidance', value=False),
        Parameter('crosswalk_stop_y_ratio', value=0.5)])
    node.timer.cancel()
    captured = []
    node.publisher = NS(publish=captured.append)
    node.mode_callback(String(data='autonomous_active'))

    def frame(light='Green', position=0.4, age=0.0, path=True, publish_crosswalk=True):
        clock.now += 0.1
        message = PathPlanningResult(tracking_source='direct' if path else 'lost', target_lane=node.current_lane)
        message.header.stamp = node.get_clock().now().to_msg()
        if path:
            message.x_points, message.y_points = [320.0] * 3, [5.0, 80.0, 179.0]
        node.path_callback(message)
        if publish_crosswalk:
            detections = DetectionArray()
            detections.header.stamp = Time(nanoseconds=node.get_clock().now().nanoseconds - int(age * 1e9)).to_msg()
            if position is not None:
                detections.detections = [crosswalk(position)]
            node.detection_callback(detections)
        node.light_callback(String(data=light))
        node.light_presence_callback(Bool(data=light.casefold() != 'none'))
        node.timer_callback()
        return captured[-1]

    yield node, clock, captured, frame
    node.destroy_node()
    rclpy.try_shutdown()


def test_red_crosswalk_approach_then_stop_then_green(controller):
    node, _, _, frame = controller
    assert frame().left_speed == 255
    assert frame('Red', 0.4).left_speed == 70
    assert frame('Red', 0.48).left_speed == 70
    assert node.state == 'red light approach to crosswalk'
    assert frame('Red', 0.5).left_speed == 0
    assert frame('None', None).left_speed == 0
    assert frame('Unknown', 0.4).left_speed == 0
    assert frame('Red', 0.4).left_speed == 0
    assert frame('Green', None).left_speed == 255


@pytest.mark.parametrize('failure', ['lost_crosswalk', 'old_crosswalk', 'lost_lane', 'stale_receipt', 'malformed_path'])
def test_red_approach_stops_on_bad_inputs_and_does_not_resume_on_red(controller, failure):
    node, clock, captured, frame = controller
    frame()
    assert frame('Red').left_speed == 70
    if failure == 'lost_crosswalk':
        command = frame('Red', None)
    elif failure == 'old_crosswalk':
        command = frame('Red', age=0.5)
    elif failure == 'lost_lane':
        command = frame('Red', path=False)
    elif failure == 'malformed_path':
        node.path = [(320.0, 200.0), (320.0, 300.0)]
        node.timer_callback()
        command = captured[-1]
    else:
        clock.now += 0.4
        command = frame('Red', publish_crosswalk=False)
    assert command.left_speed == 0
    assert frame('Red').left_speed == 0


def test_red_at_startup_or_waiting_never_launches_a_stopped_vehicle(controller):
    node, _, _, frame = controller
    assert frame('Red').left_speed == 0
    node.mode_callback(String(data='autonomous_waiting'))
    assert frame('Red').left_speed == 0


def test_direct_red_stop_can_be_selected_without_changing_lane_speeds(controller):
    node, _, _, frame = controller
    frame()
    node.red_stop_at_crosswalk = False
    assert frame('Red').left_speed == 0
    assert frame('Green').left_speed == 255


def test_red_crosswalk_approach_requires_signal_presence_when_entering(controller):
    node, _, _, frame = controller
    frame()
    node.light_presence_callback = lambda msg: None
    node.signal_present = False
    assert frame('Red').left_speed == 0
    assert node.crosswalk_stop.approach_started is None


def test_red_approach_tolerates_signal_occlusion_but_cannot_release_a_red_stop(controller):
    node, _, _, frame = controller
    frame()
    assert frame('Red', 0.3).left_speed == 70
    assert frame('None', 0.4).left_speed == 70
    assert frame('None', 0.5).left_speed == 0
    assert frame('None', 0.3).left_speed == 0
    assert frame('Green').left_speed == 255


@pytest.mark.parametrize('ratio, line_y', [(0.5, 240), (None, 325)])
def test_camera_stop_line_calibration_does_not_change_yolo_input(monkeypatch, ratio, line_y):
    import numpy as np
    from camera_perception_pkg import image_publisher_node as image_module

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = NS(isOpened=lambda: True, set=lambda *args: True,
             read=lambda: (True, frame), release=lambda: None)
    monkeypatch.setattr(image_module.cv2, 'VideoCapture', lambda _: cap)
    displayed = []
    monkeypatch.setattr(image_module.cv2, 'imshow', lambda title, image: displayed.append(image.copy()))
    monkeypatch.setattr(image_module.cv2, 'waitKey', lambda _: None)
    monkeypatch.setattr(image_module.cv2, 'destroyAllWindows', lambda: None)
    rclpy.init()
    overrides = [Parameter('crosswalk_stop_y_ratio', value=ratio)] if ratio is not None else []
    node = image_module.ImagePublisherNode(parameter_overrides=overrides)
    node.timer.cancel()
    published = []
    node.publisher = NS(publish=published.append)
    try:
        node.timer_callback()
        assert displayed[-1][line_y, 500].tolist() == [0, 0, 255]
        raw = node.bridge.imgmsg_to_cv2(published[-1], desired_encoding='bgr8')
        assert not raw.any()
        assert not frame.any()
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


def test_reference_photo_uses_far_edge_not_the_hood_contaminated_bottom():
    detection = crosswalk()
    detection.bbox.center.position.y = (325.1 + 479.3) / 2
    detection.bbox.size.y = 479.3 - 325.1
    far = crosswalk_edge_ratio([detection], 'far_edge')
    assert far == pytest.approx(325.1 / 480)
    assert crosswalk_bottom_ratio([detection]) > 0.99
    policy = RedLightStopController()
    policy.observe_light('Red')
    assert policy.evaluate(0, far - 0.02, True, True) == 'approach'
    assert policy.evaluate(0.1, far, True, True) == 'stop'
