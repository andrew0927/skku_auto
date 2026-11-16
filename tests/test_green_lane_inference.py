"""Green-light recovery is conditional, preserves color evidence and limits work."""

from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from camera_perception_pkg import lane_inference as module


class FakeResult:
    def __init__(self, *classes):
        self.boxes = [NS(cls=index, conf=0.95) for index in classes]

    def cpu(self):
        return self


class FakeModel:
    names = {0: 'Trafficlight', 1: 'crosswalk', 2: 'lane1', 3: 'lane2'}

    def __init__(self, *results):
        self.results, self.inputs = list(results), []

    def predict(self, **kwargs):
        self.inputs.append(kwargs['source'].copy())
        return [self.results.pop(0)]


def color_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (70, 190, 40)
    return frame


@pytest.mark.parametrize('lamp', [None, {'color': 'Red'}, {'color': 'Yellow'}])
def test_no_grayscale_recovery_without_a_green_lamp(monkeypatch, lamp):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: lamp)
    model = FakeModel(FakeResult())
    groups, debug = module.GreenLaneInference(model).predict(color_frame())
    assert len(groups) == len(model.inputs) == 1
    assert debug['mode'] == 'color'


def test_normal_lane_detection_does_not_change_input_or_add_inference(monkeypatch):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: None)
    model = FakeModel(FakeResult(3))
    frame = color_frame()
    _, debug = module.GreenLaneInference(model).predict(frame)
    assert debug['mode'] == 'color' and len(model.inputs) == 1
    assert np.array_equal(model.inputs[0], frame)


def test_first_recovery_uses_two_passes_then_one_and_red_restores_color(monkeypatch):
    lamp = NS(color='Green')
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color=lamp.color))
    model = FakeModel(FakeResult(), FakeResult(2, 3), FakeResult(2, 3), FakeResult(3))
    recovery = module.GreenLaneInference(model)
    frame, original = color_frame(), color_frame()
    groups, debug = recovery.predict(frame)
    assert len(groups) == 2 and debug['forward_passes'] == 2
    assert debug['mode'] == 'green_grayscale' and recovery.prefer_gray
    assert np.array_equal(model.inputs[1][:, :, 0], model.inputs[1][:, :, 1])
    assert np.array_equal(frame, original)
    _, debug = recovery.predict(frame)
    assert debug['mode'] == 'green_grayscale_cached' and debug['forward_passes'] == 1
    lamp.color = 'Red'
    _, debug = recovery.predict(frame)
    assert debug['mode'] == 'color' and not recovery.prefer_gray
    assert np.array_equal(model.inputs[-1], original)


def test_failed_recovery_never_fabricates_a_lane(monkeypatch):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color='Green'))
    model = FakeModel(FakeResult(), FakeResult())
    recovery = module.GreenLaneInference(model)
    groups, debug = recovery.predict(color_frame())
    assert len(groups) == 1 and not groups[0].boxes
    assert debug['mode'] == 'green_grayscale_failed' and not recovery.prefer_gray


def test_recovery_can_be_disabled(monkeypatch):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: pytest.fail('Disabled'))
    model = FakeModel(FakeResult())
    _, debug = module.GreenLaneInference(model, enabled=False).predict(color_frame())
    assert debug['forward_passes'] == 1


def test_merge_keeps_original_signals_and_crosswalks_and_all_new_lane_parts():
    light, crossing = NS(class_name='Trafficlight'), NS(class_name='crosswalk')
    lane1, lane2, lane2_part = (NS(class_name=name) for name in ('lane1', 'lane2', 'lane2'))
    merged = module.merge_detection_groups([
        [light, crossing], [lane1, lane2, lane2_part, NS(class_name='Trafficlight'), NS(class_name='crosswalk')]])
    assert merged == [light, crossing, lane1, lane2, lane2_part]


class ColorAwareModel(FakeModel):
    def __init__(self, color=(), gray=(2, 3)):
        super().__init__()
        self.color, self.gray = color, gray

    def predict(self, **kwargs):
        frame = kwargs['source']
        self.inputs.append(frame.copy())
        grayscale = np.array_equal(frame[:, :, 0], frame[:, :, 1])
        return [FakeResult(*(self.gray if grayscale else self.color))]


@pytest.mark.parametrize('target,other,target_id', [('lane1', 3, 2), ('lane2', 2, 3)])
def test_missing_target_is_recovered_even_if_other_lane_is_visible(monkeypatch, target, other, target_id):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color='Green'))
    model = ColorAwareModel(color=(other,), gray=(target_id,))
    inference = module.GreenLaneInference(model, target_lane=target)
    groups, debug = inference.predict(color_frame(), now=0.0)
    assert len(groups) == debug['forward_passes'] == 2
    assert inference.prefer_gray and debug['target_found'] and debug['target_lane'] == target
    assert debug['output_lanes'] == 2


@pytest.mark.parametrize('target,target_id', [('lane1', 2), ('lane2', 3)])
def test_healthy_target_never_needs_grayscale_just_because_other_lane_is_missing(monkeypatch, target, target_id):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color='Green'))
    model = ColorAwareModel(color=(target_id,))
    _, debug = module.GreenLaneInference(model, target_lane=target).predict(color_frame(), now=0.0)
    assert debug['forward_passes'] == 1 and debug['target_found']


def test_lamp_flicker_keeps_fresh_grayscale_inference_even_after_green_context_expires(monkeypatch):
    lamp = NS(color='Green')
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color=lamp.color) if lamp.color else None)
    model = ColorAwareModel()
    inference = module.GreenLaneInference(model)
    inference.predict(color_frame(), now=0.0)
    lamp.color = None
    for now in (0.1, 0.2, 0.41, 0.7, 0.82, 1.1, 1.23):
        _, debug = inference.predict(color_frame(), now=now)
        assert inference.prefer_gray and debug['target_found']
    assert not debug['green_context']
    assert len(model.inputs) < 2 * 8, 'Steady grayscale frames must not all do two forward passes'


def test_color_exit_needs_three_distinct_successful_probes_after_green_is_gone(monkeypatch):
    lamp = NS(color='Green')
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color=lamp.color) if lamp.color else None)
    model = ColorAwareModel()
    inference = module.GreenLaneInference(model)
    inference.predict(color_frame(), now=0.0, frame_stamp=1.0)
    model.color, lamp.color = (3,), None
    for index, now in enumerate((0.41, 0.82, 1.23), start=1):
        groups, debug = inference.predict(color_frame(), now=now, frame_stamp=1.0 + now)
        assert debug['color_confirmations'] == index
        assert inference.prefer_gray == (index < 3)
        if index < 3:
            assert debug['prefer_recovered_lanes'] and len(groups) == 2
    assert debug['mode'] == 'color' and len(groups) == 1


def test_repeated_source_stamp_cannot_confirm_color_recovery(monkeypatch):
    lamp = NS(color='Green')
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color=lamp.color) if lamp.color else None)
    model = ColorAwareModel()
    inference = module.GreenLaneInference(model)
    inference.predict(color_frame(), now=0.0)
    model.color, lamp.color = (3,), None
    for now in (0.41, 0.82, 1.23, 1.64):
        _, debug = inference.predict(color_frame(), now=now, frame_stamp=5.0)
    assert inference.prefer_gray and debug['color_confirmations'] == 1


@pytest.mark.parametrize('color', ['Red', 'Yellow'])
@pytest.mark.parametrize('source', ['geometry', 'hint'])
def test_red_and_yellow_immediately_end_grayscale_even_with_old_green_context(monkeypatch, color, source):
    lamp = NS(color='Green')
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color=lamp.color) if lamp.color else None)
    model = ColorAwareModel()
    inference = module.GreenLaneInference(model)
    inference.predict(color_frame(), now=0.0)
    lamp.color = color if source == 'geometry' else None
    _, debug = inference.predict(color_frame(), now=0.1, signal_color=color if source == 'hint' else None)
    assert not inference.prefer_gray and debug['mode'] == 'color'
    assert debug['forward_passes'] == 1 and not debug['green_context']


def test_recent_green_hint_can_trigger_recovery_but_expired_context_cannot(monkeypatch):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: None)
    model = ColorAwareModel(color=(3,))
    inference = module.GreenLaneInference(model)
    inference.predict(color_frame(), now=0.0, signal_color='Green')
    inference.predict(color_frame(), now=0.5)
    model.color = ()
    _, debug = inference.predict(color_frame(), now=1.1)
    assert not debug['green_context'] and debug['forward_passes'] == 1
    _, debug = inference.predict(color_frame(), now=1.2, signal_color='Green')
    assert debug['target_found'] and inference.prefer_gray


def test_target_switch_discards_color_confirmation_from_the_old_lane(monkeypatch):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color='Green'))
    model = ColorAwareModel()
    inference = module.GreenLaneInference(model)
    inference.predict(color_frame(), now=0.0)
    model.color = (3,)
    inference.predict(color_frame(), now=0.41)
    assert inference.color_streak == 1
    inference.set_target_lane('lane1')
    assert inference.color_streak == 0
    _, debug = inference.predict(color_frame(), now=0.5)
    assert debug['target_lane'] == 'lane1' and debug['target_found']
    assert debug['color_confirmations'] == 0 and inference.prefer_gray


def test_gray_failure_uses_a_fresh_color_result_and_confirms_it_before_switching(monkeypatch):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color='Green'))
    model = ColorAwareModel()
    inference = module.GreenLaneInference(model)
    inference.predict(color_frame(), now=0.0)
    model.color, model.gray = (3,), ()
    for index, now in enumerate((0.1, 0.2, 0.3), start=1):
        groups, debug = inference.predict(color_frame(), now=now, frame_stamp=1 + now)
        assert debug['target_found'] and not debug['prefer_recovered_lanes']
        assert inference.prefer_gray == (index < 3)
    assert debug['mode'] == 'color'


def test_failed_fallback_and_both_missing_cache_are_rate_limited(monkeypatch):
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color='Green'))
    model = ColorAwareModel(color=(2,), gray=())
    inference = module.GreenLaneInference(model)
    _, first = inference.predict(color_frame(), now=0.0)
    _, waiting = inference.predict(color_frame(), now=0.1)
    _, retry = inference.predict(color_frame(), now=0.41)
    assert first['forward_passes'] == retry['forward_passes'] == 2
    assert waiting['forward_passes'] == 1 and not waiting['target_found']
    assert not inference.prefer_gray
    model.gray = (3,)
    inference.predict(color_frame(), now=0.82)
    model.gray, model.color = (), ()
    inference.predict(color_frame(), now=0.9)
    _, debug = inference.predict(color_frame(), now=1.0)
    assert debug['forward_passes'] == 1 and not debug['target_found']


@pytest.mark.parametrize('now', [2.0, -1.0])
def test_frame_gap_or_clock_reset_drops_cached_signal_context(monkeypatch, now):
    lamp = NS(color='Green')
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: dict(color=lamp.color) if lamp.color else None)
    model = ColorAwareModel()
    inference = module.GreenLaneInference(model)
    inference.predict(color_frame(), now=0.0)
    lamp.color = None
    _, debug = inference.predict(color_frame(), now=now)
    assert not inference.prefer_gray and not debug['green_context']
    assert debug['forward_passes'] == 1


def test_grayscale_lane_preference_does_not_replace_colored_signals_or_crosswalks():
    signal, crossing = NS(class_name='Trafficlight'), NS(class_name='crosswalk')
    color_lane, gray_lane, gray_part = [NS(class_name='lane2') for _ in range(3)]
    merged = module.merge_detection_groups([
        [signal, crossing, color_lane],
        [gray_lane, gray_part, NS(class_name='Trafficlight'), NS(class_name='crosswalk')]],
        prefer_recovered_lanes=True)
    assert merged == [signal, crossing, gray_lane, gray_part]


def test_ros_node_updates_target_lane_and_does_not_use_stale_signal_hints(monkeypatch):
    import rclpy
    from cv_bridge import CvBridge
    from interfaces_pkg.msg import Detection
    from std_msgs.msg import String
    from camera_perception_pkg import yolov8_node as yolo_module

    clock = NS(now=10.0)
    monkeypatch.setattr(yolo_module, 'time', NS(monotonic=lambda: clock.now))
    monkeypatch.setattr(module, 'find_signal_lamp', lambda _: None)
    model = ColorAwareModel(color=(2,))
    rclpy.init()
    node = yolo_module.Yolov8Node()
    node.inference, node.enable = module.GreenLaneInference(model), True
    node.cv_bridge = CvBridge()
    node._pub = NS(publish=lambda msg: None)
    diagnostics = []
    node.debug_pub = NS(publish=diagnostics.append)
    node.detections_from_result = lambda result: [
        Detection(class_name=model.names[int(box.cls)], score=float(box.conf)) for box in result.boxes]
    try:
        frame = node.cv_bridge.cv2_to_imgmsg(color_frame(), encoding='bgr8')
        node.signal_callback(String(data='Green'))
        node.target_callback(String(data='lane1'))
        node.image_cb(frame)
        assert node.target_lane == node.inference.target_lane == 'lane1'
        assert not node.inference.prefer_gray
        node.target_callback(String(data='lane2'))
        clock.now += 0.1
        node.image_cb(frame)
        assert node.inference.prefer_gray
        node.inference.reset()
        clock.now += 2.0
        node.image_cb(frame)
        assert not node.inference.prefer_gray
        assert '"green_context": false' in diagnostics[-1].data
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


GREEN = Path('/home/donghyun/Pictures/Webcam/2026-11-08-173052.jpg')
RED = Path('/home/donghyun/Camera Image_screenshot_08.11.2026.png')
MODEL = Path(__file__).resolve().parents[1] / 'best.pt'


@pytest.mark.skipif(not all(p.exists() for p in (GREEN, RED, MODEL)), reason='Reference photos/model unavailable')
def test_real_green_photo_recovers_through_ros_message_parser_without_mutating_signal_colors():
    import rclpy
    import torch
    from cv_bridge import CvBridge
    from ultralytics import YOLO
    from camera_perception_pkg.yolov8_node import Yolov8Node
    from camera_perception_pkg.lane_geometry import LaneCenterTracker
    from camera_perception_pkg.signal_geometry import find_signal_lamp

    torch.set_num_threads(4)
    rclpy.init()
    node = Yolov8Node()
    node.yolo = YOLO(str(MODEL))
    node.threshold, node.enable = 0.5, True
    node.cv_bridge = CvBridge()
    node.inference = module.GreenLaneInference(node.yolo)
    messages, debug = [], []
    node._pub = NS(publish=messages.append)
    node.debug_pub = NS(publish=debug.append)
    try:
        red = cv2.resize(cv2.imread(str(RED)), (640, 480))
        msg = node.cv_bridge.cv2_to_imgmsg(red, encoding='bgr8')
        msg.header.stamp = node.get_clock().now().to_msg()
        node.image_cb(msg)
        assert not node.inference.prefer_gray
        tracker = LaneCenterTracker()
        assert tracker.update(messages[-1].detections, 'lane2', 0)[1] == 'direct'
        green = cv2.resize(cv2.imread(str(GREEN)), (640, 480))
        original = green.copy()
        msg = node.cv_bridge.cv2_to_imgmsg(green, encoding='bgr8')
        msg.header.stamp = node.get_clock().now().to_msg()
        node.image_cb(msg)
        output = messages[-1]
        scores = {d.class_name: d.score for d in output.detections}
        assert scores['lane1'] > 0.9 and scores['lane2'] > 0.9 and scores['crosswalk'] > 0.9
        assert output.header == msg.header
        assert tracker.update(output.detections, 'lane2', 0.1)[1] in ('direct', 'estimated')
        assert np.array_equal(green, original)
        assert np.array_equal(node.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), original)
        assert find_signal_lamp(green)['color'] == 'Green'
        node.image_cb(msg)
        assert 'green_grayscale_cached' in debug[-1].data
        msg = node.cv_bridge.cv2_to_imgmsg(red, encoding='bgr8')
        msg.header.stamp = node.get_clock().now().to_msg()
        node.image_cb(msg)
        assert not node.inference.prefer_gray
        assert '"mode": "color"' in debug[-1].data
        assert any(d.class_name == 'lane2' and d.score > 0.9 for d in messages[-1].detections)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_offline_report_uses_the_same_fixed_input_size_as_live_camera(monkeypatch):
    from tools import analyze_frame as analysis

    model = FakeModel(FakeResult(), FakeResult(), FakeResult())
    monkeypatch.setattr(analysis, 'YOLO', lambda _: model)
    monkeypatch.setattr(analysis.cv2, 'imread', lambda _: np.zeros((1080, 1920, 3), dtype=np.uint8))
    monkeypatch.setattr(analysis.torch, 'set_num_threads', lambda _: None)
    report = analysis.analyze(Path('unused.jpg'), Path('unused.pt'))
    assert report['original_shape'] == [1080, 1920, 3]
    assert report['processing_shape'] == [480, 640, 3]
    assert all(image.shape == (480, 640, 3) for image in model.inputs)
