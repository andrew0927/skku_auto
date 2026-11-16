"""Mission priorities and scan faults without opening any physical device."""

import math
from types import SimpleNamespace as NS

import pytest
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from interfaces_pkg.msg import Detection, DetectionArray, PathPlanningResult

from decision_making_pkg import motion_planner_node as motion_module
from decision_making_pkg import parking_node as parking_module
from decision_making_pkg.parking_policy import ParkingSequence
from lidar_perception_pkg.scan_geometry import scan_is_valid, sector_distance
from lidar_perception_pkg.lidar_obstacle_detector_node import LidarObstacleDetector
from lidar_perception_pkg.lidar_processor_node import LidarSensorDataProcessor


def scan(node, front=4.0, right=1.6, age=0.0):
    msg = LaserScan(angle_min=0.0, angle_max=math.radians(359),
                    angle_increment=math.radians(1), range_min=0.05, range_max=12.0)
    msg.header.stamp = Time(nanoseconds=node.get_clock().now().nanoseconds - int(age * 1e9)).to_msg()
    msg.ranges = [math.inf] * 360
    for i in list(range(345, 360)) + list(range(16)):
        msg.ranges[i] = front
    for i in range(270, 281):
        msg.ranges[i] = right
    return msg


@pytest.fixture
def mission(monkeypatch):
    clock = NS(now=100.0)
    monkeypatch.setattr(motion_module, 'time', NS(monotonic=lambda: clock.now))
    rclpy.init()
    node = motion_module.MotionPlanningNode()
    node.timer.cancel()
    results = []
    node.publisher = NS(publish=results.append)
    node.mode_callback(String(data='autonomous_active'))

    def frame(advance=0.1, light='Green', obstacle=False, refresh_lidar=True,
              crosswalk=False, lane=None, path=True):
        clock.now += advance
        if light is not None:
            node.light_callback(String(data=light))
            node.light_presence_callback(Bool(data=light.casefold() != 'none'))
        if refresh_lidar:
            node.scan_callback(scan(node))
            node.obstacle_callback(Bool(data=obstacle))
        msg = PathPlanningResult(tracking_source='direct' if path else 'lost',
                                 target_lane=lane or node.current_lane)
        msg.header.stamp = node.get_clock().now().to_msg()
        if path:
            msg.x_points, msg.y_points = [470.0, 450.0, 320.0], [5.0, 80.0, 179.0]
        node.path_callback(msg)
        detections = DetectionArray()
        detections.header = msg.header
        if crosswalk:
            detections.detections = [Detection(class_name='crosswalk')]
        node.detection_callback(detections)
        node.timer_callback()
        return results[-1]

    yield node, clock, frame
    node.destroy_node()
    rclpy.try_shutdown()


def test_missions_are_enabled_and_normal_speed_is_unchanged(mission):
    node, _, frame = mission
    assert node.obey_lights and node.crosswalk_guard and node.avoidance_enabled
    assert frame().left_speed == 255
    assert node.estimated_speed == 200


def test_red_has_priority_over_avoidance_and_lane_search(mission):
    node, _, frame = mission
    frame()
    stopped = frame(light='Red', obstacle=True)
    assert stopped.left_speed == stopped.right_speed == 0
    assert node.current_lane == 'lane2' and node.swerve_until == 0
    assert frame(light='Red', path=False).left_speed == 0


def test_stale_light_and_lidar_stop_even_with_a_valid_path(mission):
    node, _, frame = mission
    frame()
    assert frame(advance=1.1, light=None).left_speed == 0
    assert node.state == 'traffic light input stale or missing'
    assert frame().left_speed == 255
    assert frame(advance=0.7, refresh_lidar=False).left_speed == 0
    assert node.state == 'lidar input stale or missing'


@pytest.mark.parametrize('blocker', ['red', 'lidar_stale', 'light_stale', 'obstacle'])
def test_brief_path_gap_never_bypasses_mission_or_sensor_stops(mission, blocker):
    node, clock, frame = mission
    frame()
    clock.now += 0.7 if blocker != 'light_stale' else 1.1
    if blocker != 'light_stale':
        node.light_callback(String(data='Red' if blocker == 'red' else 'Green'))
    if blocker != 'lidar_stale':
        node.scan_callback(scan(node))
        node.obstacle_callback(Bool(data=blocker == 'obstacle'))
    node.timer_callback()
    assert node.last_speed == 0
    assert node.state != 'brief path gap (low-speed hold)'


def test_brief_path_gap_requires_fresh_clear_mission_inputs(mission):
    node, clock, frame = mission
    frame()
    clock.now += 0.7
    node.light_callback(String(data='Green'))
    node.scan_callback(scan(node))
    node.obstacle_callback(Bool(data=False))
    node.timer_callback()
    assert node.last_speed == 70
    assert node.state == 'brief path gap (low-speed hold)'


def test_crosswalk_tightens_steering_without_changing_speed(mission):
    node, clock, frame = mission
    frame()
    normal_change = node.previous_steering
    node.previous_steering = 0.0
    node.last_tick = clock.now
    result = frame(crosswalk=True)
    assert node.previous_steering < normal_change
    assert node.previous_steering <= 0.51
    assert result.left_speed == 255
    assert 'crosswalk steering guard' in node.state


def test_crosswalk_without_a_detected_signal_has_no_special_steering_or_speed(mission):
    node, clock, frame = mission
    frame(light='None')
    normal = node.previous_steering
    node.previous_steering = 0.0
    node.last_tick = clock.now
    result = frame(light='None', crosswalk=True)
    assert node.previous_steering == pytest.approx(normal)
    assert result.left_speed == 255
    assert 'crosswalk steering guard' not in node.state


def test_old_signal_presence_cannot_enable_crosswalk_guard(mission):
    node, clock, frame = mission
    frame(crosswalk=True)
    assert node.has_signal_context(clock.now)
    clock.now += 1.1
    node.light_callback(String(data='Green'))
    assert not node.has_signal_context(clock.now)
    assert not node.crosswalk_is_active(clock.now, node.get_clock().now().nanoseconds / 1e9)


@pytest.mark.parametrize('initial, target, steering', [('lane2', 'lane1', -7), ('lane1', 'lane2', 7)])
def test_avoidance_changes_lane_in_both_directions(mission, initial, target, steering):
    node, _, frame = mission
    node.current_lane = initial
    frame()
    command = frame(obstacle=True)
    assert node.current_lane == target
    assert command.steering == steering and command.left_speed == 80
    assert frame(advance=1.0, obstacle=True).left_speed == 80
    stopped = frame(advance=3.1, obstacle=True)
    assert stopped.left_speed == 0
    assert node.current_lane == target, 'A persistent object must not repeatedly flip target lanes'
    assert frame(advance=0.5, obstacle=False).left_speed > 0


def test_old_lane_path_cannot_be_followed_after_target_change(mission):
    node, _, frame = mission
    frame()
    node.current_lane = 'lane1'
    assert frame(lane='lane2').left_speed == 0
    assert node.state == 'waiting for the selected lane'
    assert frame(lane='lane1').left_speed == 255


def test_scan_metadata_wrap_and_missing_measurements():
    msg = LaserScan(angle_min=-math.pi, angle_increment=math.radians(0.5),
                    range_min=0.05, range_max=12.0)
    msg.ranges = [math.inf] * 720
    msg.ranges[350] = 0.7  # -5 degrees, not array index 355.
    assert scan_is_valid(msg)
    assert sector_distance(msg, 345, 15) == pytest.approx(0.7)
    assert sector_distance(msg, 270, 280) == math.inf
    msg.ranges = [0.0] * 720
    assert not scan_is_valid(msg)
    assert sector_distance(msg, 345, 15) is None


def test_processor_and_detector_keep_time_and_confirm_two_distinct_scans():
    rclpy.init()
    processor, detector = LidarSensorDataProcessor(), LidarObstacleDetector()
    processed, detected = [], []
    processor.publisher = NS(publish=processed.append)
    detector.publisher = NS(publish=detected.append)
    try:
        first = scan(processor, front=0.5)
        processor.lidar_raw_cb(first)
        assert processed[-1].header == first.header
        detector.lidar_callback(processed[-1])
        assert detected[-1].data is False
        detector.lidar_callback(processed[-1])
        assert len(detected) == 1, 'A repeated scan cannot confirm an obstacle'
        processor.lidar_raw_cb(scan(processor, front=0.5))
        detector.lidar_callback(processed[-1])
        assert detected[-1].data is True
        detector.lidar_callback(scan(detector, age=2.0))
        assert len(detected) == 2, 'An old scan must not refresh the obstacle heartbeat'
    finally:
        processor.destroy_node()
        detector.destroy_node()
        rclpy.try_shutdown()


def test_lidar_publisher_does_not_overwrite_scan_frame_when_creating_tf(monkeypatch):
    from lidar_perception_pkg import lidar_publisher_node as module

    actions = []
    fake_driver = NS(
        iter_scans=lambda **kwargs: iter([[(20, 0.0, 700), (10, 0.4, 900), (5, 90.0, 2000)]]),
        stop=lambda: actions.append('stop'), stop_motor=lambda: actions.append('motor'),
        disconnect=lambda: actions.append('disconnect'))
    monkeypatch.setattr(module.LPFL, 'RPLidar', lambda *args, **kwargs: fake_driver)
    rclpy.init()
    node = module.LidarSensorDataPublisher()
    node.timer.cancel()
    published, transforms = [], []
    node.publisher_ = NS(publish=published.append)
    node.tf_broadcaster = NS(sendTransform=transforms.append)
    try:
        node.publish_lidar_data()
        assert published[-1].header.frame_id == 'laser_frame'
        assert transforms[-1].header.frame_id == 'base_link'
        assert transforms[-1].child_frame_id == 'laser_frame'
        assert transforms[-1].transform.rotation.w == 1.0
        assert published[-1].ranges[0] == pytest.approx(0.7)
        assert published[-1].ranges[90] == pytest.approx(2.0)
        assert published[-1].angle_max == pytest.approx(math.radians(359))
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
    assert actions == ['stop', 'motor', 'disconnect']


def test_complete_backup_parking_sequence_has_stops_reverse_and_exit():
    sequence = ParkingSequence()
    sequence.start(0.0)
    sequence.tick(0.1, 1.6)
    sequence.tick(0.2, 1.6)
    assert sequence.saw_first_car
    sequence.tick(0.3, math.inf)
    sequence.tick(0.4, math.inf)
    assert sequence.state == 'GAP_FOUND'
    sequence.tick(2.3, 1.6)
    sequence.tick(2.4, 1.6)
    assert sequence.state == 'OVERSHOOT'
    assert sequence.tick(5.5, 1.6).left_speed == 0
    sequence.tick(6.1, 1.6)
    reversing = sequence.tick(6.2, 1.6)
    assert reversing.steering == 7 and reversing.left_speed == -150
    assert sequence.tick(12.7, 1.6).left_speed == 0
    sequence.tick(16.3, 1.6)
    exiting = sequence.tick(16.4, 1.6)
    assert exiting.steering == 7 and exiting.left_speed == 150
    assert sequence.tick(23.0, 1.6).steering == 0
    assert sequence.tick(38.1, 1.6).left_speed == 0
    assert sequence.state == 'DONE'


def test_parking_counts_new_scans_not_timer_ticks():
    sequence = ParkingSequence()
    sequence.start(0)
    sequence.tick(0.1, 1.0)
    for i in range(20):
        sequence.tick(0.2 + i / 10, 1.0, new_scan=False)
    assert not sequence.saw_first_car


def test_parking_requires_mode_and_stale_scan_fault_needs_restart(monkeypatch):
    clock = NS(now=100.0)
    monkeypatch.setattr(parking_module, 'time', NS(monotonic=lambda: clock.now))
    rclpy.init()
    node = parking_module.ParkingNode()
    node.timer.cancel()
    results = []
    node.publisher = NS(publish=results.append)
    try:
        node.scan_callback(scan(node))
        node.timer_callback()
        assert results[-1].left_speed == 0
        node.mode_callback(String(data='parking_active'))
        node.timer_callback()
        assert results[-1].left_speed == 150
        clock.now += 0.7
        node.timer_callback()
        assert node.faulted and results[-1].left_speed == 0
        node.scan_callback(scan(node))
        node.timer_callback()
        assert results[-1].left_speed == 0
        node.mode_callback(String(data='parking_waiting'))
        node.mode_callback(String(data='parking_active'))
        node.timer_callback()
        assert results[-1].left_speed == 150
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
