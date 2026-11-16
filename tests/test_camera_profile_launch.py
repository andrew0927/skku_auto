"""Evaluate launch conditions without starting any process or physical device."""

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node


def configured_launch(**overrides):
    path = Path(__file__).resolve().parents[1] / 'autonomous_driving.launch.py'
    spec = importlib.util.spec_from_file_location('camera_profile_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = module.generate_launch_description().entities
    context = LaunchContext()
    context.launch_configurations.update(overrides)
    for action in actions:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    nodes = {perform_substitutions(context, normalize_to_list_of_substitutions(node.node_executable)): node
             for node in actions if isinstance(node, Node)}
    return context, nodes


def enabled(context, node):
    return node.condition is None or node.condition.evaluate(context)


def test_camera_signal_profile_does_not_start_lidar_or_parking():
    context, nodes = configured_launch()
    for name in ('lidar_publisher_node', 'lidar_processor_node',
                 'lidar_obstacle_detector_node', 'parking_node'):
        assert not enabled(context, nodes[name]), name
    for name in ('image_publisher_node', 'yolov8_node', 'lane_info_extractor_node',
                 'path_planner_node', 'motion_planner_node', 'traffic_light_detector_node'):
        assert enabled(context, nodes[name]), name
    assert context.launch_configurations['enable_green_grayscale_fallback'] == 'true'
    assert context.launch_configurations['obey_traffic_lights'] == 'true'


def test_avoidance_cannot_start_with_lidar_disabled():
    context, nodes = configured_launch(enable_obstacle_avoidance='true', enable_lidar='false')
    assert not enabled(context, nodes['lidar_obstacle_detector_node'])


def test_legacy_lidar_requires_explicit_opt_in():
    context, nodes = configured_launch(enable_lidar='true', enable_obstacle_avoidance='true')
    assert enabled(context, nodes['lidar_publisher_node'])
    assert enabled(context, nodes['lidar_obstacle_detector_node'])
    assert not enabled(context, nodes['parking_node'])
