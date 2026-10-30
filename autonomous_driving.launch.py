"""Camera -> YOLO -> lane -> path -> motion -> keyboard gate -> Arduino."""

import os
from pathlib import Path
import shlex

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ws = Path(__file__).resolve().parent
    defaults = {
        'cam_num': (os.environ.get('CAMERA_INDEX', '0'), 'Camera device index'),
        'device': (os.environ.get('YOLO_DEVICE', 'auto'), 'auto, cpu, or cuda:0'),
        'model': (os.environ.get('YOLO_MODEL', str(ws / 'best.pt')), 'YOLO segmentation model'),
        'port': (os.environ.get('ARDUINO_PORT', '/dev/ttyACM0'), 'Arduino serial port'),
        'cruise_speed': (os.environ.get('DRIVING_SPEED_LIMIT', '255'), 'Forward motor PWM, 0..255'),
        'estimated_speed': ('200', 'Speed when lane2 is estimated from lane1'),
        'lane_loss_steering': ('4', 'Right steering on brief lane loss, 0..7'),
        'lane_loss_hold': ('0.4', 'Maximum seconds of recovery on lane loss'),
        'reacquire_hold': ('0.5', 'Seconds at reduced speed after estimated lane tracking'),
        'data_source': ('camera', 'camera, video, or image'),
        'img_dir': (str(ws / 'src/camera_perception_pkg/camera_perception_pkg/lib/Collected_Datasets/sample_dataset'), 'Image input directory'),
        'video_path': (str(ws / 'src/camera_perception_pkg/camera_perception_pkg/lib/Collected_Datasets/driving_simulation.mp4'), 'Video input'),
        'input_period': ('0.1', 'Seconds between camera frames (0.1 = 10 Hz)'),
        'show_image': ('true', 'Show camera, both lane masks, and target centers'),
        'enable_serial': ('true', 'Enable physical Arduino connection'),
        'enable_teleop': ('true', 'Open keyboard mode selection terminal'),
        'target_lane': ('lane2', 'Preferred lane: lane1 or lane2'),
        'steering_gain': ('0.12', 'Steering steps per degree of image heading error'),
        'steering_offset': ('0.0', 'Logical steering offset; Arduino calibration is unchanged'),
        'steering_sign': ('1.0', '1 follows the tested firmware convention; -1 reverses commands'),
        'obey_traffic_lights': ('false', 'Enable traffic-light stop logic'),
    }
    arguments = [DeclareLaunchArgument(name, default_value=value, description=description)
                 for name, (value, description) in defaults.items()]
    def param(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)
    def stop_on_exit(name):
        return [Shutdown(reason=f'{name} exited')]
    domain = os.environ.get('ROS_DOMAIN_ID', '10')
    keyboard_cmd = (
        'source /opt/ros/humble/setup.bash && '
        f'source {shlex.quote(str(ws / "install/local_setup.bash"))} && '
        'exec ros2 run decision_making_pkg teleop_keyboard_node'
    )
    actions = [SetEnvironmentVariable('ROS_DOMAIN_ID', domain)] + arguments
    actions += [
        Node(package='camera_perception_pkg', executable='image_publisher_node',
             output='screen', parameters=[{
                 'cam_num': param('cam_num', int), 'data_source': param('data_source', str),
                 'img_dir': param('img_dir', str), 'video_path': param('video_path', str),
                 'timer': param('input_period', float), 'logger': param('show_image', bool),
             }], on_exit=stop_on_exit('camera')),
        Node(package='camera_perception_pkg', executable='yolov8_node',
             output='screen', parameters=[{'model': param('model', str),
                                            'device': param('device', str)}],
             on_exit=stop_on_exit('YOLO')),
        Node(package='camera_perception_pkg', executable='lane_info_extractor_node',
             output='screen', parameters=[{'show_image': param('show_image', bool),
                                            'target_lane': param('target_lane', str)}],
             on_exit=stop_on_exit('lane extractor')),
        Node(package='decision_making_pkg', executable='path_planner_node',
             output='screen', on_exit=stop_on_exit('path planner')),
        Node(package='decision_making_pkg', executable='motion_planner_node',
             output='screen', parameters=[{
                 'driving_speed': param('cruise_speed', int),
                 'estimated_speed': param('estimated_speed', int),
                 'lane_loss_steering': param('lane_loss_steering', int),
                 'lane_loss_hold': param('lane_loss_hold', float),
                 'reacquire_hold': param('reacquire_hold', float),
                 'steering_gain': param('steering_gain', float),
                 'steering_offset': param('steering_offset', float),
                 'steering_sign': param('steering_sign', float),
                 'obey_traffic_lights': param('obey_traffic_lights', bool),
             }], on_exit=stop_on_exit('motion planner')),
        Node(package='camera_perception_pkg', executable='traffic_light_detector_node',
             output='screen', condition=IfCondition(LaunchConfiguration('obey_traffic_lights')),
             on_exit=stop_on_exit('traffic light detector')),
        Node(package='serial_communication_pkg', executable='serial_sender_node',
             output='screen', parameters=[{'port': param('port', str)}],
             condition=IfCondition(LaunchConfiguration('enable_serial')),
             on_exit=stop_on_exit('serial sender')),
        ExecuteProcess(cmd=['gnome-terminal', '--wait', '--', 'bash', '-c', keyboard_cmd],
                       output='screen', condition=IfCondition(LaunchConfiguration('enable_teleop')),
                       on_exit=stop_on_exit('keyboard terminal')),
    ]
    return LaunchDescription(actions)
