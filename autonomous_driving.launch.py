"""Camera/lidar missions with one keyboard command gate and explicit launch switches."""

import os
from pathlib import Path
import shlex

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ws = Path(__file__).resolve().parent
    defaults = {
        'cam_num': (os.environ.get('CAMERA_INDEX', '0'), 'Camera device index'),
        'device': (os.environ.get('YOLO_DEVICE', 'auto'), 'auto, cpu, or cuda:0'),
        'model': (os.environ.get('YOLO_MODEL', str(ws / 'best.pt')), 'YOLO segmentation model'),
        'enable_green_grayscale_fallback': ('true', 'Recover missed lanes with grayscale input under a green signal'),
        'green_context_hold': ('1.0', 'Seconds to retain a recent green observation for lane recovery'),
        'green_color_probe_interval': ('0.4', 'Seconds between color probes while tracking grayscale lanes'),
        'green_color_confirmations': ('3', 'Successful color probes required before leaving grayscale'),
        'port': (os.environ.get('ARDUINO_PORT', '/dev/ttyACM0'), 'Arduino serial port'),
        'lidar_port': (os.environ.get('LIDAR_PORT', '/dev/ttyUSB0'), 'RPLidar serial port'),
        'lidar_offset': ('0.0', 'Mounted lidar rotation in degrees'),
        'lidar_flip': ('true', 'Mirror scan angles, as in the backup'),
        'cruise_speed': (os.environ.get('DRIVING_SPEED_LIMIT', '255'), 'Normal forward PWM'),
        'estimated_speed': ('200', 'PWM when estimating the target from the other lane'),
        'lane_loss_hold': ('3.0', 'Maximum low-speed lane search time'),
        'recovery_speed': ('70', 'PWM during lane search'),
        'recovery_steering': ('2', 'Search steering magnitude in -7..7 units'),
        'reacquire_hold': ('0.5', 'Reduced-speed hold after estimated tracking'),
        'path_gap_grace': ('0.3', 'Extra low-speed grace after a brief path message gap'),
        'data_source': ('camera', 'camera, video, or image'),
        'img_dir': (str(ws / 'src/camera_perception_pkg/camera_perception_pkg/lib/Collected_Datasets/sample_dataset'), 'Image input directory'),
        'video_path': (str(ws / 'src/camera_perception_pkg/camera_perception_pkg/lib/Collected_Datasets/driving_simulation.mp4'), 'Video input'),
        'input_period': ('0.1', 'Seconds between camera frames'),
        'show_image': ('true', 'Show camera and target lane masks'),
        'enable_serial': ('true', 'Enable physical Arduino connection'),
        'enable_teleop': ('true', 'Open keyboard mode selection terminal'),
        'enable_lidar': ('false', 'Optional legacy lidar; off in the camera/signal profile'),
        'enable_obstacle_avoidance': ('false', 'Optional legacy lidar avoidance; off by default'),
        'enable_parking': ('false', 'Optional legacy parking; off without a rear camera'),
        'enable_crosswalk_guard': ('true', 'Limit steering changes at crosswalks; speed unchanged'),
        'crosswalk_steering_rate': ('5.0', 'Maximum steering steps per second at a crosswalk'),
        'red_stop_at_crosswalk': ('true', 'Approach a visible crosswalk slowly before stopping on red'),
        'crosswalk_stop_y_ratio': ('0.6770833333333334', 'Reference-image crosswalk stop line, measured from image top'),
        'crosswalk_stop_edge': ('far_edge', 'far_edge uses the top of the crosswalk box; near_edge uses its bottom'),
        'crosswalk_approach_speed': ('70', 'Maximum PWM when approaching a red-light stop line'),
        'crosswalk_approach_timeout': ('3.0', 'Maximum seconds of low-speed red-light approach'),
        'crosswalk_position_timeout': ('0.3', 'Maximum age of the crosswalk position'),
        'target_lane': ('lane2', 'Initial target lane: lane1 or lane2'),
        'steering_gain': ('0.12', 'Steering steps per heading degree'),
        'steering_offset': ('0.0', 'Logical steering offset'),
        'steering_sign': ('1.0', '1 follows firmware convention; -1 reverses commands'),
        'obey_traffic_lights': ('true', 'Stop on red; stop if signal input is stale'),
        'swerve_speed': ('80', 'PWM during obstacle avoidance'),
        'swerve_duration': ('4.0', 'Seconds of obstacle avoidance'),
        'swerve_cooldown': ('4.5', 'Minimum seconds between avoidance starts'),
        'parking_speed': ('150', 'PWM during automatic parking'),
    }
    arguments = [DeclareLaunchArgument(name, default_value=value, description=description)
                 for name, (value, description) in defaults.items()]

    def param(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)

    def stop_on_exit(name):
        return [Shutdown(reason=f'{name} exited')]

    avoidance = PythonExpression([
        "'", LaunchConfiguration('enable_lidar'), "'.lower() == 'true' and '",
        LaunchConfiguration('enable_obstacle_avoidance'), "'.lower() == 'true'"])
    keyboard_cmd = (
        'source /opt/ros/humble/setup.bash && '
        f'source {shlex.quote(str(ws / "install/local_setup.bash"))} && '
        'exec ros2 run decision_making_pkg teleop_keyboard_node'
    )
    actions = [SetEnvironmentVariable('ROS_DOMAIN_ID', os.environ.get('ROS_DOMAIN_ID', '10'))] + arguments
    actions += [
        Node(package='camera_perception_pkg', executable='image_publisher_node',
             output='screen', parameters=[{
                 'cam_num': param('cam_num', int), 'data_source': param('data_source', str),
                 'img_dir': param('img_dir', str), 'video_path': param('video_path', str),
                 'timer': param('input_period', float), 'logger': param('show_image', bool),
                 'crosswalk_stop_y_ratio': param('crosswalk_stop_y_ratio', float),
                 'crosswalk_stop_edge': param('crosswalk_stop_edge', str),
             }], on_exit=stop_on_exit('camera')),
        Node(package='camera_perception_pkg', executable='yolov8_node',
             output='screen', parameters=[{'model': param('model', str), 'device': param('device', str),
                 'enable_green_grayscale_fallback': param('enable_green_grayscale_fallback', bool),
                 'target_lane': param('target_lane', str),
                 'green_context_hold': param('green_context_hold', float),
                 'green_color_probe_interval': param('green_color_probe_interval', float),
                 'green_color_confirmations': param('green_color_confirmations', int)}],
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
                 'lane_loss_hold': param('lane_loss_hold', float),
                 'recovery_speed': param('recovery_speed', int),
                 'recovery_steering': param('recovery_steering', int),
                 'reacquire_hold': param('reacquire_hold', float),
                 'path_gap_grace': param('path_gap_grace', float),
                 'steering_gain': param('steering_gain', float),
                 'steering_offset': param('steering_offset', float),
                 'steering_sign': param('steering_sign', float),
                 'target_lane': param('target_lane', str),
                 'enable_crosswalk_guard': param('enable_crosswalk_guard', bool),
                 'crosswalk_steering_rate': param('crosswalk_steering_rate', float),
                 'red_stop_at_crosswalk': param('red_stop_at_crosswalk', bool),
                 'crosswalk_stop_y_ratio': param('crosswalk_stop_y_ratio', float),
                 'crosswalk_stop_edge': param('crosswalk_stop_edge', str),
                 'crosswalk_approach_speed': param('crosswalk_approach_speed', int),
                 'crosswalk_approach_timeout': param('crosswalk_approach_timeout', float),
                 'crosswalk_position_timeout': param('crosswalk_position_timeout', float),
                 'obey_traffic_lights': param('obey_traffic_lights', bool),
                 'enable_obstacle_avoidance': ParameterValue(avoidance, value_type=bool),
                 'swerve_speed': param('swerve_speed', int),
                 'swerve_duration': param('swerve_duration', float),
                 'swerve_cooldown': param('swerve_cooldown', float),
             }], on_exit=stop_on_exit('motion planner')),
        Node(package='camera_perception_pkg', executable='traffic_light_detector_node',
             output='screen', condition=IfCondition(LaunchConfiguration('obey_traffic_lights')),
             on_exit=stop_on_exit('traffic light detector')),
        Node(package='lidar_perception_pkg', executable='lidar_publisher_node',
             output='screen', parameters=[{'port': param('lidar_port', str)}],
             condition=IfCondition(LaunchConfiguration('enable_lidar')),
             on_exit=stop_on_exit('lidar publisher')),
        Node(package='lidar_perception_pkg', executable='lidar_processor_node',
             output='screen', parameters=[{'offset': param('lidar_offset', float),
                                           'flip': param('lidar_flip', bool)}],
             condition=IfCondition(LaunchConfiguration('enable_lidar')),
             on_exit=stop_on_exit('lidar processor')),
        Node(package='lidar_perception_pkg', executable='lidar_obstacle_detector_node',
             output='screen', condition=IfCondition(avoidance),
             on_exit=stop_on_exit('lidar obstacle detector')),
        Node(package='decision_making_pkg', executable='parking_node',
             output='screen', parameters=[{'speed': param('parking_speed', int)}],
             condition=IfCondition(LaunchConfiguration('enable_parking')),
             on_exit=stop_on_exit('parking')),
        Node(package='serial_communication_pkg', executable='serial_sender_node',
             output='screen', parameters=[{'port': param('port', str)}],
             condition=IfCondition(LaunchConfiguration('enable_serial')),
             on_exit=stop_on_exit('serial sender')),
        ExecuteProcess(cmd=['gnome-terminal', '--wait', '--', 'bash', '-c', keyboard_cmd],
                       output='screen', condition=IfCondition(LaunchConfiguration('enable_teleop')),
                       on_exit=stop_on_exit('keyboard terminal')),
    ]
    return LaunchDescription(actions)
