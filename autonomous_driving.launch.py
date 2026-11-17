"""Camera -> YOLO -> lane -> path -> motion -> keyboard gate -> Arduino.

record:=true saves camera frames and commands to laps/; replay_bag:=<bag> feeds a recorded
bag back through the pipeline without camera, Arduino or keyboard (see replay_lap.sh).
"""

from datetime import datetime
import os
from pathlib import Path
import shlex

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction,
                            SetEnvironmentVariable, Shutdown, TimerAction)
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
        'corner_speed': ('150', 'Speed at full steering (slower in curves); -1 = same as cruise_speed'),
        'estimated_speed': ('150', 'Speed when lane2 is estimated (uncertain); -1 = same as cruise_speed'),
        'hold_time': ('0.5', 'Seconds the last moving command is kept at speed 150 (shorter when faster)'),
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
        'crosswalk_stop_y': ('440', 'Stop for red when the crosswalk bottom edge passes this image y (0..480)'),
        'red_approach_speed': ('150', 'Speed approaching/crossing the crosswalk; -1 = same as cruise_speed'),
        'crossing_time': ('2.5', 'Seconds (at speed 150, shorter when faster) of holding the lane heading after green'),
        'red_release_time': ('3.0', 'Start when red has not been seen for this long (0 disables)'),
        'light_min_score': ('0.5', 'Ignore traffic-light detections below this YOLO score'),
        'enable_camera': ('true', 'Publish camera frames (false when replaying a bag)'),
        'use_sim_time': ('false', 'Follow the /clock of a replayed bag'),
        'record': ('false', 'Record camera frames and commands into laps/'),
        'replay_bag': ('', 'Recorded bag to feed through the pipeline instead of the camera'),
        'replay_delay': ('10.0', 'Seconds to wait for YOLO to load before replaying'),
        'keyboard_mode': ('', '1 or 2 skips the keyboard mode prompt'),
    }
    arguments = [DeclareLaunchArgument(name, default_value=value, description=description)
                 for name, (value, description) in defaults.items()]
    def param(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)
    def stop_on_exit(name):
        return [Shutdown(reason=f'{name} exited')]
    sim_time = {'use_sim_time': param('use_sim_time', bool)}
    domain = os.environ.get('ROS_DOMAIN_ID', '10')
    keyboard_cmd = (
        'export TELEOP_MODE="$1" && '
        f'export ROS_DOMAIN_ID={domain} ROS_LOCALHOST_ONLY={os.environ.get("ROS_LOCALHOST_ONLY", "0")} && '
        'source /opt/ros/humble/setup.bash && '
        f'source {shlex.quote(str(ws / "install/local_setup.bash"))} && '
        'exec ros2 run decision_making_pkg teleop_keyboard_node'
        ' --ros-args -p auto_speed_limit:='
    )
    actions = [SetEnvironmentVariable('ROS_DOMAIN_ID', domain)] + arguments
    actions += [
        Node(package='camera_perception_pkg', executable='image_publisher_node',
             output='screen', parameters=[{
                 'cam_num': param('cam_num', int), 'data_source': param('data_source', str),
                 'img_dir': param('img_dir', str), 'video_path': param('video_path', str),
                 'timer': param('input_period', float), 'logger': param('show_image', bool),
             }], condition=IfCondition(LaunchConfiguration('enable_camera')),
             on_exit=stop_on_exit('camera')),
        Node(package='camera_perception_pkg', executable='yolov8_node',
             output='screen', parameters=[sim_time, {'model': param('model', str),
                                                      'device': param('device', str)}],
             on_exit=stop_on_exit('YOLO')),
        Node(package='camera_perception_pkg', executable='lane_info_extractor_node',
             output='screen', parameters=[sim_time, {'show_image': param('show_image', bool),
                                                      'target_lane': param('target_lane', str)}],
             on_exit=stop_on_exit('lane extractor')),
        Node(package='decision_making_pkg', executable='path_planner_node',
             output='screen', parameters=[sim_time], on_exit=stop_on_exit('path planner')),
        Node(package='decision_making_pkg', executable='motion_planner_node',
             output='both', parameters=[sim_time, {
                 'driving_speed': param('cruise_speed', int),
                 'estimated_speed': param('estimated_speed', int),
                 'corner_speed': param('corner_speed', int),
                 'hold_time': param('hold_time', float),
                 'reacquire_hold': param('reacquire_hold', float),
                 'steering_gain': param('steering_gain', float),
                 'steering_offset': param('steering_offset', float),
                 'steering_sign': param('steering_sign', float),
                 'obey_traffic_lights': param('obey_traffic_lights', bool),
                 'red_approach_speed': param('red_approach_speed', int),
                 'crossing_time': param('crossing_time', float),
                 'red_release_time': param('red_release_time', float),
             }], on_exit=stop_on_exit('motion planner')),
        Node(package='camera_perception_pkg', executable='traffic_light_detector_node',
             output='both', parameters=[sim_time, {'show_image': param('show_image', bool),
                                          'crosswalk_stop_y': param('crosswalk_stop_y', int),
                                          'min_score': param('light_min_score', float)}],
             condition=IfCondition(LaunchConfiguration('obey_traffic_lights')),
             on_exit=stop_on_exit('traffic light detector')),
        Node(package='serial_communication_pkg', executable='serial_sender_node',
             output='screen', parameters=[{'port': param('port', str)}],
             condition=IfCondition(LaunchConfiguration('enable_serial')),
             on_exit=stop_on_exit('serial sender')),
        ExecuteProcess(cmd=['gnome-terminal', '--wait', '--', 'bash', '-c',
                            [keyboard_cmd, LaunchConfiguration('cruise_speed')],
                            'teleop', LaunchConfiguration('keyboard_mode')],
                       output='screen', condition=IfCondition(LaunchConfiguration('enable_teleop')),
                       on_exit=stop_on_exit('keyboard terminal')),
        OpaqueFunction(function=lambda context: record_actions(context, ws, defaults)),
        OpaqueFunction(function=replay_actions),
    ]
    return LaunchDescription(actions)


# Ground truth: planner output, what was actually sent, driving mode and states.
RECORD_TOPICS = ['/image_raw', '/autonomous_control_signal', '/topic_control_signal',
                 '/driving_mode', '/driving_mode_reset', '/control_state',
                 '/yolov8_traffic_light_info', '/crosswalk_near', '/crosswalk_ahead', '/lane_tracking_source',
                 '/override_control_signal', '/remote_key']
# Inputs a replay feeds back; everything else is recomputed from the frames.
REPLAY_INPUTS = ['/image_raw', '/driving_mode', '/driving_mode_reset']
REPLAY_OUTPUTS = ['/autonomous_control_signal', '/control_state', '/yolov8_traffic_light_info',
                  '/crosswalk_near', '/lane_tracking_source']
# Settings that describe the hardware session, not the driving logic.
NOT_REPLAYED = {'cam_num', 'port', 'data_source', 'img_dir', 'video_path', 'enable_serial',
                'enable_teleop', 'enable_camera', 'use_sim_time', 'record', 'replay_bag',
                'replay_delay', 'show_image', 'device'}


def stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def record_actions(context, ws, defaults):
    if context.launch_configurations.get('record', 'false').lower() != 'true':
        return []
    lap = ws / 'laps' / f'lap_{stamp()}'
    lap.mkdir(parents=True)
    settings = [f'{name}:={context.launch_configurations[name]}'
                for name in defaults if name not in NOT_REPLAYED]
    (lap / 'launch_args.txt').write_text('\n'.join(settings) + '\n')
    print(f'[record] saving to {lap}', flush=True)
    return [ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o', str(lap / 'bag')] + RECORD_TOPICS,
                           output='screen')]


def replay_actions(context):
    bag = context.launch_configurations.get('replay_bag', '')
    if not bag:
        return []
    bag = Path(bag).resolve()
    out = bag.parent / f'replay_{stamp()}'
    qos = out.parent / 'replay_qos.yaml'
    # /driving_mode is latched by the keyboard node; the planner subscribes transient-local.
    qos.write_text('/driving_mode:\n  history: keep_last\n  depth: 1\n'
                   '  reliability: reliable\n  durability: transient_local\n')
    delay = float(context.launch_configurations.get('replay_delay', '10.0'))
    print(f'[replay] {bag} -> {out}', flush=True)
    return [
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '--use-sim-time', '-o', str(out)] + REPLAY_OUTPUTS,
                       output='screen'),
        TimerAction(period=delay, actions=[ExecuteProcess(
            cmd=['ros2', 'bag', 'play', str(bag), '--clock', '--topics'] + REPLAY_INPUTS
                + ['--qos-profile-overrides-path', str(qos)],
            output='screen', on_exit=[Shutdown(reason='replay finished')])]),
    ]
