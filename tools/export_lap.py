#!/usr/bin/env python3
"""Export a time range of a recorded lap as a ROS-free dataset: frames + actions + preview.

usage: export_lap.py <lap dir> START END NAME
Writes shared/NAME/ (images/, actions.csv, preview.mp4, code/) and shared/NAME.zip.
Times are seconds since the first camera frame of the recording.
"""

import csv
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_lap import read_bag, value_at  # noqa: E402

WS = Path(__file__).resolve().parent.parent
TOPICS = ['/image_raw', '/topic_control_signal', '/autonomous_control_signal', '/override_control_signal',
          '/driving_mode', '/control_state', '/lane_tracking_source', '/yolov8_traffic_light_info',
          '/crosswalk_near']


def main(lap, start, end, name):
    lap, start, end = Path(lap), float(start), float(end)
    out = WS / 'shared' / name
    if out.exists():
        shutil.rmtree(out)
    (out / 'images').mkdir(parents=True)
    data = read_bag(lap / 'bag', TOPICS)
    images = data['/image_raw']
    t0 = images[0][0]
    rows = []
    video = cv2.VideoWriter(str(out / 'preview.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
    for t, msg in images:
        rel = t - t0
        if not start <= rel <= end:
            continue
        frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1).copy()
        index = len(rows)
        file = f'images/{index:05d}.jpg'
        cv2.imwrite(str(out / file), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        sent = value_at(data['/topic_control_signal'], t)
        code = value_at(data['/autonomous_control_signal'], t)
        override_on = any(0 <= t - ot <= 0.3 for ot, _ in data['/override_control_signal'])
        get = lambda topic: getattr(value_at(data[topic], t), 'data', '')
        row = {
            'frame': index, 'time_s': round(rel - start, 3), 'image': file,
            'steering': getattr(sent, 'steering', 0), 'speed': getattr(sent, 'left_speed', 0),
            'code_steering': getattr(code, 'steering', ''), 'code_speed': getattr(code, 'left_speed', ''),
            'human_override': int(override_on),
            'mode': get('/driving_mode'), 'control_state': get('/control_state'),
            'lane_source': get('/lane_tracking_source'), 'traffic_light': get('/yolov8_traffic_light_info'),
            'crosswalk_near': get('/crosswalk_near'),
        }
        rows.append(row)
        overlay = frame.copy()
        cv2.putText(overlay, f't={row["time_s"]:5.1f}s steer {row["steering"]:+d} speed {row["speed"]}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(overlay, row['control_state'][:60], (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        center = 320 + int(row['steering'] * 30)
        cv2.arrowedLine(overlay, (320, 470), (center, 400), (0, 255, 255), 3)
        video.write(overlay)
    video.release()
    with open(out / 'actions.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Code that produced these actions, for reproducibility.
    (out / 'code').mkdir()
    with tarfile.open(out / 'code' / 'ros2_ws_src.tar.gz', 'w:gz') as tar:
        for item in ['src', 'tools', 'tests', 'autonomous_driving.launch.py', 'run_autonomous.sh',
                     'run_traffic_light.sh', 'replay_lap.sh', 'README.md']:
            tar.add(WS / item, arcname=item,
                    filter=lambda ti: None if '__pycache__' in ti.name or 'Collected_Datasets' in ti.name else ti)
    shutil.copy(lap / 'launch_args.txt', out / 'code' / 'launch_args.txt')
    git = subprocess.run(['git', '-C', str(WS), 'log', '-1', '--format=%h %s'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(WS), 'status', '--short'], capture_output=True, text=True).stdout.count('\n')

    steer = np.array([r['steering'] for r in rows])
    speed = np.array([r['speed'] for r in rows])
    duration = rows[-1]['time_s']
    stops = sum(1 for a, b in zip(speed[:-1], speed[1:]) if a > 0 and b == 0)
    (out / 'README.md').write_text(f"""# {name}

One full autonomous lap of the kids-car track (traffic-light version), recorded on
{time.strftime('%Y-%m-%d', time.localtime(t0))}. The car completed the lap without leaving the road:
it stopped at the red light, crossed on green, and went through the S-curve and the circle.

| | |
|---|---|
| Frames | {len(rows)} ({len(rows) / duration:.1f} fps), {duration:.1f} s |
| Camera | Logitech C920, 640x480 BGR, front of the car |
| Source recording | `{lap.name}`, {start:.1f}-{end:.1f} s after its first frame |
| Steering range in this lap | {steer.min()} .. {steer.max()} |
| Speed range | {speed.min()} .. {speed.max()} |
| Red-light stops | {stops} |
| Override frames | {sum(r['human_override'] for r in rows)} |

## Files

- `images/00000.jpg ...` camera frames in time order.
- `actions.csv` one row per frame:
  - `steering`, `speed`: **the action actually sent to the Arduino** (the label).
    Steering is -7 (full left) .. +7 (full right); speed is motor PWM -255..255 (0 = stop).
  - `code_steering`, `code_speed`: what the driving code computed (equal to the label while
    driving autonomously).
  - `human_override`: 1 if a manual/remote override command was in effect.
  - `mode`, `control_state`, `lane_source`, `traffic_light`, `crosswalk_near`: context from the
    pipeline at that moment (driving mode, planner state, lane2 direct/estimated/lost, light
    colour, crosswalk within the stop line).
  - `time_s`: seconds since the first frame of this lap.
- `preview.mp4` the lap at 10 fps with the action drawn on each frame.
- `code/` the code that drove (`ros2_ws_src.tar.gz`, git `{git}` plus {dirty} uncommitted changes) and
  its launch settings (`launch_args.txt`, model `best.pt` = autolabeled YOLOv8s-seg with classes
  Trafficlight, crosswalk, lane1, lane2).

The label is each frame's latest command at or before the frame time; commands run at 10 Hz,
the camera at ~8 fps here, so a label can lag its frame by up to ~0.1 s.
""")
    archive = shutil.make_archive(str(out), 'zip', root_dir=out.parent, base_dir=name)
    print(f'{len(rows)} frames, {duration:.1f} s -> {out}')
    print(f'zip: {archive} ({Path(archive).stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
