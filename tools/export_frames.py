#!/usr/bin/env python3
"""Extract frames from a recorded lap for hand labelling (by hand).

usage: export_frames.py <lap dir> [--hz 2]
Creates in the lap dir:
  frames/            every sampled frame, full 640x480
  sheets/            6 frames per image with index and time, for quick review
  code_actions.csv   what the driving code commanded at each sampled frame
  labels.csv         template to fill: steering (-7 left .. +7 right) and go/stop
The sheets deliberately do not show the code's actions, so the labels stay independent.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_lap import read_bag, value_at  # noqa: E402


def to_bgr(msg):
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lap')
    parser.add_argument('--hz', type=float, default=2.0)
    args = parser.parse_args()
    lap = Path(args.lap)
    data = read_bag(lap / 'bag', ['/image_raw', '/autonomous_control_signal', '/topic_control_signal',
                                  '/control_state', '/driving_mode'])
    images = data['/image_raw']
    if not images:
        sys.exit('No camera frames in this recording.')
    t0 = images[0][0]
    picked, next_t = [], t0
    for t, msg in images:
        if t >= next_t:
            picked.append((t, msg))
            next_t = t + 1.0 / args.hz
    (lap / 'frames').mkdir(exist_ok=True)
    (lap / 'sheets').mkdir(exist_ok=True)
    rows = []
    for index, (t, msg) in enumerate(picked):
        frame = to_bgr(msg)
        name = f'f{index:04d}_t{t - t0:06.1f}.jpg'
        cv2.imwrite(str(lap / 'frames' / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        auto = value_at(data['/autonomous_control_signal'], t)
        sent = value_at(data['/topic_control_signal'], t)
        mode = value_at(data['/driving_mode'], t)
        state = value_at(data['/control_state'], t)
        rows.append({'index': index, 'time': round(t - t0, 2), 'file': name,
                     'code_steering': getattr(auto, 'steering', ''), 'code_speed': getattr(auto, 'left_speed', ''),
                     'sent_steering': getattr(sent, 'steering', ''), 'sent_speed': getattr(sent, 'left_speed', ''),
                     'mode': getattr(mode, 'data', ''), 'state': getattr(state, 'data', '')})
    with open(lap / 'code_actions.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = lap / 'labels.csv'
    if not labels.exists():
        with open(labels, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'time', 'steering', 'go', 'note'])
            for row in rows:
                writer.writerow([row['index'], row['time'], '', '', ''])

    # Contact sheets: 3 x 2 tiles of 426x320 with a centre line to judge heading.
    tile_w, tile_h = 426, 320
    for start in range(0, len(rows), 6):
        sheet = np.full((2 * (tile_h + 30), 3 * tile_w, 3), 255, np.uint8)
        for k, row in enumerate(rows[start:start + 6]):
            frame = cv2.imread(str(lap / 'frames' / row['file']))
            frame = cv2.resize(frame, (tile_w, tile_h))
            cv2.line(frame, (tile_w // 2, tile_h // 2), (tile_w // 2, tile_h), (255, 0, 255), 1)
            x, y = (k % 3) * tile_w, (k // 3) * (tile_h + 30)
            sheet[y + 30:y + 30 + tile_h, x:x + tile_w] = frame
            cv2.putText(sheet, f'#{row["index"]}  t={row["time"]:.1f}s', (x + 8, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(str(lap / 'sheets' / f'sheet_{start // 6:03d}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f'{len(rows)} frames ({args.hz} Hz) -> {lap}/frames, {(len(rows) + 5) // 6} sheets -> {lap}/sheets')
    print(f'Fill {labels} (steering -7..+7, go 1/0), then: tools/compare_labels.py {lap}')


if __name__ == '__main__':
    main()
