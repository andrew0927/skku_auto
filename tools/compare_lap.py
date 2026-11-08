#!/usr/bin/env python3
"""Compare the actions recorded while driving with the actions recomputed from the same frames.

usage: compare_lap.py <lap dir> <replay dir>
Writes summary.txt, comparison.csv, comparison.png and frames of the largest disagreements
into the replay dir.
"""

import bisect
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

MATCH_TOLERANCE = 0.06   # s; the planner publishes every 0.1 s in both runs
BIG_STEERING_ERROR = 2   # steering steps that count as a clear disagreement


def read_bag(path, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    data = {t: [] for t in topics}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        data[topic].append((stamp / 1e9, deserialize_message(raw, get_message(types[topic]))))
    return data


def nearest(times, t):
    i = bisect.bisect_left(times, t)
    best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - t), default=None)
    return best if best is not None and abs(times[best] - t) <= MATCH_TOLERANCE else None


def value_at(series, t):
    """Latest value at or before t (for states that are published on change)."""
    times = [s for s, _ in series]
    i = bisect.bisect_right(times, t) - 1
    return series[i][1] if i >= 0 else None


def spans(rows, bad):
    out, start = [], None
    for i, row in enumerate(rows):
        if bad(row) and start is None:
            start = i
        elif not bad(row) and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(rows) - 1))
    return out


def main(lap, replay):
    lap, replay = Path(lap), Path(replay)
    truth = read_bag(lap / 'bag', ['/autonomous_control_signal', '/topic_control_signal',
                                   '/control_state', '/driving_mode', '/image_raw'])
    again = read_bag(replay, ['/autonomous_control_signal', '/control_state'])
    rec, rep = truth['/autonomous_control_signal'], again['/autonomous_control_signal']
    if not rec or not rep:
        sys.exit(f'No planner output to compare (recorded {len(rec)}, replayed {len(rep)}).')
    rep_times = [t for t, _ in rep]
    t0 = rec[0][0]
    rows = []
    for t, cmd in rec:
        j = nearest(rep_times, t)
        if j is None:
            continue
        other = rep[j][1]
        mode = value_at(truth['/driving_mode'], t)
        rows.append({
            'time': round(t - t0, 2),
            'active': bool(mode is not None and mode.data == 'autonomous_active'),
            'steer_truth': cmd.steering, 'steer_replay': other.steering,
            'speed_truth': cmd.left_speed, 'speed_replay': other.left_speed,
            'state_truth': getattr(value_at(truth['/control_state'], t), 'data', ''),
            'state_replay': getattr(value_at(again['/control_state'], rep[j][0]), 'data', ''),
            'abs_time': t,
        })
    if not rows:
        sys.exit('Recorded and replayed outputs do not overlap in time.')

    def summary(subset, title):
        if not subset:
            return [f'{title}: no samples']
        d_steer = np.array([abs(r['steer_truth'] - r['steer_replay']) for r in subset])
        d_speed = np.array([abs(r['speed_truth'] - r['speed_replay']) for r in subset])
        go = np.array([(r['speed_truth'] > 0) == (r['speed_replay'] > 0) for r in subset])
        state = np.array([r['state_truth'] == r['state_replay'] for r in subset])
        return [
            f'{title}: {len(subset)} samples ({len(subset) / 10:.1f} s)',
            f'  steering  exact {100 * np.mean(d_steer == 0):5.1f}%   within ±1 {100 * np.mean(d_steer <= 1):5.1f}%'
            f'   mean error {d_steer.mean():.2f}   max {d_steer.max()}',
            f'  speed     exact {100 * np.mean(d_speed == 0):5.1f}%   mean error {d_speed.mean():.1f}',
            f'  go/stop agreement {100 * go.mean():5.1f}%',
            f'  control state agreement {100 * state.mean():5.1f}%',
        ]

    active = [r for r in rows if r['active']]
    lines = [f'lap    : {lap}', f'replay : {replay}',
             f'matched {len(rows)} of {len(rec)} recorded planner outputs', '']
    lines += summary(active, 'While driving autonomously') + [''] + summary(rows, 'Whole recording')

    bad = lambda r: (abs(r['steer_truth'] - r['steer_replay']) >= BIG_STEERING_ERROR
                     or (r['speed_truth'] > 0) != (r['speed_replay'] > 0))
    worst = sorted(spans(rows, bad), key=lambda s: s[1] - s[0], reverse=True)[:5]
    images = truth['/image_raw']
    image_times = [t for t, _ in images]
    if worst:
        lines += ['', 'Largest disagreements (frames saved as disagreement_*.png):']
    for k, (a, b) in enumerate(sorted(worst), 1):
        r = rows[a]
        lines.append(f'  {k}. t={r["time"]:.1f}-{rows[b]["time"]:.1f}s  truth steer {r["steer_truth"]} speed '
                     f'{r["speed_truth"]} [{r["state_truth"]}]  vs replay steer {r["steer_replay"]} speed '
                     f'{r["speed_replay"]} [{r["state_replay"]}]')
        i = bisect.bisect_right(image_times, r['abs_time']) - 1
        if i >= 0:
            msg = images[i][1]
            frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1).copy()
            cv2.putText(frame, f't={r["time"]:.1f}s truth {r["steer_truth"]}/{r["speed_truth"]}  '
                               f'replay {r["steer_replay"]}/{r["speed_replay"]}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(replay / f'disagreement_{k}_t{r["time"]:.0f}s.png'), frame)

    text = '\n'.join(lines)
    print(text)
    (replay / 'summary.txt').write_text(text + '\n')
    with open(replay / 'comparison.csv', 'w', newline='') as f:
        fields = [k for k in rows[0] if k != 'abs_time']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = [r['time'] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, key, label in ((axes[0], 'steer', 'steering'), (axes[1], 'speed', 'speed')):
        ax.step(t, [r[f'{key}_truth'] for r in rows], where='post', label='recorded (truth)', linewidth=1.6)
        ax.step(t, [r[f'{key}_replay'] for r in rows], where='post', label='replay', linewidth=1.0)
        for a, b in spans(rows, lambda r: r['active']):
            ax.axvspan(rows[a]['time'], rows[b]['time'], color='0.9', zorder=0)
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
    axes[1].set_xlabel('time since start of recording [s]  (grey: autonomous driving)')
    fig.tight_layout()
    fig.savefig(replay / 'comparison.png', dpi=110)
    print(f'\nSaved summary.txt, comparison.csv, comparison.png in {replay}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
