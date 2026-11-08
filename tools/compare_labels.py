#!/usr/bin/env python3
"""Compare hand labels (labels.csv) with the driving code's actions.

usage: compare_labels.py <lap dir> [prediction]
prediction: the lap's own recording (default, what the code did while driving) or a
replay_* dir made by replay_lap.sh (what the modified code does on the same frames).
Writes label_summary.txt, label_comparison.csv and label_comparison.png next to the prediction.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_lap import read_bag, value_at  # noqa: E402


def main(lap, prediction=None):
    lap = Path(lap)
    source = Path(prediction) if prediction else lap / 'bag'
    out = source if source.name != 'bag' else lap
    images = read_bag(lap / 'bag', ['/image_raw', '/driving_mode'])
    t0 = images['/image_raw'][0][0]
    pred = read_bag(source, ['/autonomous_control_signal', '/control_state'])
    rows = []
    with open(lap / 'labels.csv') as f:
        for label in csv.DictReader(f):
            if label['steering'].strip() == '' or label['go'].strip() == '':
                continue
            t = t0 + float(label['time'])
            cmd = value_at(pred['/autonomous_control_signal'], t)
            if cmd is None:
                continue
            mode = value_at(images['/driving_mode'], t)
            state = value_at(pred['/control_state'], t)
            rows.append({'index': int(label['index']), 'time': float(label['time']),
                         'label_steering': int(label['steering']), 'label_go': int(label['go']),
                         'code_steering': cmd.steering, 'code_speed': cmd.left_speed,
                         'active': getattr(mode, 'data', '') == 'autonomous_active',
                         'state': getattr(state, 'data', ''), 'note': label.get('note', '')})
    if not rows:
        sys.exit('labels.csv has no filled rows that overlap the prediction.')
    d = np.array([abs(r['label_steering'] - r['code_steering']) for r in rows])
    go = np.array([r['label_go'] == int(r['code_speed'] > 0) for r in rows])
    sign = np.array([np.sign(r['label_steering']) == np.sign(r['code_steering'])
                     or abs(r['label_steering'] - r['code_steering']) <= 1 for r in rows])
    lines = [f'labels : {lap / "labels.csv"} ({len(rows)} labelled frames)', f'code   : {source}', '',
             f'steering exact {100 * np.mean(d == 0):5.1f}%   within ±1 {100 * np.mean(d <= 1):5.1f}%'
             f'   within ±2 {100 * np.mean(d <= 2):5.1f}%   mean error {d.mean():.2f}',
             f'steering direction agreement {100 * sign.mean():5.1f}%',
             f'go/stop agreement {100 * go.mean():5.1f}%', '',
             'Worst frames (label vs code):']
    worst = sorted(rows, key=lambda r: (r['label_go'] != int(r['code_speed'] > 0),
                                        abs(r['label_steering'] - r['code_steering'])), reverse=True)[:15]
    for r in sorted(worst, key=lambda r: r['index']):
        if abs(r['label_steering'] - r['code_steering']) < 2 and r['label_go'] == int(r['code_speed'] > 0):
            continue
        lines.append(f'  #{r["index"]:4d} t={r["time"]:6.1f}s  label steer {r["label_steering"]:+d} '
                     f'{"go" if r["label_go"] else "stop"}  | code steer {r["code_steering"]:+d} speed '
                     f'{r["code_speed"]} [{r["state"]}]  {r["note"]}')
    text = '\n'.join(lines)
    print(text)
    (out / 'label_summary.txt').write_text(text + '\n')
    with open(out / 'label_comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = [r['time'] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, [r['label_steering'] for r in rows], 'o-', label='label (hand)', markersize=3)
    axes[0].plot(t, [r['code_steering'] for r in rows], 's-', label='code', markersize=3, alpha=0.7)
    axes[0].set_ylabel('steering (+ right)')
    axes[1].plot(t, [r['label_go'] for r in rows], 'o-', label='label go', markersize=3)
    axes[1].plot(t, [int(r['code_speed'] > 0) for r in rows], 's-', label='code go', markersize=3, alpha=0.7)
    axes[1].set_ylabel('go (1) / stop (0)')
    axes[1].set_xlabel('time since start of recording [s]')
    for ax in axes:
        ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'label_comparison.png', dpi=110)
    print(f'\nSaved label_summary.txt, label_comparison.csv, label_comparison.png in {out}')


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
