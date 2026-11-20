#!/usr/bin/env python3
"""Obstacle episodes of a recorded lap, drawn frame by frame with what the car did and the judgment.

usage:
  obstacle_review.py extract <lap dir> [--step 0.5]
      Finds the stretches where a car was close ahead (box bottom >= 300, ~2 m), saves their frames to
      <lap>/obstacle_review/ep<N>/ with the commanded direction drawn on them, and writes
      <lap>/obstacle_review/episodes.json and labels.csv (one row per episode to fill in).
  obstacle_review.py render <lap dir>
      Reads labels.csv (episode, t_from, t_to, should, note; several rows per episode allowed) and draws
      on every frame: the actual direction (white arrow), the judgment (green = right, red = wrong) and,
      when wrong, the direction it should have gone (green arrow). Also writes sheet.jpg per episode.

should: hard_left | hard_right | left | right | straight | stop | reverse_nose_left | reverse_nose_right | any
(hard = steering 5 or more: a lane change; left/right 2-4: following a curve)
Directions are the commanded steering as seen from the car: left = steering <= -2, right >= +2.
"""

import argparse
import csv
import json
import math
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
NEAR_Y = 300          # box bottom: ~2 m between the cars
GAP = 1.5             # s between close sightings that still belong to one episode
PAD = 1.0             # s of frames before / after


def read(db, name):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    row = db.execute('select id, type from topics where name=?', (name,)).fetchone()
    if row is None:
        return []
    T = get_message(row[1])
    return [(ts / 1e9, deserialize_message(d, T))
            for ts, d in db.execute('select timestamp, data from messages where topic_id=? order by timestamp', (row[0],))]


def value_at(rows, t):
    lo, hi, found = 0, len(rows) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if rows[mid][0] <= t:
            found, lo = rows[mid][1], mid + 1
        else:
            hi = mid - 1
    return found


def boxes(msg):
    out = []
    for d in msg.detections if msg else []:
        cx, cy, w, h = d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size.x, d.bbox.size.y
        out.append([d.id, round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)])
    return out


def action(steer, speed):
    if speed == 0:
        return 'stop'
    if speed < 0:
        return 'reverse_nose_left' if steer > 0 else 'reverse_nose_right' if steer < 0 else 'reverse'
    if steer <= -5:
        return 'hard_left'
    if steer >= 5:
        return 'hard_right'
    return 'left' if steer <= -2 else 'right' if steer >= 2 else 'straight'


KO = {'left_any': '왼쪽', 'right_any': '오른쪽', 'hard_left': '크게 왼쪽(차선 변경)', 'hard_right': '크게 오른쪽(차선 변경)', 'left': '왼쪽', 'right': '오른쪽', 'straight': '직진', 'stop': '정지', 'reverse': '후진',
      'reverse_nose_left': '후진(앞머리 왼쪽)', 'reverse_nose_right': '후진(앞머리 오른쪽)', 'any': '판단 없음'}


# Left-right order of the directions (reverse by where the nose turns).
SIDE = {'left_any': -1.5, 'right_any': 1.5, 'hard_left': -2, 'reverse_nose_left': -1.5, 'left': -1, 'straight': 0, 'stop': 0, 'reverse': 0,
        'any': 0, 'right': 1, 'reverse_nose_right': 1.5, 'hard_right': 2}


def head(draw, tail, tip, color, width):
    draw.line([tail, tip], fill=color, width=width)
    ang = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
    pts = [tip] + [(tip[0] - 26 * math.cos(ang + s), tip[1] - 26 * math.sin(ang + s)) for s in (0.5, -0.5)]
    draw.polygon(pts, fill=color)


def arrow(draw, kind, center, color, size=70, width=9):
    """Arrow of the driving direction: up = forward, tilted by the steering; down = reverse."""
    x, y = center
    tilt = {'hard_left': -1.0, 'left': -0.45, 'straight': 0.0, 'right': 0.45, 'hard_right': 1.0,
            'left_any': -0.7, 'right_any': 0.7}
    if kind in tilt:
        head(draw, (x, y), (x + tilt[kind] * size, y - size), color, width)
    elif kind.startswith('reverse'):
        dx = {'reverse_nose_left': 0.6, 'reverse_nose_right': -0.6}.get(kind, 0.0)   # rear swings opposite
        head(draw, (x, y - size // 2), (x + dx * size, y + size // 2), color, width)
    elif kind == 'stop':
        draw.rectangle([x - 22, y - 50, x + 22, y - 6], outline=color, width=width)


def matches(act, should):
    if should == 'any':
        return True
    if should in ('left_any', 'right_any'):
        return act in (('left', 'hard_left') if should == 'left_any' else ('right', 'hard_right'))
    if should.startswith('reverse') and act.startswith('reverse'):
        return should == act or should == 'reverse'
    return act == should


def extract(lap, step):
    db = sqlite3.connect(str(next((lap / 'bag').glob('*.db3'))))
    images = read(db, '/image_raw')
    dets = read(db, '/vehicle_detections')
    ctl = read(db, '/topic_control_signal')
    state = read(db, '/control_state')
    mode = read(db, '/driving_mode')
    own = read(db, '/camera_obstacle_info')
    if not images or not dets:
        sys.exit('no images or vehicle detections in this lap')
    t0 = images[0][0]
    close = [t for t, m in dets if any(b[4] >= NEAR_Y for b in boxes(m))]
    episodes = []
    for t in close:
        if episodes and t - episodes[-1][1] <= GAP:
            episodes[-1][1] = t
        else:
            episodes.append([t, t])
    episodes = [e for e in episodes if e[1] - e[0] >= 0.3]
    out_dir = lap / 'obstacle_review'
    out_dir.mkdir(exist_ok=True)
    info = []
    font = ImageFont.truetype(FONT, 18)
    for n, (a, b) in enumerate(episodes, 1):
        ep = out_dir / f'ep{n:02d}'
        ep.mkdir(exist_ok=True)
        frames, last = [], -1e9
        for t, msg in images:
            if t < a - PAD or t > b + PAD or t - last < step:
                continue
            last = t
            c = value_at(ctl, t)
            steer, speed = (c.steering, c.left_speed) if c else (0, 0)
            rec = {'t': round(t - t0, 2), 'steer': steer, 'speed': speed, 'action': action(steer, speed),
                   'state': (value_at(state, t).data if value_at(state, t) else ''),
                   'mode': (value_at(mode, t).data if value_at(mode, t) else ''),
                   'own': bool(value_at(own, t).data) if value_at(own, t) else False,
                   'cars': boxes(value_at(dets, t))}
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3].copy()
            cv2.imwrite(str(ep / f"{rec['t']:07.2f}_raw.jpg"), img)
            frames.append(rec)
        info.append({'episode': n, 't_from': round(a - t0, 2), 't_to': round(b - t0, 2), 'frames': frames})
        draw_episode(ep, frames, [], font)
    (out_dir / 'episodes.json').write_text(json.dumps(info, indent=1, ensure_ascii=False))
    labels = out_dir / 'labels.csv'
    if not labels.exists():
        with labels.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 't_from', 't_to', 'should', 'note'])
            for e in info:
                w.writerow([e['episode'], e['t_from'] - PAD, e['t_to'] + PAD, '', ''])
    print(f'{len(info)} episodes -> {out_dir}')
    for e in info:
        print(f"  ep{e['episode']:02d} {e['t_from']:7.1f}-{e['t_to']:7.1f}s  {len(e['frames'])} frames")


def label_for(labels, n, t):
    for row in labels:
        if int(row['episode']) == n and float(row['t_from']) <= t <= float(row['t_to']) and row['should']:
            return row
    return None


def draw_episode(ep, frames, labels, font, n=None):
    tiles = []
    for rec in frames:
        img = cv2.imread(str(ep / f"{rec['t']:07.2f}_raw.jpg"))
        for side, x0, y0, x1, y1 in rec['cars']:
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
            cv2.putText(img, f'{side or "?"} {y1}', (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, 640, 52], fill=(0, 0, 0))
        act = rec['action']
        draw.text((8, 2), f"{rec['t']:.1f}s  실제: {KO.get(act, act)} (조향 {rec['steer']:+d}, 속도 {rec['speed']:+d})"
                          f"  {'자율' if rec['mode'] == 'autonomous_active' else '대기/수동'}", font=font, fill=(255, 255, 255))
        draw.text((8, 26), rec['state'][:52], font=font, fill=(180, 180, 180))
        arrow(draw, act, (320, 420), (255, 255, 255))
        row = label_for(labels, n, rec['t']) if labels else None
        if row is not None:
            ok = matches(act, row['should'])
            color = (40, 220, 40) if ok else (240, 40, 40)
            draw.rectangle([0, 440, 640, 480], fill=(0, 0, 0))
            text = '맞음' if ok else f"틀림 → {KO.get(row['should'], row['should'])}"
            if row.get('note'):
                text += f"  ({row['note']})"
            draw.text((8, 448), text, font=font, fill=color)
            draw.rectangle([0, 0, 6, 480], fill=color)
            draw.rectangle([634, 0, 640, 480], fill=color)
            if not ok:
                # The right answer sits on its side of the actual arrow: left of it when it should have gone
                # more to the left, right of it when more to the right.
                more_left = SIDE.get(row['should'], 0) < SIDE.get(act, 0)
                arrow(draw, row['should'], (320 - 130 if more_left else 320 + 130, 420), (40, 220, 40))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(ep / f"{rec['t']:07.2f}.jpg"), out)
        tiles.append(cv2.resize(out, (320, 240)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(ep / 'sheet.jpg'), sheet)


def render(lap):
    out_dir = lap / 'obstacle_review'
    info = json.loads((out_dir / 'episodes.json').read_text())
    with (out_dir / 'labels.csv').open() as f:
        labels = list(csv.DictReader(f))
    font = ImageFont.truetype(FONT, 18)
    summary = []
    for e in info:
        n = e['episode']
        draw_episode(out_dir / f'ep{n:02d}', e['frames'], labels, font, n)
        judged = [(matches(r['action'], row['should'])) for r in e['frames']
                  for row in [label_for(labels, n, r['t'])] if row is not None and row['should'] != 'any']
        summary.append((n, len(judged), sum(judged)))
    for n, total, ok in summary:
        print(f'  ep{n:02d}: {ok}/{total} frames right')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('extract', 'render'))
    p.add_argument('lap', type=Path)
    p.add_argument('--step', type=float, default=0.5)
    a = p.parse_args()
    extract(a.lap, a.step) if a.command == 'extract' else render(a.lap)


if __name__ == '__main__':
    main()
