#!/usr/bin/env python3
"""Stop-and-go driving by hand while recording frames and actions.

  step_driver.py serve                 open camera + Arduino, record into laps/step_<time>/
  step_driver.py step STEER SPEED SECS [note...]
                                         drive for SECS (<= 1.5 s), stop, save a snapshot
  step_driver.py look                  save a snapshot without moving
  step_driver.py quit                  stop the car and close everything

The server sends a stop command whenever no step is running, so the car only moves during
a step. Every camera frame at 10 Hz is saved with the command in effect (actions.csv), and
each decision with the snapshot it was based on goes to decisions.csv: together they are a
dataset that can be used later without the car.
"""

import csv
import glob
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import serial

WS = Path(__file__).resolve().parent.parent
SOCKET = '/tmp/step_driver.sock'
MAX_SECS = 1.5
MAX_SPEED = 180
MAX_STEER = 7
RATE = 10.0


def message(steer, speed):
    return f's{int(steer)}l{int(speed)}r{int(speed)}\n'.encode('ascii')


class Driver:
    def __init__(self):
        self.lap = WS / 'laps' / f'step_{datetime.now():%Y%m%d_%H%M%S}'
        for sub in ('frames', 'snapshots'):
            (self.lap / sub).mkdir(parents=True)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_time = 0.0
        self.steer, self.speed, self.until = 0, 0, 0.0
        self.running = True
        self.step = 0
        self.start = time.monotonic()

        cams = sorted(glob.glob('/dev/v4l/by-id/*C920*-video-index0'))
        if not cams:
            raise SystemExit('C920 camera not found')
        self.cap = cv2.VideoCapture(int(os.path.realpath(cams[0]).replace('/dev/video', '')))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.read()[0]:
            raise SystemExit('Cannot read the camera')
        self.serial = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1, write_timeout=0.2, exclusive=True)
        time.sleep(2.5)  # opening the port resets the Mega
        self.send(0, 0)

        self.actions = open(self.lap / 'actions.csv', 'w', newline='')
        self.actions_csv = csv.writer(self.actions)
        self.actions_csv.writerow(['time', 'file', 'steering', 'speed', 'step'])
        self.decisions = open(self.lap / 'decisions.csv', 'w', newline='')
        self.decisions_csv = csv.writer(self.decisions)
        self.decisions_csv.writerow(['step', 'time', 'snapshot', 'steering', 'speed', 'secs', 'note'])

    def send(self, steer, speed):
        self.serial.write(message(steer, speed))

    def camera_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.frame_time = frame, time.monotonic()

    def control_loop(self):
        index = 0
        while self.running:
            tick = time.monotonic()
            with self.lock:
                moving = tick < self.until
                steer, speed = self.steer, (self.speed if moving else 0)
                frame = self.frame
            self.send(steer, speed)
            if frame is not None:
                name = f'{index:05d}.jpg'
                cv2.imwrite(str(self.lap / 'frames' / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.actions_csv.writerow([f'{tick - self.start:.2f}', name, steer, speed,
                                           self.step if moving else ''])
                index += 1
            time.sleep(max(0.0, 1.0 / RATE - (time.monotonic() - tick)))

    def snapshot(self):
        time.sleep(0.4)  # let the car settle and a fresh frame arrive
        with self.lock:
            frame = self.frame.copy()
        path = self.lap / 'snapshots' / f'step_{self.step:04d}.jpg'
        cv2.line(frame, (320, 240), (320, 479), (255, 0, 255), 1)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path

    def handle(self, words):
        if words[0] == 'look':
            return str(self.snapshot())
        if words[0] == 'step':
            steer = max(-MAX_STEER, min(MAX_STEER, int(words[1])))
            speed = max(-MAX_SPEED, min(MAX_SPEED, int(words[2])))
            secs = max(0.0, min(MAX_SECS, float(words[3])))
            note = ' '.join(words[4:])
            self.step += 1
            self.decisions_csv.writerow([self.step, f'{time.monotonic() - self.start:.2f}',
                                         f'step_{self.step - 1:04d}.jpg', steer, speed, secs, note])
            self.decisions.flush()
            with self.lock:
                self.steer, self.speed, self.until = steer, speed, time.monotonic() + secs
            time.sleep(secs)
            with self.lock:
                self.until = 0.0
            return str(self.snapshot())
        if words[0] == 'quit':
            self.running = False
            return 'bye'
        return 'unknown command'

    def serve(self):
        threading.Thread(target=self.camera_loop, daemon=True).start()
        threading.Thread(target=self.control_loop, daemon=True).start()
        if os.path.exists(SOCKET):
            os.unlink(SOCKET)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET)
        server.listen(1)
        server.settimeout(0.5)
        print(f'recording into {self.lap}', flush=True)
        self.snapshot()  # step_0000.jpg: the view before the first move
        try:
            while self.running:
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    words = conn.recv(1024).decode().split()
                    try:
                        reply = self.handle(words) if words else 'empty'
                    except (ValueError, IndexError) as error:
                        reply = f'error: {error}'
                    conn.sendall(reply.encode())
        finally:
            self.running = False
            with self.lock:
                self.until = 0.0
            time.sleep(0.2)
            for _ in range(3):
                self.send(0, 0)
                time.sleep(0.05)
            self.serial.close()
            self.cap.release()
            self.actions.close()
            self.decisions.close()
            server.close()
            os.unlink(SOCKET)
            print('stopped', flush=True)


def client(words):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET)
    sock.sendall(' '.join(words).encode())
    sock.settimeout(10)
    print(sock.recv(4096).decode())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == 'serve':
        driver = Driver()
        signal.signal(signal.SIGTERM, lambda *_: setattr(driver, 'running', False))
        signal.signal(signal.SIGINT, lambda *_: setattr(driver, 'running', False))
        driver.serve()
    else:
        client(sys.argv[1:])
