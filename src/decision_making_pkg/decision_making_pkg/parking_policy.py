"""The backup's right-side gap, reverse parking and exit sequence."""

import math

from .control_policy import DriveCommand, clamp


class ParkingSequence:
    def __init__(self, speed=150, obstacle=3.0, gap=3.5, wall=1.6, gain=14.0,
                 first_align=0.2, second_align=1.0, second_delay=2.0,
                 overshoot=3.0, stop=0.5, reverse=6.5, dwell=3.5,
                 exit_right=6.5, exit_straight=15.0):
        if not 0 < speed <= 255 or not 0 < obstacle < gap or wall <= 0:
            raise ValueError('Invalid parking geometry or speed')
        self.speed, self.obstacle, self.gap, self.wall, self.gain = speed, obstacle, gap, wall, gain
        self.first_align, self.second_align, self.second_delay = first_align, second_align, second_delay
        self.durations = dict(OVERSHOOT=overshoot, STOP_AFTER_OVERSHOOT=stop,
                              REVERSE_RIGHT=reverse, PARKED=dwell,
                              EXIT_RIGHT=exit_right, EXIT_STRAIGHT=exit_straight)
        if min(first_align, second_align, second_delay, *self.durations.values()) < 0:
            raise ValueError('Parking durations must be nonnegative')
        self.reset()

    def reset(self):
        self.state = 'IDLE'
        self.started = self.first_car_time = 0.0
        self.saw_first_car = False
        self.car_count = self.gap_count = 0

    def transition(self, state, now):
        self.state, self.started = state, now

    def start(self, now):
        self.reset()
        self.transition('ALIGNING_AND_SCANNING', now)

    def tick(self, now, distance, new_scan=True):
        if self.state in ('IDLE', 'DONE') or distance is None or math.isnan(distance) or distance <= 0:
            return DriveCommand()
        elapsed = now - self.started
        steering, speed = 0, self.speed
        if self.state == 'ALIGNING_AND_SCANNING':
            if not self.saw_first_car or now - self.first_car_time <= self.first_align:
                if distance < self.obstacle:
                    steering = self.gain * (distance - self.wall)
            if new_scan:
                if not self.saw_first_car:
                    self.car_count = self.car_count + 2 if distance < self.obstacle else max(0, self.car_count - 1)
                    if self.car_count >= 4:
                        self.saw_first_car, self.first_car_time, self.car_count = True, now, 0
                else:
                    self.gap_count = self.gap_count + 2 if distance > self.gap else max(0, self.gap_count - 1)
                    if self.gap_count >= 4:
                        self.car_count = self.gap_count = 0
                        self.transition('GAP_FOUND', now)
        elif self.state == 'GAP_FOUND':
            if new_scan:
                detected = distance < self.obstacle and now - self.first_car_time > self.second_delay
                self.car_count = self.car_count + 2 if detected else max(0, self.car_count - 1)
                if self.car_count >= 4:
                    self.car_count = 0
                    self.transition('OVERSHOOT', now)
        elif self.state == 'OVERSHOOT':
            if elapsed <= self.second_align and distance < self.obstacle:
                steering = self.gain * (distance - self.wall)
            if elapsed >= self.durations[self.state]:
                self.transition('STOP_AFTER_OVERSHOOT', now)
                speed = 0
        elif self.state == 'STOP_AFTER_OVERSHOOT':
            speed = 0
            if elapsed >= self.durations[self.state]:
                self.transition('REVERSE_RIGHT', now)
        elif self.state == 'REVERSE_RIGHT':
            steering, speed = 7, -self.speed
            if elapsed >= self.durations[self.state]:
                self.transition('PARKED', now)
                steering, speed = 0, 0
        elif self.state == 'PARKED':
            speed = 0
            if elapsed >= self.durations[self.state]:
                self.transition('EXIT_RIGHT', now)
        elif self.state == 'EXIT_RIGHT':
            steering = 7
            if elapsed >= self.durations[self.state]:
                self.transition('EXIT_STRAIGHT', now)
                steering = 0
        elif self.state == 'EXIT_STRAIGHT':
            if elapsed >= self.durations[self.state]:
                self.transition('DONE', now)
                speed = 0
        return DriveCommand.bounded(clamp(steering, -7, 7), speed, speed)
