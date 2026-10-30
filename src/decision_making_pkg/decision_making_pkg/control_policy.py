"""Vehicle commands and keyboard mode transitions, independent of ROS."""

from dataclasses import dataclass
from bisect import bisect_left
import math


MAX_STEERING = 7  # Must match the tested driving.ino.
MAX_DRIVING_SPEED = 255


def clamp(value, low, high):
    return max(low, min(high, value))


@dataclass(frozen=True)
class DriveCommand:
    steering: int = 0
    left_speed: int = 0
    right_speed: int = 0

    @classmethod
    def bounded(cls, steering, left_speed, right_speed):
        return cls(int(clamp(round(steering), -7, 7)),
                   int(clamp(round(left_speed), -MAX_DRIVING_SPEED, MAX_DRIVING_SPEED)),
                   int(clamp(round(right_speed), -MAX_DRIVING_SPEED, MAX_DRIVING_SPEED)))


class DriveModeController:
    def __init__(self, mode='manual', manual_speed_limit=255, command_timeout=0.5):
        if mode not in ('manual', 'autonomous'):
            raise ValueError('Choose manual or autonomous')
        self.mode = mode
        self.active = False
        self.manual = DriveCommand()
        self.auto = DriveCommand()
        self.last_auto_time = None
        self.manual_speed_limit = int(clamp(manual_speed_limit, 0, MAX_DRIVING_SPEED))
        self.command_timeout = command_timeout

    @property
    def state(self):
        if self.mode == 'manual':
            return 'manual'
        return 'autonomous_active' if self.active else 'autonomous_waiting'

    def receive_auto(self, command, now):
        self.auto = DriveCommand.bounded(command.steering, command.left_speed,
                                        command.right_speed)
        self.last_auto_time = now

    def stop(self):
        self.active = False
        self.manual = DriveCommand()
        self.auto = DriveCommand()
        self.last_auto_time = None

    def handle_key(self, key):
        if key in ('q', '\x03'):
            self.stop()
            return 'quit'
        if key in ('1', '2'):
            self.stop()
            self.mode = 'manual' if key == '1' else 'autonomous'
            return 'mode_changed'
        if key == 'x':
            self.stop()
            return 'stopped'
        if key == ' ':
            if self.mode == 'autonomous' and not self.active:
                # Use the latest fresh planner command on the same keypress.
                # command() still rejects missing or timed-out inputs.
                self.manual = DriveCommand()
                self.active = True
                return 'auto_started'
            self.stop()
            return 'stopped'
        if self.mode == 'manual':
            steering, speed = self.manual.steering, self.manual.left_speed
            if key == 'w':
                speed += 10
            elif key == 's':
                speed -= 10
            elif key == 'a':
                steering -= 1
            elif key == 'd':
                steering += 1
            else:
                return 'ignored'
            speed = clamp(speed, -self.manual_speed_limit, self.manual_speed_limit)
            self.manual = DriveCommand.bounded(steering, speed, speed)
            return 'manual_command'
        return 'ignored'

    def command(self, now):
        if self.mode == 'manual':
            return self.manual
        if (self.active and self.last_auto_time is not None
                and 0 <= now - self.last_auto_time <= self.command_timeout):
            return self.auto
        return DriveCommand()


def path_steering(points, car_x=320.0, car_y=179.0, lookahead=100.0,
                  gain=0.12, offset=0.0, sign=1.0):
    """Positive image x error means right steering, as in driving.ino."""
    if len(points) < 2 or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('Invalid path')
    ordered = sorted(points, key=lambda point: point[1])
    ys = [point[1] for point in ordered]
    target_y = clamp(car_y - lookahead, ys[0], ys[-1])
    if car_y - target_y <= 0:
        raise ValueError('Path must contain a point ahead of the car')
    index = bisect_left(ys, target_y)
    if index == 0:
        target_x = ordered[0][0]
    else:
        x0, y0 = ordered[index - 1]
        x1, y1 = ordered[index]
        if y1 <= y0:
            raise ValueError('Duplicate path heights')
        target_x = x0 + (x1 - x0) * (target_y - y0) / (y1 - y0)
    heading = math.degrees(math.atan2(target_x - car_x, car_y - target_y))
    return clamp(sign * gain * heading + offset, -MAX_STEERING, MAX_STEERING)
