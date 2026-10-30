from itertools import product

import pytest

from decision_making_pkg.control_policy import DriveCommand, DriveModeController, path_steering
from serial_communication_pkg.lib.protocol_convert_func_lib import convert_serial_message


def test_space_immediately_uses_a_fresh_command():
    controller = DriveModeController('autonomous')
    controller.receive_auto(DriveCommand(2, 255, 255), 10.0)
    assert controller.command(10.1) == DriveCommand()
    assert controller.handle_key(' ') == 'auto_started'
    assert controller.command(10.1) == DriveCommand(2, 255, 255)
    controller.receive_auto(DriveCommand(2, 255, 255), 10.2)
    assert controller.command(10.3) == DriveCommand(2, 255, 255)
    assert controller.command(10.8) == DriveCommand()
    controller.handle_key(' ')
    assert controller.command(10.3) == DriveCommand()


@pytest.mark.parametrize('command_time', [None, 9.0, 11.0])
def test_space_cannot_use_missing_stale_or_future_command(command_time):
    controller = DriveModeController('autonomous')
    if command_time is not None:
        controller.receive_auto(DriveCommand(2, 120, 120), command_time)
    controller.handle_key(' ')
    assert controller.command(10.0) == DriveCommand()


@pytest.mark.parametrize('key', ['1', '2', 'x', 'q', '\x03'])
def test_mode_switch_or_stop_discards_previous_motion(key):
    controller = DriveModeController('autonomous')
    controller.handle_key(' ')
    controller.receive_auto(DriveCommand(7, 120, 120), 1.0)
    controller.handle_key(key)
    assert not controller.active
    assert controller.command(1.1) == DriveCommand()


def test_manual_and_firmware_limits():
    controller = DriveModeController(manual_speed_limit=120)
    for _ in range(40):
        controller.handle_key('w')
        controller.handle_key('a')
    assert controller.command(0) == DriveCommand(-7, 120, 120)
    controller.handle_key('2')
    controller.handle_key('w')
    assert controller.command(0) == DriveCommand()
    controller.handle_key(' ')
    controller.receive_auto(DriveCommand(50, 1000, -1000), 0)
    assert controller.command(0.1) == DriveCommand(7, 255, -255)


def test_steering_direction_straight_and_limits():
    assert path_steering([(320, 5), (320, 179)]) == 0
    assert path_steering([(400, 5), (320, 179)]) > 0
    assert path_steering([(240, 5), (320, 179)]) < 0
    assert path_steering([(4000, 5), (320, 179)]) == 7
    assert path_steering([(400, 5), (320, 179)], sign=-1) < 0
    with pytest.raises(ValueError):
        path_steering([(float('nan'), 5), (320, 179)])
    with pytest.raises(ValueError):
        path_steering([(320, 180), (320, 200)])


def test_packets_fit_existing_arduino_buffer_in_both_directions():
    for steering, left, right in product([-100, -7, 0, 7, 100],
                                         [-500, -255, 0, 255, 500],
                                         [-500, -255, 0, 255, 500]):
        packet = convert_serial_message(steering, left, right)
        assert packet.endswith('\n')
        assert len(packet.rstrip('\n')) <= 14
    assert convert_serial_message(-100, -500, 500) == 's-7l-255r255\n'


def test_out_of_range_260_never_reaches_arduino():
    assert DriveCommand.bounded(0, 260, 260) == DriveCommand(0, 255, 255)
    assert convert_serial_message(0, 260, 260) == 's0l255r255\n'
