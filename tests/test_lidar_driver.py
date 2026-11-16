"""Driver startup and unplug timeouts using a fake serial input buffer."""

import logging
from types import SimpleNamespace as NS

import pytest

from lidar_perception_pkg.lib import lidar_perception_func_lib as module


def driver():
    node = object.__new__(module.RPLidar)
    node.scanning = [False, 0, 'normal']
    node.logger = logging.getLogger('fake_lidar')
    node.timeout = 0.3
    return node


def test_start_discards_previous_session_bytes_before_health_request(monkeypatch):
    node, buffer = driver(), NS(size=32)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    commands = []
    node._serial = NS(inWaiting=lambda: buffer.size,
                      flushInput=lambda: setattr(buffer, 'size', 0))
    node._send_cmd = commands.append

    def health():
        assert buffer.size == 0
        return 'Good', 0

    node.get_health = health
    normal = module._SCAN_TYPE['normal']
    node._read_descriptor = lambda: (normal['size'], False, normal['response'])
    node.start()
    assert commands == [module.STOP_BYTE, normal['byte']]
    assert node.scanning == [True, normal['size'], 'normal']


def test_busy_health_raises_a_driver_error_instead_of_returning_a_string():
    node = driver()
    node._serial = NS(inWaiting=lambda: 10)
    with pytest.raises(module.RPLidarException, match='buffer'):
        node.get_health()


def test_unplugged_sensor_cannot_block_response_read_forever(monkeypatch):
    node = driver()
    node._serial = NS(inWaiting=lambda: 0)
    times = iter([10.0, 10.4])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(times))
    with pytest.raises(module.RPLidarException, match='Timed out'):
        node._read_response(5)
