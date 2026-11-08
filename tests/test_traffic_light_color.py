from types import SimpleNamespace

import numpy as np

from camera_perception_pkg.lib.camera_perception_func_lib import get_traffic_light_color

HSV = {
    'red1': (np.array([0, 100, 95]), np.array([10, 255, 255])),
    'red2': (np.array([160, 100, 95]), np.array([179, 255, 255])),
    'yellow': (np.array([20, 100, 95]), np.array([30, 255, 255])),
    'green': (np.array([40, 100, 95]), np.array([90, 255, 255])),
}


def box(cx, cy, w, h):
    return SimpleNamespace(center=SimpleNamespace(position=SimpleNamespace(x=cx, y=cy)),
                           size=SimpleNamespace(x=w, y=h))


def test_unlit_light_is_not_red():
    image = np.zeros((480, 640, 3), np.uint8)
    assert get_traffic_light_color(image, box(100, 100, 40, 80), HSV) == 'Unknown'


def test_lit_colors_and_out_of_frame_box():
    image = np.zeros((480, 640, 3), np.uint8)
    image[70:90, 90:110] = (0, 0, 255)
    assert get_traffic_light_color(image, box(100, 100, 40, 80), HSV) == 'Red'
    image[70:90, 90:110] = (0, 255, 0)
    assert get_traffic_light_color(image, box(100, 100, 40, 80), HSV) == 'Green'
    assert get_traffic_light_color(image, box(-100, -100, 40, 80), HSV) == 'Unknown'
