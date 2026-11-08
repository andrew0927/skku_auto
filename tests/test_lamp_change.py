import numpy as np

from camera_perception_pkg.lamp_change import LampChangeDetector


def housing(red_on, green_on, green_bgr=(255, 255, 255)):
    image = np.zeros((480, 640, 3), np.uint8)
    image[100:120, 100:160] = (20, 20, 20)          # dark housing
    if red_on:
        image[104:116, 104:116] = (0, 0, 255)       # left lamp red
    if green_on:
        image[104:116, 144:156] = green_bgr         # right lamp lit
    return image


BOX = (100, 100, 160, 120)


def test_dark_lamp_lighting_up_after_red_is_reported_as_green():
    detector = LampChangeDetector()
    red = housing(True, False)
    assert detector.update(red, BOX, 'Red', stopped=False, now=0.0) == 'Red'
    assert detector.reference is None
    assert detector.update(red, BOX, 'Red', stopped=True, now=0.0) == 'Red'
    assert detector.reference is None, 'the car still rocks right after stopping'
    assert detector.update(red, BOX, 'Red', stopped=True, now=1.0) == 'Red'
    assert detector.reference is not None
    # Still red: an Unknown frame does not become green.
    assert detector.update(red, BOX, 'Unknown', stopped=True, now=1.1) == 'Unknown'
    # Red lamp off, right lamp lit but washed out white, box not even detected: light changed.
    assert detector.update(housing(False, True), None, 'None', stopped=True, now=1.5) == 'Green'
    # A red lamp that merely moved (box jitter) is not a change.
    shifted = housing(False, False)
    shifted[104:116, 144:156] = (0, 0, 255)
    assert detector.update(shifted, BOX, 'Unknown', stopped=True, now=1.6) == 'Unknown'
    # Leaving the stop clears the reference.
    assert detector.update(housing(False, True), BOX, 'Unknown', stopped=False, now=2.0) == 'Unknown'
    assert detector.reference is None


def test_all_dark_or_whole_scene_change_is_not_green():
    detector = LampChangeDetector()
    detector.update(housing(True, False), BOX, 'Red', stopped=True, now=0.0)
    detector.update(housing(True, False), BOX, 'Red', stopped=True, now=1.0)
    assert detector.update(housing(False, False), BOX, 'Unknown', stopped=True, now=1.2) == 'Unknown'
    bright = np.full((480, 640, 3), 200, np.uint8)  # camera moved or exposure jumped
    assert detector.update(bright, BOX, 'Unknown', stopped=True, now=1.4) == 'Unknown'
    assert detector.update(bright, BOX, 'Unknown', stopped=True, now=1.5) == 'Unknown'


def test_green_lamp_detected_even_when_the_red_lamp_is_out_of_view():
    image = np.zeros((480, 640, 3), np.uint8)
    image[100:120, 0:30] = (20, 20, 20)             # only the right end of the light is visible
    box = (0, 100, 30, 120)
    detector = LampChangeDetector()
    detector.update(image, box, 'Unknown', stopped=True, now=0.0)
    detector.update(image, box, 'Unknown', stopped=True, now=1.0)
    lit = image.copy()
    lit[104:116, 10:22] = (200, 255, 200)           # pale bluish-green LED
    assert detector.update(lit, None, 'Unknown', stopped=True, now=1.3) == 'Green'


def test_lamps_read_from_pixels_while_yolo_misses_the_light():
    image = np.zeros((480, 640, 3), np.uint8)
    image[100:140, 100:220] = (20, 20, 20)
    image[105:135, 105:135] = (0, 0, 255)            # red lamp lit
    box = (100, 100, 220, 140)
    detector = LampChangeDetector()
    assert detector.update(image, box, 'Red', stopped=False, now=0.0) == 'Red'
    # Stopped: YOLO reports no light at all, the pixels still say red.
    for k in range(10):
        assert detector.update(image, None, 'None', stopped=True, now=0.5 + 0.1 * k) == 'Red'
    image[105:135, 105:135] = (20, 20, 20)
    image[105:135, 185:215] = (150, 255, 120)        # green lamp lit
    assert detector.update(image, None, 'None', stopped=True, now=2.0) == 'Green'
