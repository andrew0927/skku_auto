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
    # Still red: the lit red lamp is read inside the box even when the colour check said Unknown.
    assert detector.update(red, BOX, 'Unknown', stopped=True, now=1.1) == 'Red'
    # Red lamp off, right lamp lit but washed out white, box not even detected: light changed.
    assert detector.update(housing(False, True), None, 'None', stopped=True, now=1.5) == 'Green'
    # A red lamp that merely moved (box jitter) is not a change.
    shifted = housing(False, False)
    shifted[104:116, 144:156] = (0, 0, 255)
    assert detector.update(shifted, BOX, 'Unknown', stopped=True, now=1.6) == 'Red'
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
    image[90:150, 100:220] = (20, 20, 20)
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



def test_amber_lamp_is_yellow_and_round_lamps_only():
    import cv2
    from camera_perception_pkg.lamp_change import find_lamp_blob, read_lamps
    image = np.zeros((480, 640, 3), np.uint8)
    image[90:150, 100:220] = (20, 20, 20)
    cv2.circle(image, (160, 120), 22, (0, 170, 255), -1)        # amber middle lamp (close)
    assert read_lamps(image, (100, 100, 220, 140))[0] == 'Yellow'
    assert find_lamp_blob(image)[0] == 'Yellow'
    bench = np.zeros((480, 640, 3), np.uint8)
    bench[150:200, 50:300] = (0, 120, 255)                      # orange bench: not round
    assert find_lamp_blob(bench) == (None, None)
    lamp = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(lamp, (60, 80), 22, (0, 0, 255), -1)             # lit red lamp, YOLO missed it
    color, box = find_lamp_blob(lamp)
    assert color == 'Red' and box[0] <= 42 and box[2] > 150      # housing extends to the right


def test_dull_orange_block_is_not_an_amber_lamp():
    # The orange bench seen on 11-18: bright (V~220) and orange, roughly square, but no glowing core.
    from camera_perception_pkg.lamp_change import find_lamp_blob
    bench = np.zeros((480, 640, 3), np.uint8)
    bench[180:230, 100:160] = (40, 130, 220)
    assert find_lamp_blob(bench) == (None, None)


def test_vending_machine_screen_is_not_a_lamp():
    import cv2
    from camera_perception_pkg.lamp_change import find_lamp_blob
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(image, (130, 176), 7, (0, 0, 255), -1)          # small lit red screen, area ~150
    assert find_lamp_blob(image) == (None, None)


def test_lamp_cut_by_the_image_edge_is_found():
    import cv2
    from camera_perception_pkg.lamp_change import find_lamp_blob
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(image, (10, 90), 60, (0, 0, 255), -1)            # half outside the image
    cv2.circle(image, (10, 90), 25, (255, 255, 255), -1)        # saturated core
    assert find_lamp_blob(image)[0] == 'Red'

