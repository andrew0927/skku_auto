import numpy as np

from camera_perception_pkg.lane_info_extractor_node import lane_at
from camera_perception_pkg.vehicle_detector_node import keep_box


def masks():
    lane1 = np.zeros((480, 640), np.uint8)
    lane2 = np.zeros((480, 640), np.uint8)
    lane1[250:480, 0:300] = 255     # left lane
    lane2[250:480, 330:640] = 255   # right lane (30 px centre line gap)
    return {'lane1': lane1, 'lane2': lane2}


def test_car_is_placed_by_the_road_strip_in_front_of_it():
    assert lane_at(masks(), (100, 300, 200, 380)) == 'lane1'
    assert lane_at(masks(), (420, 300, 560, 380)) == 'lane2'
    # Straddling the centre line, mostly over lane2.
    assert lane_at(masks(), (260, 300, 460, 380)) == 'lane2'
    # Half out of view at the left edge, standing off the road (above the lane masks): unknown.
    assert lane_at(masks(), (0, 150, 40, 230)) is None
    # Right in front: no road strip visible below the box.
    assert lane_at(masks(), (200, 300, 400, 479)) is None
    assert lane_at({}, (100, 300, 200, 380)) is None


def test_own_hood_is_not_a_car():
    assert not keep_box((200, 455, 480, 480))       # flat box on the bonnet
    assert keep_box((250, 300, 380, 420))           # a car ahead
    assert keep_box((250, 380, 400, 479))           # a close car reaching the bottom


def road_with_centre_line(x_line=320):
    import numpy as np
    img = np.full((480, 640, 3), (40, 40, 40), np.uint8)          # dark road
    img[:, :60] = img[:, 580:] = (60, 160, 60)                     # green mat at the sides
    for y in range(250, 480, 40):                                  # dashes, 20 rows on 20 off
        img[y:y + 20, x_line - 5:x_line + 5] = (235, 235, 235)
    return img


def test_centre_line_tells_which_lane_a_car_is_in():
    from camera_perception_pkg.center_line import car_side
    img = road_with_centre_line(320)
    assert car_side(img, (150, 240, 280, 320)) == 'lane1'     # left of the dashes
    assert car_side(img, (360, 240, 490, 320)) == 'lane2'     # right of them
    assert car_side(img, (260, 240, 380, 320)) == 'line'      # on top of the line
    flat = road_with_centre_line(320); flat[:, 300:340] = (40, 40, 40)
    assert car_side(flat, (150, 240, 280, 320)) is None       # no dashes: unknown


def test_centre_line_far_from_the_car_is_not_used():
    from camera_perception_pkg.center_line import car_side
    img = road_with_centre_line(320)
    assert car_side(img, (0, 255, 22, 334)) is None            # tiny car at the image edge


def test_lane_under_car_reads_the_mask_above_the_hood():
    import numpy as np
    from camera_perception_pkg.lane_info_extractor_node import lane_under_car
    lane1 = np.zeros((480, 640), np.uint8); lane1[:, :300] = 255
    lane2 = np.zeros((480, 640), np.uint8); lane2[:, 300:] = 255
    assert lane_under_car({'lane1': lane1, 'lane2': lane2}) is None      # on the line between
    lane2[:, 260:] = 255; lane1[:, 260:] = 0
    assert lane_under_car({'lane1': lane1, 'lane2': lane2}) == 'lane2'


def test_close_car_beside_us_is_not_a_reason_to_change_lanes():
    import rclpy
    from types import SimpleNamespace
    from interfaces_pkg.msg import DetectionArray, Detection, BoundingBox2D
    from camera_perception_pkg.lane_info_extractor_node import Yolov8InfoExtractor
    rclpy.init()
    try:
        node = Yolov8InfoExtractor(parameter_overrides=[rclpy.parameter.Parameter('show_image', value=False)])
        got = []
        node.own_pub = SimpleNamespace(publish=lambda m: got.append(m.data))
        def car(x0, y0, x1, y1, side):
            d = Detection(class_name='car', id=side); d.bbox = BoundingBox2D()
            d.bbox.center.position.x, d.bbox.center.position.y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            d.bbox.size.x, d.bbox.size.y = float(x1 - x0), float(y1 - y0)
            return d
        for boxes, want in ((((0, 288, 115, 454),), False), (((250, 250, 400, 320),), True)):
            msg = DetectionArray(); msg.detections = [car(*b, 'lane2') for b in boxes]
            node.vehicle_callback(msg)
            assert got[-1] is want
        node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_car_on_the_line_at_the_image_edge_uses_the_lane_mask():
    import numpy as np, rclpy
    from types import SimpleNamespace
    from interfaces_pkg.msg import DetectionArray, Detection, BoundingBox2D
    from camera_perception_pkg.lane_info_extractor_node import Yolov8InfoExtractor
    rclpy.init()
    try:
        node = Yolov8InfoExtractor(parameter_overrides=[rclpy.parameter.Parameter('show_image', value=False)])
        got = []
        node.own_pub = SimpleNamespace(publish=lambda m: got.append(m.data))
        lane1 = np.zeros((480, 640), np.uint8); lane1[:, :200] = 255   # lane1 on the left
        lane2 = np.zeros((480, 640), np.uint8); lane2[:, 200:] = 255
        node.masks = {'lane1': lane1, 'lane2': lane2}
        d = Detection(class_name='car', id='line'); d.bbox = BoundingBox2D()
        d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size.x, d.bbox.size.y = 45.0, 300.0, 90.0, 50.0
        msg = DetectionArray(); msg.detections = [d]
        node.vehicle_callback(msg)          # 11-19 225.8 s: a lane1 car read "line" -> not ours (lane2)
        assert got[-1] is False
        node.destroy_node()
    finally:
        rclpy.try_shutdown()

