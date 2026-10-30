"""Publish stamped BGR camera frames at a configurable inference input rate."""

from pathlib import Path
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImagePublisherNode(Node):
    def __init__(self, **kwargs):
        super().__init__('image_publisher_node', **kwargs)
        self.source = self.declare_parameter('data_source', 'camera').value
        self.cam_num = self.declare_parameter('cam_num', 0).value
        self.img_dir = self.declare_parameter('img_dir', '').value
        self.video_path = self.declare_parameter('video_path', '').value
        self.show_image = self.declare_parameter('logger', True).value
        period = self.declare_parameter('timer', 0.1).value
        if period <= 0:
            raise ValueError('timer must be positive')
        self.bridge = CvBridge()
        self.cap = None
        self.index = 0
        self.files = []
        if self.source in ('camera', 'video'):
            self.cap = cv2.VideoCapture(self.cam_num if self.source == 'camera' else self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open {self.source}: {self.cam_num if self.source == "camera" else self.video_path}')
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.source == 'image':
            directory = Path(self.img_dir)
            self.files = sorted(p for p in directory.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
            if not self.files:
                raise RuntimeError(f'No images in {directory}')
        else:
            raise ValueError('data_source must be camera, video or image')
        self.publisher = self.create_publisher(Image, 'image_raw', 1)
        self.timer = self.create_timer(period, self.timer_callback)
        self.last_warning = -10.0
        self.get_logger().info(f'Image input: {self.source}, camera={self.cam_num}, period={period}s')

    def timer_callback(self):
        if self.source == 'image':
            frame = cv2.imread(str(self.files[self.index]))
            self.index = (self.index + 1) % len(self.files)
            ok = frame is not None
        else:
            ok, frame = self.cap.read()
            if not ok and self.source == 'video':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ok:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.last_warning > 2.0:
                self.get_logger().warning('No image frame; no new path can be generated')
                self.last_warning = now
            return
        frame = cv2.resize(frame, (640, 480))
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.publisher.publish(msg)
        if self.show_image:
            cv2.imshow('Camera Image', frame)
            cv2.waitKey(1)

    def close(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
