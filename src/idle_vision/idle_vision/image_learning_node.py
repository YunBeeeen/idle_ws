"""Bridge RealSense image topics into fixed-size learning streams."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .vision_utils import stamp_ns as message_stamp_ns


def _image_encoding_for_depth(array: np.ndarray, fallback: str) -> str:
    if fallback:
        return fallback
    if array.dtype == np.uint16:
        return "16UC1"
    if array.dtype == np.float32:
        return "32FC1"
    if array.dtype == np.uint8:
        return "8UC1"
    return "passthrough"


class ImageLearningNode(Node):
    """Subscribe to camera images, resize them, republish, and optionally save."""

    def __init__(self) -> None:
        super().__init__("image_learning_node")
        self._bridge = CvBridge()

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("output_color_topic", "/idle_vision/learning/color/image_raw")
        self.declare_parameter("output_depth_topic", "/idle_vision/learning/depth/image_raw")
        self.declare_parameter("status_topic", "/idle_vision/learning/status")
        self.declare_parameter("enable_depth", True)
        self.declare_parameter("publish_processed", True)
        self.declare_parameter("resize_width", 640)
        self.declare_parameter("resize_height", 480)
        self.declare_parameter("output_color_encoding", "bgr8")
        self.declare_parameter("mirror", False)
        self.declare_parameter("save_dir", "")
        self.declare_parameter("save_every_n", 0)
        self.declare_parameter("depth_match_timeout_s", 0.25)
        self.declare_parameter("log_every_n", 30)

        self._color_topic = str(self.get_parameter("color_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._output_color_topic = str(self.get_parameter("output_color_topic").value)
        self._output_depth_topic = str(self.get_parameter("output_depth_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._enable_depth = bool(self.get_parameter("enable_depth").value)
        self._publish_processed = bool(self.get_parameter("publish_processed").value)
        self._resize_width = int(self.get_parameter("resize_width").value)
        self._resize_height = int(self.get_parameter("resize_height").value)
        self._output_color_encoding = str(self.get_parameter("output_color_encoding").value)
        self._mirror = bool(self.get_parameter("mirror").value)
        self._save_dir_text = str(self.get_parameter("save_dir").value)
        self._save_every_n = int(self.get_parameter("save_every_n").value)
        self._depth_match_timeout_s = float(self.get_parameter("depth_match_timeout_s").value)
        self._log_every_n = int(self.get_parameter("log_every_n").value)

        if self._output_color_encoding not in ("bgr8", "rgb8"):
            self.get_logger().warning(
                "output_color_encoding should normally be bgr8 or rgb8; using bgr8"
            )
            self._output_color_encoding = "bgr8"
        if self._resize_width < 0 or self._resize_height < 0:
            raise ValueError("resize_width and resize_height must be >= 0")
        if self._save_every_n < 0:
            raise ValueError("save_every_n must be >= 0")

        self._color_count = 0
        self._depth_count = 0
        self._last_depth: Optional[np.ndarray] = None
        self._last_depth_encoding = ""
        self._last_depth_stamp_ns = 0
        self._last_depth_saved_path = ""

        self._save_root: Optional[Path] = None
        self._manifest_path: Optional[Path] = None
        if self._save_dir_text:
            self._save_root = Path(self._save_dir_text).expanduser()
            (self._save_root / "color").mkdir(parents=True, exist_ok=True)
            (self._save_root / "depth").mkdir(parents=True, exist_ok=True)
            self._manifest_path = self._save_root / "manifest.csv"
            if not self._manifest_path.exists():
                with self._manifest_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "stamp_ns",
                            "color_path",
                            "depth_path",
                            "width",
                            "height",
                            "depth_age_s",
                        ]
                    )

        self._color_pub = None
        self._depth_pub = None
        if self._publish_processed:
            self._color_pub = self.create_publisher(
                Image,
                self._output_color_topic,
                qos_profile_sensor_data,
            )
            if self._enable_depth:
                self._depth_pub = self.create_publisher(
                    Image,
                    self._output_depth_topic,
                    qos_profile_sensor_data,
                )
        self._status_pub = self.create_publisher(String, self._status_topic, 10)

        self.create_subscription(
            Image,
            self._color_topic,
            self._on_color,
            qos_profile_sensor_data,
        )
        if self._enable_depth:
            self.create_subscription(
                Image,
                self._depth_topic,
                self._on_depth,
                qos_profile_sensor_data,
            )

        self.get_logger().info(
            "image bridge ready: color=%s depth=%s -> %s %s"
            % (
                self._color_topic,
                self._depth_topic if self._enable_depth else "disabled",
                self._output_color_topic,
                self._output_depth_topic if self._enable_depth else "disabled",
            )
        )

    def _resize(self, image: np.ndarray, *, interpolation: int) -> np.ndarray:
        if self._resize_width == 0 or self._resize_height == 0:
            return image
        return cv2.resize(
            image,
            (self._resize_width, self._resize_height),
            interpolation=interpolation,
        )

    def _copy_header(self, out: Image, src: Image) -> None:
        out.header.stamp = src.header.stamp
        out.header.frame_id = src.header.frame_id

    def _on_color(self, msg: Image) -> None:
        self._color_count += 1
        try:
            color_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"color conversion failed: {exc}")
            return

        color_bgr = self._resize(color_bgr, interpolation=cv2.INTER_AREA)
        if self._mirror:
            color_bgr = cv2.flip(color_bgr, 1)

        publish_image = color_bgr
        if self._output_color_encoding == "rgb8":
            publish_image = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        if self._color_pub is not None:
            out_msg = self._bridge.cv2_to_imgmsg(
                publish_image,
                encoding=self._output_color_encoding,
            )
            self._copy_header(out_msg, msg)
            self._color_pub.publish(out_msg)

        if self._save_every_n > 0 and self._color_count % self._save_every_n == 0:
            self._save_sample(msg, color_bgr)

        if self._log_every_n > 0 and self._color_count % self._log_every_n == 0:
            self.get_logger().info(
                "frames color=%d depth=%d size=%dx%d"
                % (
                    self._color_count,
                    self._depth_count,
                    color_bgr.shape[1],
                    color_bgr.shape[0],
                )
            )

        self._publish_status(msg, color_bgr)

    def _on_depth(self, msg: Image) -> None:
        self._depth_count += 1
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warning(f"depth conversion failed: {exc}")
            return

        depth = self._resize(depth, interpolation=cv2.INTER_NEAREST)
        if self._mirror:
            depth = cv2.flip(depth, 1)

        self._last_depth = depth.copy()
        self._last_depth_encoding = msg.encoding
        self._last_depth_stamp_ns = message_stamp_ns(msg) or time.time_ns()

        if self._depth_pub is not None:
            encoding = _image_encoding_for_depth(depth, msg.encoding)
            out_msg = self._bridge.cv2_to_imgmsg(depth, encoding=encoding)
            self._copy_header(out_msg, msg)
            self._depth_pub.publish(out_msg)

    def _save_sample(self, msg: Image, color_bgr: np.ndarray) -> None:
        if self._save_root is None or self._manifest_path is None:
            return

        stamp_ns = message_stamp_ns(msg) or time.time_ns()
        color_path = self._save_root / "color" / f"{stamp_ns:019d}.jpg"
        if not cv2.imwrite(str(color_path), color_bgr):
            self.get_logger().warning(f"failed to save color image: {color_path}")
            return

        depth_path_text = ""
        depth_age_s = ""
        if self._last_depth is not None:
            depth_age = abs(stamp_ns - self._last_depth_stamp_ns) / 1_000_000_000.0
            if depth_age <= self._depth_match_timeout_s:
                depth_path = self._save_root / "depth" / f"{stamp_ns:019d}"
                if self._last_depth.dtype == np.uint16:
                    depth_file = depth_path.with_suffix(".png")
                    if cv2.imwrite(str(depth_file), self._last_depth):
                        depth_path_text = str(depth_file)
                else:
                    depth_file = depth_path.with_suffix(".npy")
                    np.save(str(depth_file), self._last_depth)
                    depth_path_text = str(depth_file)
                depth_age_s = f"{depth_age:.6f}"

        with self._manifest_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    stamp_ns,
                    str(color_path),
                    depth_path_text,
                    color_bgr.shape[1],
                    color_bgr.shape[0],
                    depth_age_s,
                ]
            )

    def _publish_status(self, msg: Image, color_bgr: np.ndarray) -> None:
        stamp_ns = message_stamp_ns(msg) or time.time_ns()
        depth_age_s = None
        if self._last_depth_stamp_ns:
            depth_age_s = abs(stamp_ns - self._last_depth_stamp_ns) / 1_000_000_000.0

        status = {
            "stamp_ns": stamp_ns,
            "color_frames": self._color_count,
            "depth_frames": self._depth_count,
            "width": int(color_bgr.shape[1]),
            "height": int(color_bgr.shape[0]),
            "depth_age_s": depth_age_s,
            "save_dir": self._save_dir_text,
            "save_every_n": self._save_every_n,
        }
        out = String()
        out.data = json.dumps(status, sort_keys=True)
        self._status_pub.publish(out)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ImageLearningNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
