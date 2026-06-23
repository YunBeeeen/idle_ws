"""Publish HSV statistics from a sampled image ROI for threshold tuning."""

from __future__ import annotations

import json
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

from .color_segmentation import parse_hsv_ranges_json
from .vision_utils import stamp_ns


def _clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    percentiles = np.percentile(values, [5, 10, 50, 90, 95], axis=0)
    return {
        "min": [int(x) for x in values.min(axis=0)],
        "p05": [int(round(x)) for x in percentiles[0]],
        "p10": [int(round(x)) for x in percentiles[1]],
        "median": [int(round(x)) for x in percentiles[2]],
        "p90": [int(round(x)) for x in percentiles[3]],
        "p95": [int(round(x)) for x in percentiles[4]],
        "max": [int(x) for x in values.max(axis=0)],
    }


class HsvTunerNode(Node):
    """Inspect HSV values around a selected image point."""

    def __init__(self) -> None:
        super().__init__("hsv_tuner_node")
        self._bridge = CvBridge()

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("sample_u", -1)
        self.declare_parameter("sample_v", -1)
        self.declare_parameter("roi_half_size_px", 12)
        self.declare_parameter("publish_every_n", 5)
        self.declare_parameter("h_margin", 8)
        self.declare_parameter("s_margin", 40)
        self.declare_parameter("v_margin", 40)
        self.declare_parameter("hsv_ranges_json", "")
        self.declare_parameter("status_topic", "/idle_vision/hsv_tuner/status")
        self.declare_parameter("debug_topic", "/idle_vision/hsv_tuner/debug_image")
        self.declare_parameter("mask_topic", "/idle_vision/hsv_tuner/mask")
        self.declare_parameter("publish_debug", True)

        self._color_topic = str(self.get_parameter("color_topic").value)
        self._sample_u = int(self.get_parameter("sample_u").value)
        self._sample_v = int(self.get_parameter("sample_v").value)
        self._roi_half_size_px = int(self.get_parameter("roi_half_size_px").value)
        self._publish_every_n = int(self.get_parameter("publish_every_n").value)
        self._h_margin = int(self.get_parameter("h_margin").value)
        self._s_margin = int(self.get_parameter("s_margin").value)
        self._v_margin = int(self.get_parameter("v_margin").value)
        self._hsv_ranges_json = str(self.get_parameter("hsv_ranges_json").value)
        self._publish_debug = bool(self.get_parameter("publish_debug").value)
        self._hsv_ranges = (
            parse_hsv_ranges_json(self._hsv_ranges_json)
            if self._hsv_ranges_json.strip()
            else []
        )

        if self._roi_half_size_px < 0:
            raise ValueError("roi_half_size_px must be >= 0")
        if self._publish_every_n <= 0:
            raise ValueError("publish_every_n must be > 0")

        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self._debug_pub = None
        self._mask_pub = None
        if self._publish_debug:
            self._debug_pub = self.create_publisher(
                Image,
                str(self.get_parameter("debug_topic").value),
                qos_profile_sensor_data,
            )
            self._mask_pub = self.create_publisher(
                Image,
                str(self.get_parameter("mask_topic").value),
                qos_profile_sensor_data,
            )

        self._frames = 0
        self.create_subscription(
            Image,
            self._color_topic,
            self._on_color,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "HSV tuner ready: color=%s sample=(%d,%d) roi_half=%d"
            % (
                self._color_topic,
                self._sample_u,
                self._sample_v,
                self._roi_half_size_px,
            )
        )

    def _on_color(self, msg: Image) -> None:
        self._frames += 1
        if self._frames % self._publish_every_n != 0:
            return

        try:
            color_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"color conversion failed: {exc}")
            return

        h, w = color_bgr.shape[:2]
        u = self._sample_u if self._sample_u >= 0 else w // 2
        v = self._sample_v if self._sample_v >= 0 else h // 2
        u = _clip_int(u, 0, w - 1)
        v = _clip_int(v, 0, h - 1)

        r = self._roi_half_size_px
        x0 = _clip_int(u - r, 0, w - 1)
        x1 = _clip_int(u + r + 1, 1, w)
        y0 = _clip_int(v - r, 0, h - 1)
        y1 = _clip_int(v + r + 1, 1, h)

        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        roi_values = hsv[y0:y1, x0:x1].reshape(-1, 3)
        hsv_stats = _stats(roi_values)
        suggested = self._suggest_range(hsv_stats)

        mask = self._make_mask(hsv)
        mask_area_px = int(np.count_nonzero(mask)) if mask is not None else 0
        mask_ratio = float(mask_area_px / float(w * h)) if mask is not None else 0.0

        payload = {
            "stamp_ns": stamp_ns(msg) or time.time_ns(),
            "frame": self._frames,
            "image_size": [int(w), int(h)],
            "sample_px": [int(u), int(v)],
            "roi_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "roi_half_size_px": int(r),
            "hsv": hsv_stats,
            "suggested_hsv_ranges_json": json.dumps([suggested]) if suggested else "",
            "active_hsv_ranges": [
                [list(lower), list(upper)] for lower, upper in self._hsv_ranges
            ],
            "mask_area_px": mask_area_px,
            "mask_ratio": mask_ratio,
        }
        out = String()
        out.data = json.dumps(payload, sort_keys=True)
        self._status_pub.publish(out)

        self._publish_debug_images(msg, color_bgr, mask, (x0, y0, x1, y1), (u, v), suggested)

    def _suggest_range(self, hsv_stats: dict) -> list[int]:
        if not hsv_stats:
            return []
        low = hsv_stats["p05"]
        high = hsv_stats["p95"]
        return [
            _clip_int(low[0] - self._h_margin, 0, 180),
            _clip_int(low[1] - self._s_margin, 0, 255),
            _clip_int(low[2] - self._v_margin, 0, 255),
            _clip_int(high[0] + self._h_margin, 0, 180),
            _clip_int(high[1] + self._s_margin, 0, 255),
            _clip_int(high[2] + self._v_margin, 0, 255),
        ]

    def _make_mask(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        if not self._hsv_ranges:
            return None
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self._hsv_ranges:
            mask |= cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
        return mask

    def _publish_debug_images(
        self,
        msg: Image,
        color_bgr: np.ndarray,
        mask: Optional[np.ndarray],
        roi_xyxy: tuple[int, int, int, int],
        sample_px: tuple[int, int],
        suggested: list[int],
    ) -> None:
        if not self._publish_debug:
            return

        if mask is not None and self._mask_pub is not None:
            mask_msg = self._bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = msg.header
            self._mask_pub.publish(mask_msg)

        if self._debug_pub is None:
            return

        x0, y0, x1, y1 = roi_xyxy
        u, v = sample_px
        debug = color_bgr.copy()
        cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
        cv2.drawMarker(
            debug,
            (u, v),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        label = "HSV ROI"
        if suggested:
            label += f" {suggested}"
        cv2.putText(
            debug,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header = msg.header
        self._debug_pub.publish(debug_msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = HsvTunerNode()
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
