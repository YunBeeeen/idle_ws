"""Extract depth for one visible object mask.

The first detector is intentionally simple: HSV color thresholding for colored
blocks. The depth/statistics/output side is kept separate enough that a learned
segmenter can replace only the mask-making step later.
"""

from __future__ import annotations

import json
import math
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String

from .color_segmentation import (
    contour_color_stats,
    make_color_mask,
    passes_color_quality,
    resolve_detector_ranges,
)
from .vision_utils import depth_values_to_meters, stamp_ns


def _finite_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


class ObjectDepthNode(Node):
    """Find one target-colored object and publish its masked depth."""

    def __init__(self) -> None:
        super().__init__("object_depth_node")
        self._bridge = CvBridge()

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("target_color", "auto")
        self.declare_parameter("hsv_ranges_json", "")
        self.declare_parameter("min_area_px", 500)
        self.declare_parameter("max_area_px", 0)
        self.declare_parameter("morph_kernel_size", 5)
        self.declare_parameter("depth_erode_px", 3)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)
        self.declare_parameter("depth_percentile", 50.0)
        self.declare_parameter("depth_match_timeout_s", 0.25)
        self.declare_parameter("use_color_ratio_mask", True)
        self.declare_parameter("use_depth_candidate_mask", True)
        self.declare_parameter("publish_nan_on_miss", True)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("depth_m_topic", "/idle_vision/object_depth/depth_m")
        self.declare_parameter("point_topic", "/idle_vision/object_depth/point")
        self.declare_parameter("bbox_topic", "/idle_vision/object_depth/bbox")
        self.declare_parameter("mask_topic", "/idle_vision/object_depth/mask")
        self.declare_parameter("debug_topic", "/idle_vision/object_depth/debug_image")
        self.declare_parameter("status_topic", "/idle_vision/object_depth/status")

        self._color_topic = str(self.get_parameter("color_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._target_color = str(self.get_parameter("target_color").value)
        self._hsv_ranges_json = str(self.get_parameter("hsv_ranges_json").value)
        self._min_area_px = int(self.get_parameter("min_area_px").value)
        self._max_area_px = int(self.get_parameter("max_area_px").value)
        self._morph_kernel_size = int(self.get_parameter("morph_kernel_size").value)
        self._depth_erode_px = int(self.get_parameter("depth_erode_px").value)
        self._depth_scale = float(self.get_parameter("depth_scale").value)
        self._min_depth_m = float(self.get_parameter("min_depth_m").value)
        self._max_depth_m = float(self.get_parameter("max_depth_m").value)
        self._depth_percentile = float(self.get_parameter("depth_percentile").value)
        self._depth_match_timeout_s = float(
            self.get_parameter("depth_match_timeout_s").value
        )
        self._use_color_ratio_mask = bool(
            self.get_parameter("use_color_ratio_mask").value
        )
        self._use_depth_candidate_mask = bool(
            self.get_parameter("use_depth_candidate_mask").value
        )
        self._publish_nan_on_miss = bool(self.get_parameter("publish_nan_on_miss").value)
        self._publish_debug = bool(self.get_parameter("publish_debug").value)

        self._detector_ranges = resolve_detector_ranges(
            target_color=self._target_color,
            hsv_ranges_json=self._hsv_ranges_json,
        )
        self._detector_names = list(self._detector_ranges.keys())

        if self._min_area_px < 0:
            raise ValueError("min_area_px must be >= 0")
        if self._max_area_px < 0:
            raise ValueError("max_area_px must be >= 0")
        if self._morph_kernel_size < 0:
            raise ValueError("morph_kernel_size must be >= 0")
        if self._depth_erode_px < 0:
            raise ValueError("depth_erode_px must be >= 0")
        if not 0.0 <= self._depth_percentile <= 100.0:
            raise ValueError("depth_percentile must be between 0 and 100")

        self._last_depth: Optional[np.ndarray] = None
        self._last_depth_encoding = ""
        self._last_depth_stamp_ns = 0
        self._camera_info: Optional[CameraInfo] = None
        self._frames = 0

        self._depth_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("depth_m_topic").value),
            10,
        )
        self._point_pub = self.create_publisher(
            PointStamped,
            str(self.get_parameter("point_topic").value),
            10,
        )
        self._bbox_pub = self.create_publisher(
            String,
            str(self.get_parameter("bbox_topic").value),
            10,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self._mask_pub = None
        self._debug_pub = None
        if self._publish_debug:
            self._mask_pub = self.create_publisher(
                Image,
                str(self.get_parameter("mask_topic").value),
                qos_profile_sensor_data,
            )
            self._debug_pub = self.create_publisher(
                Image,
                str(self.get_parameter("debug_topic").value),
                qos_profile_sensor_data,
            )

        self.create_subscription(
            Image,
            self._color_topic,
            self._on_color,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self._depth_topic,
            self._on_depth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "object depth ready: target=%s detectors=%s color=%s depth=%s camera_info=%s"
            % (
                self._target_color,
                ",".join(self._detector_names),
                self._color_topic,
                self._depth_topic,
                self._camera_info_topic,
            )
        )

    def _on_depth(self, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warning(f"depth conversion failed: {exc}")
            return

        self._last_depth = depth.copy()
        self._last_depth_encoding = msg.encoding
        self._last_depth_stamp_ns = stamp_ns(msg) or time.time_ns()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _on_color(self, msg: Image) -> None:
        self._frames += 1
        try:
            color_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"color conversion failed: {exc}")
            return

        detection = self._detect_object(msg, color_bgr)
        if detection is None:
            empty_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            self._publish_miss(msg, color_bgr, empty_mask, reason="no_object")
            return
        detected_color, mask, contour, candidates = detection

        object_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)

        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        center_u, center_v = self._contour_center(contour, x, y, w, h)

        depth_info = self._masked_depth(msg, object_mask)
        if depth_info is None:
            bbox = {
                "found": True,
                "target_color": self._target_color,
                "detected_color": detected_color,
                "bbox": [int(x), int(y), int(w), int(h)],
                "center_px": [float(center_u), float(center_v)],
                "area_px": area,
                "depth_m": None,
                "candidates": candidates,
            }
            self._publish_bbox(bbox)
            self._publish_miss(
                msg,
                color_bgr,
                object_mask,
                reason="no_valid_depth",
                bbox=(x, y, w, h),
                center=(center_u, center_v),
                detected_color=detected_color,
                candidates=candidates,
            )
            return

        depth_m, depth_min_m, depth_max_m, depth_count, depth_age_s = depth_info
        point = self._deproject(msg, center_u, center_v, depth_m)

        depth_msg = Float32()
        depth_msg.data = float(depth_m)
        self._depth_pub.publish(depth_msg)
        if point is not None:
            self._point_pub.publish(point)

        bbox = {
            "found": True,
            "target_color": self._target_color,
            "detected_color": detected_color,
            "bbox": [int(x), int(y), int(w), int(h)],
            "center_px": [float(center_u), float(center_v)],
            "area_px": area,
            "depth_m": float(depth_m),
            "depth_min_m": float(depth_min_m),
            "depth_max_m": float(depth_max_m),
            "depth_count": int(depth_count),
            "depth_age_s": float(depth_age_s),
            "candidates": candidates,
            "point_xyz_m": (
                [point.point.x, point.point.y, point.point.z] if point is not None else None
            ),
        }
        self._publish_bbox(bbox)
        self._publish_status(msg, bbox)
        self._publish_debug_images(
            msg,
            color_bgr,
            object_mask,
            bbox=(x, y, w, h),
            center=(center_u, center_v),
            depth_m=depth_m,
            detected_color=detected_color,
        )

    def _detect_object(self, color_msg: Image, color_bgr: np.ndarray):
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        best = None
        candidates = []
        depth_candidate_mask = self._depth_candidate_mask(color_msg, color_bgr.shape[:2])
        for color_name, ranges in self._detector_ranges.items():
            mask = self._make_mask(color_bgr, hsv, color_name, ranges)
            if depth_candidate_mask is not None:
                mask = cv2.bitwise_and(mask, depth_candidate_mask)
            for contour, area in self._candidate_contours(mask):
                color_stats = contour_color_stats(color_bgr, hsv, contour)
                if not passes_color_quality(color_name, color_stats):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                candidate = {
                    "color": color_name,
                    "area_px": float(area),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "color_stats": color_stats,
                }
                candidates.append(candidate)
                if best is None or area > best[0]:
                    best = (area, color_name, mask, contour)

        candidates.sort(key=lambda item: item["area_px"], reverse=True)
        if best is None:
            return None

        _, color_name, mask, contour = best
        return color_name, mask, contour, candidates

    def _make_mask(
        self,
        color_bgr: np.ndarray,
        hsv: np.ndarray,
        color_key: str,
        ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    ) -> np.ndarray:
        mask = make_color_mask(
            color_bgr,
            hsv,
            color_key,
            ranges,
            use_ratio=self._use_color_ratio_mask,
        )

        if self._morph_kernel_size > 1:
            k = self._morph_kernel_size
            kernel = np.ones((k, k), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _depth_candidate_mask(
        self,
        color_msg: Image,
        image_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        if not self._use_depth_candidate_mask or self._last_depth is None:
            return None

        color_stamp_ns = stamp_ns(color_msg) or time.time_ns()
        depth_age_s = abs(color_stamp_ns - self._last_depth_stamp_ns) / 1_000_000_000.0
        if depth_age_s > self._depth_match_timeout_s:
            return None

        depth = self._last_depth
        if depth.shape[:2] != image_shape:
            depth = cv2.resize(
                depth,
                (image_shape[1], image_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        values_m = depth_values_to_meters(
            depth,
            encoding=self._last_depth_encoding,
            depth_scale=self._depth_scale,
        )
        valid = np.isfinite(values_m)
        valid &= values_m > 0.0
        valid &= values_m >= self._min_depth_m
        valid &= values_m <= self._max_depth_m
        return np.where(valid, 255, 0).astype(np.uint8)

    def _candidate_contours(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._min_area_px:
                continue
            if self._max_area_px > 0 and area > self._max_area_px:
                continue
            candidates.append((area, contour))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [(contour, area) for area, contour in candidates]

    def _contour_center(
        self,
        contour,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> tuple[float, float]:
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        return x + w * 0.5, y + h * 0.5

    def _masked_depth(
        self,
        color_msg: Image,
        object_mask: np.ndarray,
    ) -> Optional[tuple[float, float, float, int, float]]:
        if self._last_depth is None:
            return None

        color_stamp_ns = stamp_ns(color_msg) or time.time_ns()
        depth_age_s = abs(color_stamp_ns - self._last_depth_stamp_ns) / 1_000_000_000.0
        if depth_age_s > self._depth_match_timeout_s:
            return None

        depth = self._last_depth
        if depth.shape[:2] != object_mask.shape[:2]:
            depth = cv2.resize(
                depth,
                (object_mask.shape[1], object_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        depth_mask = object_mask
        if self._depth_erode_px > 1:
            k = self._depth_erode_px
            kernel = np.ones((k, k), dtype=np.uint8)
            depth_mask = cv2.erode(depth_mask, kernel)

        raw_values = depth[depth_mask > 0]
        if raw_values.size == 0:
            return None

        if np.issubdtype(raw_values.dtype, np.floating):
            raw_values = raw_values[np.isfinite(raw_values)]
        raw_values = raw_values[raw_values > 0]
        if raw_values.size == 0:
            return None

        values_m = depth_values_to_meters(
            raw_values,
            encoding=self._last_depth_encoding,
            depth_scale=self._depth_scale,
        )
        values_m = values_m[np.isfinite(values_m)]
        values_m = values_m[
            (values_m >= self._min_depth_m) & (values_m <= self._max_depth_m)
        ]
        if values_m.size == 0:
            return None

        depth_m = float(np.percentile(values_m, self._depth_percentile))
        return (
            depth_m,
            float(np.min(values_m)),
            float(np.max(values_m)),
            int(values_m.size),
            float(depth_age_s),
        )

    def _deproject(
        self,
        color_msg: Image,
        u: float,
        v: float,
        depth_m: float,
    ) -> Optional[PointStamped]:
        if self._camera_info is None:
            return None

        fx = float(self._camera_info.k[0])
        fy = float(self._camera_info.k[4])
        cx = float(self._camera_info.k[2])
        cy = float(self._camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None

        out = PointStamped()
        out.header.stamp = color_msg.header.stamp
        out.header.frame_id = self._camera_info.header.frame_id or color_msg.header.frame_id
        out.point.x = (float(u) - cx) * depth_m / fx
        out.point.y = (float(v) - cy) * depth_m / fy
        out.point.z = depth_m
        return out

    def _publish_bbox(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self._bbox_pub.publish(msg)

    def _publish_status(self, color_msg: Image, payload: dict) -> None:
        status = {
            "stamp_ns": stamp_ns(color_msg) or time.time_ns(),
            "frames": self._frames,
            **payload,
        }
        msg = String()
        msg.data = json.dumps(status, sort_keys=True)
        self._status_pub.publish(msg)

    def _publish_miss(
        self,
        color_msg: Image,
        color_bgr: np.ndarray,
        mask: np.ndarray,
        *,
        reason: str,
        bbox: Optional[tuple[int, int, int, int]] = None,
        center: Optional[tuple[float, float]] = None,
        detected_color: Optional[str] = None,
        candidates: Optional[list[dict]] = None,
    ) -> None:
        if self._publish_nan_on_miss:
            depth_msg = Float32()
            depth_msg.data = float("nan")
            self._depth_pub.publish(depth_msg)

        payload = {
            "found": False,
            "reason": reason,
            "target_color": self._target_color,
            "detected_color": detected_color,
            "depth_m": None,
        }
        if bbox is not None:
            payload["bbox"] = [int(v) for v in bbox]
        if center is not None:
            payload["center_px"] = [float(center[0]), float(center[1])]
        if candidates is not None:
            payload["candidates"] = candidates

        self._publish_bbox(payload)
        self._publish_status(color_msg, payload)
        self._publish_debug_images(
            color_msg,
            color_bgr,
            mask,
            bbox=bbox,
            center=center,
            depth_m=None,
            detected_color=detected_color,
        )

    def _publish_debug_images(
        self,
        color_msg: Image,
        color_bgr: np.ndarray,
        mask: np.ndarray,
        *,
        bbox: Optional[tuple[int, int, int, int]],
        center: Optional[tuple[float, float]],
        depth_m: Optional[float],
        detected_color: Optional[str],
    ) -> None:
        if not self._publish_debug:
            return

        if self._mask_pub is not None:
            mask_msg = self._bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = color_msg.header
            self._mask_pub.publish(mask_msg)

        if self._debug_pub is None:
            return

        debug = color_bgr.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
        if center is not None:
            cv2.circle(debug, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
        label = detected_color or self._target_color
        depth_value = _finite_float(depth_m)
        if depth_value is not None:
            label += f" {depth_value:.3f}m"
        cv2.putText(
            debug,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header = color_msg.header
        self._debug_pub.publish(debug_msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ObjectDepthNode()
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
