"""Detect colored boxes and publish color, position, and image-plane yaw."""

from __future__ import annotations

import json
import math
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .color_segmentation import (
    contour_color_stats,
    make_color_mask,
    passes_color_quality,
    resolve_detector_ranges,
)
from .vision_utils import depth_values_to_meters, stamp_ns

COLOR_RGBA = {
    "red": (1.0, 0.05, 0.05, 1.0),
    "orange": (1.0, 0.45, 0.0, 1.0),
    "yellow": (1.0, 0.9, 0.0, 1.0),
    "green": (0.05, 0.85, 0.1, 1.0),
    "blue": (0.1, 0.35, 1.0, 1.0),
    "purple": (0.65, 0.2, 1.0, 1.0),
    "white": (1.0, 1.0, 1.0, 1.0),
    "black": (0.02, 0.02, 0.02, 1.0),
    "custom": (0.0, 1.0, 1.0, 1.0),
}


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = float(iw * ih)
    union = float(aw * ah + bw * bh) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _normalize_box_yaw_deg(angle_deg: float) -> float:
    """Normalize a rectangle long-edge angle to [-90, 90) degrees."""
    while angle_deg >= 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return angle_deg


def _angle_delta_deg(a_deg: float, b_deg: float) -> float:
    delta = a_deg - b_deg
    while delta >= 90.0:
        delta -= 180.0
    while delta < -90.0:
        delta += 180.0
    return delta


def _rect_yaw_deg(rect) -> float:
    (_, _), (width, height), angle = rect
    yaw = float(angle)
    if width < height:
        yaw += 90.0
    return _normalize_box_yaw_deg(yaw)


def _rect_aspect_ratio(rect) -> float:
    (_, _), (width, height), _ = rect
    short = max(min(float(width), float(height)), 1.0)
    long = max(float(width), float(height))
    return long / short


def _rotate_vector(q, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    qx = float(q.x)
    qy = float(q.y)
    qz = float(q.z)
    qw = float(q.w)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return xyz
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    x, y, z = xyz

    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)

    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def _transform_point(transform, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    rx, ry, rz = _rotate_vector(transform.transform.rotation, xyz)
    return (
        rx + float(transform.transform.translation.x),
        ry + float(transform.transform.translation.y),
        rz + float(transform.transform.translation.z),
    )


def _transform_yaw(transform, yaw_rad: float) -> float:
    vx = math.cos(yaw_rad)
    vy = math.sin(yaw_rad)
    bx, by, _ = _rotate_vector(transform.transform.rotation, (vx, vy, 0.0))
    return math.atan2(by, bx)


def _short_frame_id(frame_id: str) -> str:
    if frame_id == "camera_color_optical_frame":
        return "cam_optical"
    return frame_id or "unknown"


def _draw_label(
    image: np.ndarray,
    lines: list[str],
    anchor: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 2
    line_gap = 4
    margin = 4
    sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0]
        for line in lines
    ]
    text_width = max((size[0] for size in sizes), default=0)
    text_height = sum(size[1] for size in sizes) + line_gap * max(len(lines) - 1, 0)
    image_h, image_w = image.shape[:2]

    x = int(anchor[0])
    y = int(anchor[1])
    if x + text_width + margin > image_w:
        x = image_w - text_width - margin
    if x < margin:
        x = margin
    if y - text_height - margin < 0:
        y = text_height + margin
    if y > image_h - margin:
        y = image_h - margin

    cursor_y = y - text_height
    for line, (_, line_h) in zip(lines, sizes):
        cursor_y += line_h
        cv2.putText(
            image,
            line,
            (x, cursor_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += line_gap


def _parse_homography_json(text: str) -> Optional[np.ndarray]:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"plane_homography_json parse failed: {exc}") from exc

    array = np.asarray(raw, dtype=np.float64)
    if array.shape == (9,):
        array = array.reshape((3, 3))
    if array.shape != (3, 3):
        raise ValueError("plane_homography_json must be a flat 9-list or 3x3 list")
    if not np.all(np.isfinite(array)):
        raise ValueError("plane_homography_json must contain finite numbers")
    return array


class BoxPoseNode(Node):
    """Publish a sorted list of colored box poses from aligned RGB/depth."""

    def __init__(self) -> None:
        super().__init__("box_pose_node")
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
        self.declare_parameter("nms_iou_threshold", 0.4)
        self.declare_parameter("use_color_ratio_mask", True)
        self.declare_parameter("use_depth_candidate_mask", True)
        self.declare_parameter("yaw_stabilize_aspect_ratio", 1.15)
        self.declare_parameter("yaw_track_max_px", 80.0)
        self.declare_parameter("max_boxes", 20)
        self.declare_parameter("sort_by", "x")
        self.declare_parameter("require_depth", True)
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("plane_frame", "")
        self.declare_parameter("plane_z_m", 0.0)
        self.declare_parameter("plane_homography_json", "")
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("boxes_topic", "/idle_vision/box_poses")
        self.declare_parameter("pose_array_topic", "/idle_vision/box_pose_array")
        self.declare_parameter("marker_topic", "/idle_vision/box_pose/markers")
        self.declare_parameter("debug_topic", "/idle_vision/box_pose/debug_image")
        self.declare_parameter("mask_topic", "/idle_vision/box_pose/mask")

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
        self._nms_iou_threshold = float(self.get_parameter("nms_iou_threshold").value)
        self._use_color_ratio_mask = bool(
            self.get_parameter("use_color_ratio_mask").value
        )
        self._use_depth_candidate_mask = bool(
            self.get_parameter("use_depth_candidate_mask").value
        )
        self._yaw_stabilize_aspect_ratio = float(
            self.get_parameter("yaw_stabilize_aspect_ratio").value
        )
        self._yaw_track_max_px = float(self.get_parameter("yaw_track_max_px").value)
        self._max_boxes = int(self.get_parameter("max_boxes").value)
        self._sort_by = str(self.get_parameter("sort_by").value).strip().lower()
        self._require_depth = bool(self.get_parameter("require_depth").value)
        self._base_frame = str(self.get_parameter("base_frame").value).strip()
        self._plane_frame = str(self.get_parameter("plane_frame").value).strip()
        self._plane_z_m = float(self.get_parameter("plane_z_m").value)
        self._plane_homography = _parse_homography_json(
            str(self.get_parameter("plane_homography_json").value)
        )
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
        if not 0.0 <= self._nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be between 0 and 1")
        if self._yaw_stabilize_aspect_ratio < 1.0:
            raise ValueError("yaw_stabilize_aspect_ratio must be >= 1.0")
        if self._yaw_track_max_px < 0.0:
            raise ValueError("yaw_track_max_px must be >= 0")
        if self._max_boxes <= 0:
            raise ValueError("max_boxes must be > 0")

        self._last_depth: Optional[np.ndarray] = None
        self._last_depth_encoding = ""
        self._last_depth_stamp_ns = 0
        self._camera_info: Optional[CameraInfo] = None
        self._previous_yaw_tracks: list[dict] = []
        self._frames = 0
        self._warned_tf_missing = False
        self._tf_buffer = Buffer() if self._base_frame else None
        self._tf_listener = (
            TransformListener(self._tf_buffer, self) if self._tf_buffer is not None else None
        )

        self._boxes_pub = self.create_publisher(
            String,
            str(self.get_parameter("boxes_topic").value),
            10,
        )
        self._pose_array_pub = self.create_publisher(
            PoseArray,
            str(self.get_parameter("pose_array_topic").value),
            10,
        )
        self._marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
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
            "box pose ready: target=%s detectors=%s color=%s depth=%s base_frame=%s plane=%s"
            % (
                self._target_color,
                ",".join(self._detector_names),
                self._color_topic,
                self._depth_topic,
                self._base_frame or "disabled",
                self._plane_frame or ("homography" if self._plane_homography is not None else "disabled"),
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

        frame_id = self._output_frame_id(msg)
        base_transform = self._lookup_base_transform(frame_id)
        pose_frame_id = self._base_frame if base_transform is not None else frame_id
        if self._plane_homography is not None:
            pose_frame_id = self._plane_frame or self._base_frame or "plane"

        depth_candidate_mask = self._depth_candidate_mask(msg, color_bgr.shape[:2])
        candidates, combined_mask = self._detect_candidates(
            color_bgr,
            depth_candidate_mask=depth_candidate_mask,
        )
        boxes = []
        for candidate in candidates:
            object_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
            cv2.drawContours(
                object_mask,
                [candidate["contour"]],
                -1,
                255,
                thickness=cv2.FILLED,
            )

            depth_info = self._masked_depth(msg, object_mask)
            if depth_info is None and self._require_depth:
                continue

            depth_m = None
            depth_min_m = None
            depth_max_m = None
            depth_count = 0
            depth_age_s = None
            if depth_info is not None:
                depth_m, depth_min_m, depth_max_m, depth_count, depth_age_s = depth_info

            center_u, center_v = candidate["center_px"]
            camera_xyz = self._deproject(center_u, center_v, depth_m)
            plane_pose = None
            plane_yaw_rad = None
            if camera_xyz is None:
                plane_pose = self._project_pixel_to_plane(center_u, center_v)
                plane_yaw_rad = self._project_pixel_yaw_to_plane(
                    center_u,
                    center_v,
                    candidate["yaw_rad"],
                )

            if camera_xyz is None and plane_pose is None and self._require_depth:
                continue

            pose_xyz = camera_xyz if camera_xyz is not None else plane_pose
            pose_yaw_rad = candidate["yaw_rad"]
            if plane_yaw_rad is not None:
                pose_yaw_rad = plane_yaw_rad
            if base_transform is not None and camera_xyz is not None:
                pose_xyz = _transform_point(base_transform, camera_xyz)
                pose_yaw_rad = _transform_yaw(base_transform, candidate["yaw_rad"])

            boxes.append(
                {
                    "color": candidate["color_label"],
                    "color_key": candidate["color_key"],
                    "pose_frame_id": pose_frame_id,
                    "x_m": pose_xyz[0] if pose_xyz is not None else None,
                    "y_m": pose_xyz[1] if pose_xyz is not None else None,
                    "z_m": pose_xyz[2] if pose_xyz is not None else depth_m,
                    "yaw_rad": pose_yaw_rad,
                    "yaw_deg": math.degrees(pose_yaw_rad),
                    "camera_frame_id": frame_id,
                    "camera_x_m": camera_xyz[0] if camera_xyz is not None else None,
                    "camera_y_m": camera_xyz[1] if camera_xyz is not None else None,
                    "camera_z_m": camera_xyz[2] if camera_xyz is not None else depth_m,
                    "camera_yaw_rad": candidate["yaw_rad"],
                    "camera_yaw_deg": candidate["yaw_deg"],
                    "center_px": [float(center_u), float(center_v)],
                    "bbox": [int(v) for v in candidate["bbox"]],
                    "area_px": float(candidate["area_px"]),
                    "depth_m": depth_m,
                    "depth_min_m": depth_min_m,
                    "depth_max_m": depth_max_m,
                    "depth_count": int(depth_count),
                    "depth_age_s": depth_age_s,
                    "rect_points_px": candidate["rect_points_px"],
                    "color_stats": candidate["color_stats"],
                }
            )

        boxes = self._sort_boxes(boxes)[: self._max_boxes]
        for idx, box in enumerate(boxes):
            box["index"] = idx

        payload = {
            "stamp_ns": stamp_ns(msg) or time.time_ns(),
            "frame_id": frame_id,
            "pose_frame_id": pose_frame_id,
            "count": len(boxes),
            "boxes": boxes,
        }
        out = String()
        out.data = json.dumps(payload, sort_keys=True)
        self._boxes_pub.publish(out)

        pose_array = PoseArray()
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = pose_frame_id
        pose_array.poses = [
            pose for pose in (self._pose_from_box(box) for box in boxes) if pose is not None
        ]
        self._pose_array_pub.publish(pose_array)
        self._publish_markers(msg, pose_frame_id, boxes)

        self._publish_debug_images(msg, color_bgr, combined_mask, boxes)

    def _detect_candidates(
        self,
        color_bgr: np.ndarray,
        *,
        depth_candidate_mask: Optional[np.ndarray],
    ) -> tuple[list[dict], np.ndarray]:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        raw_candidates = []

        for color_key, ranges in self._detector_ranges.items():
            mask = self._make_mask(color_bgr, hsv, color_key, ranges)
            if depth_candidate_mask is not None:
                mask = cv2.bitwise_and(mask, depth_candidate_mask)
            accepted_mask = np.zeros(mask.shape, dtype=np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self._min_area_px:
                    continue
                if self._max_area_px > 0 and area > self._max_area_px:
                    continue
                color_stats = contour_color_stats(color_bgr, hsv, contour)
                if not passes_color_quality(color_key, color_stats):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                rect = cv2.minAreaRect(contour)
                rect_points = cv2.boxPoints(rect)
                rect_points_i = np.asarray(rect_points, dtype=np.int32)
                yaw_deg = _rect_yaw_deg(rect)
                aspect_ratio = _rect_aspect_ratio(rect)
                cv2.drawContours(accepted_mask, [contour], -1, 255, thickness=cv2.FILLED)
                raw_candidates.append(
                    {
                        "color_key": color_key,
                        "color_label": color_key.title(),
                        "contour": contour,
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "area_px": area,
                        "center_px": (float(rect[0][0]), float(rect[0][1])),
                        "yaw_deg": float(yaw_deg),
                        "yaw_rad": float(math.radians(yaw_deg)),
                        "aspect_ratio": float(aspect_ratio),
                        "rect_points_px": rect_points_i.tolist(),
                        "color_stats": color_stats,
                    }
                )
            combined_mask |= accepted_mask

        raw_candidates.sort(key=lambda item: item["area_px"], reverse=True)
        kept = []
        for candidate in raw_candidates:
            if any(
                _bbox_iou(candidate["bbox"], existing["bbox"]) > self._nms_iou_threshold
                for existing in kept
            ):
                continue
            kept.append(candidate)
            if len(kept) >= self._max_boxes:
                break

        self._stabilize_yaws(kept)
        return kept, combined_mask

    def _stabilize_yaws(self, candidates: list[dict]) -> None:
        previous_tracks = self._previous_yaw_tracks
        used_previous: set[int] = set()
        new_tracks = []

        for candidate in candidates:
            center_x, center_y = candidate["center_px"]
            best_index = -1
            best_dist = float("inf")
            for index, previous in enumerate(previous_tracks):
                if index in used_previous:
                    continue
                if previous["color_key"] != candidate["color_key"]:
                    continue
                prev_x, prev_y = previous["center_px"]
                dist = math.hypot(center_x - prev_x, center_y - prev_y)
                if dist < best_dist:
                    best_dist = dist
                    best_index = index

            if best_index >= 0 and best_dist <= self._yaw_track_max_px:
                previous_yaw = float(previous_tracks[best_index]["yaw_deg"])
                yaw_options = [float(candidate["yaw_deg"])]
                if candidate["aspect_ratio"] <= self._yaw_stabilize_aspect_ratio:
                    yaw_options.append(
                        _normalize_box_yaw_deg(float(candidate["yaw_deg"]) + 90.0)
                    )
                yaw_deg = min(
                    yaw_options,
                    key=lambda yaw: abs(_angle_delta_deg(yaw, previous_yaw)),
                )
                candidate["yaw_deg"] = float(yaw_deg)
                candidate["yaw_rad"] = float(math.radians(yaw_deg))
                used_previous.add(best_index)

            new_tracks.append(
                {
                    "color_key": candidate["color_key"],
                    "center_px": candidate["center_px"],
                    "yaw_deg": candidate["yaw_deg"],
                }
            )

        self._previous_yaw_tracks = new_tracks

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
        u: float,
        v: float,
        depth_m: Optional[float],
    ) -> Optional[tuple[float, float, float]]:
        if depth_m is None or self._camera_info is None:
            return None
        fx = float(self._camera_info.k[0])
        fy = float(self._camera_info.k[4])
        cx = float(self._camera_info.k[2])
        cy = float(self._camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None
        x_m = (float(u) - cx) * depth_m / fx
        y_m = (float(v) - cy) * depth_m / fy
        return x_m, y_m, depth_m

    def _project_pixel_to_plane(
        self,
        u: float,
        v: float,
    ) -> Optional[tuple[float, float, float]]:
        if self._plane_homography is None:
            return None
        pixel = np.array([float(u), float(v), 1.0], dtype=np.float64)
        projected = self._plane_homography @ pixel
        scale = float(projected[2])
        if abs(scale) < 1e-9:
            return None
        x_m = float(projected[0] / scale)
        y_m = float(projected[1] / scale)
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            return None
        return x_m, y_m, self._plane_z_m

    def _project_pixel_yaw_to_plane(
        self,
        u: float,
        v: float,
        yaw_rad: float,
    ) -> Optional[float]:
        if self._plane_homography is None:
            return None
        p0 = self._project_pixel_to_plane(u, v)
        sample_px = 20.0
        p1 = self._project_pixel_to_plane(
            u + math.cos(yaw_rad) * sample_px,
            v + math.sin(yaw_rad) * sample_px,
        )
        if p0 is None or p1 is None:
            return None
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        return math.atan2(dy, dx)

    def _pose_from_box(self, box: dict) -> Optional[Pose]:
        if box["x_m"] is None or box["y_m"] is None or box["z_m"] is None:
            return None
        return self._pose_from_point_and_yaw(
            (float(box["x_m"]), float(box["y_m"]), float(box["z_m"])),
            float(box["yaw_rad"]),
        )

    def _pose_from_point_and_yaw(
        self,
        point_xyz: Optional[tuple[float, float, float]],
        yaw_rad: float,
    ) -> Optional[Pose]:
        if point_xyz is None:
            return None
        pose = Pose()
        pose.position.x = float(point_xyz[0])
        pose.position.y = float(point_xyz[1])
        pose.position.z = float(point_xyz[2])
        pose.orientation.z = math.sin(yaw_rad * 0.5)
        pose.orientation.w = math.cos(yaw_rad * 0.5)
        return pose

    def _sort_boxes(self, boxes: list[dict]) -> list[dict]:
        if self._sort_by == "area":
            return sorted(boxes, key=lambda item: item["area_px"], reverse=True)
        if self._sort_by == "color":
            return sorted(boxes, key=lambda item: (item["color"], item["center_px"][0]))
        return sorted(boxes, key=lambda item: item["center_px"][0])

    def _output_frame_id(self, color_msg: Image) -> str:
        if self._camera_info is not None and self._camera_info.header.frame_id:
            return self._camera_info.header.frame_id
        return color_msg.header.frame_id

    def _lookup_base_transform(self, source_frame: str):
        if not self._base_frame or self._tf_buffer is None:
            return None
        if not source_frame or source_frame == self._base_frame:
            return None
        try:
            return self._tf_buffer.lookup_transform(
                self._base_frame,
                source_frame,
                Time(),
            )
        except TransformException as exc:
            if not self._warned_tf_missing:
                self.get_logger().warning(
                    "base transform unavailable: %s -> %s (%s). "
                    "Publishing camera-frame x/y/yaw until calibration TF is available."
                    % (source_frame, self._base_frame, exc)
                )
                self._warned_tf_missing = True
            return None

    def _publish_markers(
        self,
        color_msg: Image,
        frame_id: str,
        boxes: list[dict],
    ) -> None:
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.header.stamp = color_msg.header.stamp
        delete_all.header.frame_id = frame_id
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        marker_id = 1
        for box in boxes:
            if box["x_m"] is None or box["y_m"] is None or box["z_m"] is None:
                continue

            color = COLOR_RGBA.get(box["color_key"], COLOR_RGBA["custom"])
            x = float(box["x_m"])
            y = float(box["y_m"])
            z = float(box["z_m"])
            yaw = float(box["yaw_rad"])

            sphere = Marker()
            sphere.header.stamp = color_msg.header.stamp
            sphere.header.frame_id = frame_id
            sphere.ns = "box_centers"
            sphere.id = marker_id
            marker_id += 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.035
            sphere.scale.y = 0.035
            sphere.scale.z = 0.035
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = color
            marker_array.markers.append(sphere)

            arrow = Marker()
            arrow.header.stamp = color_msg.header.stamp
            arrow.header.frame_id = frame_id
            arrow.ns = "box_yaw"
            arrow.id = marker_id
            marker_id += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = x
            arrow.pose.position.y = y
            arrow.pose.position.z = z
            arrow.pose.orientation.z = math.sin(yaw * 0.5)
            arrow.pose.orientation.w = math.cos(yaw * 0.5)
            arrow.scale.x = 0.12
            arrow.scale.y = 0.018
            arrow.scale.z = 0.018
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = color
            marker_array.markers.append(arrow)

            text = Marker()
            text.header.stamp = color_msg.header.stamp
            text.header.frame_id = frame_id
            text.ns = "box_labels"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + 0.07
            text.pose.orientation.w = 1.0
            text.scale.z = 0.045
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            frame_text = _short_frame_id(str(box.get("pose_frame_id", "")))
            text.text = (
                f'{box["color"]} [{frame_text}]\n'
                f'x={x:.3f} y={y:.3f} z={z:.3f}\n'
                f'yaw={box["yaw_deg"]:.1f} deg'
            )
            marker_array.markers.append(text)

        self._marker_pub.publish(marker_array)

    def _publish_debug_images(
        self,
        color_msg: Image,
        color_bgr: np.ndarray,
        combined_mask: np.ndarray,
        boxes: list[dict],
    ) -> None:
        if not self._publish_debug:
            return

        if self._mask_pub is not None:
            mask_msg = self._bridge.cv2_to_imgmsg(combined_mask, encoding="mono8")
            mask_msg.header = color_msg.header
            self._mask_pub.publish(mask_msg)

        if self._debug_pub is None:
            return

        debug = color_bgr.copy()
        for box in boxes:
            pts = np.asarray(box["rect_points_px"], dtype=np.int32)
            cv2.polylines(debug, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            u, v = box["center_px"]
            cv2.circle(debug, (int(u), int(v)), 4, (0, 0, 255), -1)
            x_text = "nan" if box["x_m"] is None else f'{box["x_m"]:.3f}'
            y_text = "nan" if box["y_m"] is None else f'{box["y_m"]:.3f}'
            z_text = "nan" if box["z_m"] is None else f'{box["z_m"]:.3f}'
            frame_text = _short_frame_id(str(box.get("pose_frame_id", "")))
            label_lines = [
                f'{box["color"]} [{frame_text}]',
                f'x={x_text} y={y_text} z={z_text}',
                f'yaw={box["yaw_deg"]:.1f}',
            ]
            _draw_label(
                debug,
                label_lines,
                (int(u) + 8, int(v) - 8),
            )

        debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header = color_msg.header
        self._debug_pub.publish(debug_msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = BoxPoseNode()
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
