"""Select an HSV-detected box from natural-language commands parsed by Ollama."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from typing import Optional

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


COLOR_ALIASES = {
    "red": "red",
    "빨강": "red",
    "빨간": "red",
    "빨간색": "red",
    "초록": "green",
    "초록색": "green",
    "녹색": "green",
    "연두": "green",
    "연두색": "green",
    "green": "green",
    "blue": "blue",
    "파랑": "blue",
    "파란": "blue",
    "파란색": "blue",
    "하늘": "blue",
    "하늘색": "blue",
}

SPATIAL_ALIASES = {
    "왼쪽": "left",
    "좌측": "left",
    "left": "left",
    "오른쪽": "right",
    "우측": "right",
    "right": "right",
    "가운데": "center",
    "중앙": "center",
    "center": "center",
    "가까운": "nearest",
    "가장 가까운": "nearest",
    "nearest": "nearest",
    "먼": "farthest",
    "가장 먼": "farthest",
    "farthest": "farthest",
    "큰": "largest",
    "가장 큰": "largest",
    "largest": "largest",
}


def _extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_color(value: object) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ("", "none", "null", "any", "all", "전체", "아무거나"):
        return None
    return COLOR_ALIASES.get(key, key if key in ("red", "green", "blue") else None)


def _normalize_spatial(value: object) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ("", "none", "null", "any"):
        return None
    return SPATIAL_ALIASES.get(key, key if key in {"left", "right", "center", "nearest", "farthest", "largest"} else None)


def _parse_with_rules(command: str) -> dict:
    lowered = command.lower()
    parsed: dict[str, object] = {
        "color_key": None,
        "spatial": None,
        "object": "box",
        "raw_command": command,
        "parser": "rules",
    }
    for token, color in COLOR_ALIASES.items():
        if token in lowered:
            parsed["color_key"] = color
            break
    for token, spatial in SPATIAL_ALIASES.items():
        if token in lowered:
            parsed["spatial"] = spatial
            break
    return parsed


def _box_color_key(box: dict) -> str:
    return str(box.get("color_key") or box.get("color") or "").strip().lower()


def _finite_number(value: object) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class QwenBoxSelectorNode(Node):
    """Parse commands with Qwen/Ollama and publish the selected detected box."""

    def __init__(self) -> None:
        super().__init__("qwen_box_selector_node")

        self.declare_parameter("command_topic", "/idle_vision/qwen/command")
        self.declare_parameter("boxes_topic", "/idle_vision/box_poses")
        self.declare_parameter("parsed_topic", "/idle_vision/qwen/parsed_command")
        self.declare_parameter("selected_box_topic", "/idle_vision/qwen/selected_box")
        self.declare_parameter("target_pose_topic", "/idle_vision/qwen/target_pose")
        self.declare_parameter("status_topic", "/idle_vision/qwen/status")
        self.declare_parameter("ollama_url", "http://127.0.0.1:11434/api/generate")
        self.declare_parameter("ollama_model", "qwen2.5:7b")
        self.declare_parameter("use_ollama", True)
        self.declare_parameter("ollama_timeout_s", 4.0)
        self.declare_parameter("default_spatial", "largest")
        self.declare_parameter("spatial_reference", "image")

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._boxes_topic = str(self.get_parameter("boxes_topic").value)
        self._ollama_url = str(self.get_parameter("ollama_url").value)
        self._ollama_model = str(self.get_parameter("ollama_model").value)
        self._use_ollama = bool(self.get_parameter("use_ollama").value)
        self._ollama_timeout_s = float(self.get_parameter("ollama_timeout_s").value)
        self._default_spatial = _normalize_spatial(
            self.get_parameter("default_spatial").value
        ) or "largest"
        self._spatial_reference = (
            str(self.get_parameter("spatial_reference").value).strip().lower()
        )
        if self._spatial_reference not in ("image", "pose"):
            raise ValueError("spatial_reference must be 'image' or 'pose'")

        self._latest_boxes_payload: Optional[dict] = None
        self._latest_parsed: Optional[dict] = None
        self._latest_command = ""

        self._parsed_pub = self.create_publisher(
            String,
            str(self.get_parameter("parsed_topic").value),
            10,
        )
        self._selected_pub = self.create_publisher(
            String,
            str(self.get_parameter("selected_box_topic").value),
            10,
        )
        self._pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("target_pose_topic").value),
            10,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )

        self.create_subscription(String, self._command_topic, self._on_command, 10)
        self.create_subscription(String, self._boxes_topic, self._on_boxes, 10)

        self.get_logger().info(
            "qwen selector ready: command=%s boxes=%s model=%s use_ollama=%s"
            % (
                self._command_topic,
                self._boxes_topic,
                self._ollama_model,
                self._use_ollama,
            )
        )

    def _on_boxes(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._publish_status({"ok": False, "error": f"box JSON parse failed: {exc}"})
            return
        self._latest_boxes_payload = payload
        if self._latest_parsed is not None:
            self._publish_selection()

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip()
        if not command:
            return
        self._latest_command = command
        parsed = None
        if self._use_ollama:
            parsed = self._parse_with_ollama(command)
        if parsed is None:
            parsed = _parse_with_rules(command)
        parsed = self._normalize_parsed(parsed, command)
        self._latest_parsed = parsed

        out = String()
        out.data = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        self._parsed_pub.publish(out)
        self._publish_selection()

    def _parse_with_ollama(self, command: str) -> Optional[dict]:
        prompt = (
            "You parse robot vision commands for colored blocks. "
            "Return JSON only, no markdown. "
            "Schema: {\"color_key\":\"red|green|blue|null\", "
            "\"spatial\":\"left|right|center|nearest|farthest|largest|null\", "
            "\"object\":\"box|block|null\"}. "
            "Korean color aliases: 빨강=red, 초록/연두=green, 파랑/하늘색=blue. "
            "Command: " + command
        )
        body = json.dumps(
            {
                "model": self._ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._ollama_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._ollama_timeout_s,
            ) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self._publish_status(
                {
                    "ok": False,
                    "error": f"Ollama parse failed, using rules fallback: {exc}",
                }
            )
            return None

        parsed = _extract_json_object(str(response_data.get("response", "")))
        if parsed is None:
            self._publish_status(
                {
                    "ok": False,
                    "error": "Ollama response did not contain a JSON object",
                    "raw_response": response_data.get("response", ""),
                }
            )
        return parsed

    def _normalize_parsed(self, parsed: dict, command: str) -> dict:
        color = _normalize_color(
            parsed.get("color_key", parsed.get("color", parsed.get("target_color")))
        )
        spatial = _normalize_spatial(parsed.get("spatial", parsed.get("position")))
        if spatial is None:
            spatial = self._default_spatial
        return {
            "color_key": color,
            "spatial": spatial,
            "object": str(parsed.get("object") or "box"),
            "raw_command": command,
            "parser": str(parsed.get("parser") or ("ollama" if self._use_ollama else "rules")),
        }

    def _publish_selection(self) -> None:
        parsed = self._latest_parsed
        payload = self._latest_boxes_payload
        if parsed is None:
            return
        if payload is None:
            self._publish_selected(
                {
                    "found": False,
                    "reason": "no box_poses received yet",
                    "parsed": parsed,
                    "command": self._latest_command,
                }
            )
            return

        boxes = payload.get("boxes", [])
        if not isinstance(boxes, list):
            boxes = []
        color_key = parsed.get("color_key")
        candidates = [
            box for box in boxes
            if not color_key or _box_color_key(box) == color_key
        ]
        if not candidates:
            self._publish_selected(
                {
                    "found": False,
                    "reason": "no matching detected box",
                    "parsed": parsed,
                    "command": self._latest_command,
                    "box_count": len(boxes),
                    "pose_frame_id": payload.get("pose_frame_id"),
                }
            )
            return

        selected = self._select_candidate(candidates, str(parsed.get("spatial") or "largest"))
        result = {
            "found": True,
            "command": self._latest_command,
            "parsed": parsed,
            "selected_index": selected.get("index"),
            "box": selected,
            "pose_frame_id": selected.get("pose_frame_id") or payload.get("pose_frame_id"),
            "x_m": selected.get("x_m"),
            "y_m": selected.get("y_m"),
            "z_m": selected.get("z_m"),
            "yaw_deg": selected.get("yaw_deg"),
            "depth_m": selected.get("depth_m"),
            "candidate_count": len(candidates),
        }
        self._publish_selected(result)
        self._publish_pose(selected, result["pose_frame_id"])

    def _select_candidate(self, candidates: list[dict], spatial: str) -> dict:
        if spatial == "nearest":
            return min(candidates, key=lambda box: _finite_number(box.get("depth_m")) or float("inf"))
        if spatial == "farthest":
            return max(candidates, key=lambda box: _finite_number(box.get("depth_m")) or float("-inf"))
        if spatial == "left":
            if self._spatial_reference == "image":
                return min(candidates, key=lambda box: self._center_u(box))
            return min(candidates, key=lambda box: _finite_number(box.get("x_m")) or float("inf"))
        if spatial == "right":
            if self._spatial_reference == "image":
                return max(candidates, key=lambda box: self._center_u(box))
            return max(candidates, key=lambda box: _finite_number(box.get("x_m")) or float("-inf"))
        if spatial == "center":
            if self._spatial_reference == "image":
                return min(candidates, key=lambda box: abs(self._center_u(box) - 320.0))
            return min(
                candidates,
                key=lambda box: math.hypot(
                    _finite_number(box.get("x_m")) or 0.0,
                    _finite_number(box.get("y_m")) or 0.0,
                ),
            )
        return max(candidates, key=lambda box: _finite_number(box.get("area_px")) or 0.0)

    @staticmethod
    def _center_u(box: dict) -> float:
        center = box.get("center_px")
        if isinstance(center, list) and center and isinstance(center[0], (int, float)):
            return float(center[0])
        bbox = box.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 3:
            return float(bbox[0]) + float(bbox[2]) * 0.5
        return 0.0

    def _publish_pose(self, box: dict, frame_id: object) -> None:
        x = _finite_number(box.get("x_m"))
        y = _finite_number(box.get("y_m"))
        z = _finite_number(box.get("z_m"))
        yaw_deg = _finite_number(box.get("yaw_deg")) or 0.0
        if x is None or y is None or z is None:
            return
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = str(frame_id or box.get("pose_frame_id") or "")
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        yaw_rad = math.radians(yaw_deg)
        pose.pose.orientation.z = math.sin(yaw_rad * 0.5)
        pose.pose.orientation.w = math.cos(yaw_rad * 0.5)
        self._pose_pub.publish(pose)

    def _publish_selected(self, payload: dict) -> None:
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._selected_pub.publish(out)
        self._publish_status({"ok": bool(payload.get("found")), "selected": payload})

    def _publish_status(self, payload: dict) -> None:
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._status_pub.publish(out)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = QwenBoxSelectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
