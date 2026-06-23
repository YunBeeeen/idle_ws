"""Online GRU contact detector for real-robot monitoring.

This node is intentionally monitoring-only.  It subscribes to motor state and
applied motor command topics, reconstructs the same sensorless feature vector
used by the offline GRU, and publishes P(contact) plus a thresholded state.
It is not a certified safety function.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import rclpy
from msgs.msg import MotorCMDArray, MotorStateArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32


DEFAULT_TAU_LIMIT_BY_MOTOR = {1: 6.0, 2: 20.0, 3: 10.0, 4: 6.0, 5: 6.0, 6: 6.0}
SAFETY_NOTE = (
    "contact_detector_node is for monitoring/research only. "
    "Do not use this output as a certified safety stop."
)


@dataclass
class StateSample:
    q: float
    qdot: float
    stamp_s: float
    recv_s: float


@dataclass
class CommandSample:
    q_des: float
    qd_des: float
    kp: float
    kd: float
    tau_ff: float
    stamp_s: float
    recv_s: float


def stamp_to_sec(stamp: Any, fallback_s: float) -> float:
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return float(fallback_s)
    return float(sec) + 1.0e-9 * float(nanosec)


def sigmoid(value: float) -> float:
    clipped = max(-60.0, min(60.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def clip_symmetric(value: float, limit: float) -> float:
    if not math.isfinite(limit) or limit <= 0.0:
        return float(value)
    return float(max(-limit, min(limit, value)))


def parse_float_map_json(text: str, field_name: str) -> dict[int, float]:
    try:
        payload = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be JSON object text: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    out: dict[int, float] = {}
    for key, value in payload.items():
        out[int(key)] = float(value)
    return out


def original_42_feature_names(dof: int, use_delta_features: bool) -> list[str]:
    names: list[str] = []
    for prefix in ("q", "qdot", "e_q", "tau_cmd"):
        names.extend([f"{prefix}{idx + 1}" for idx in range(dof)])
    if use_delta_features:
        for prefix in ("delta_e_q", "delta_qdot", "delta_tau_cmd"):
            names.extend([f"{prefix}{idx + 1}" for idx in range(dof)])
    return names


ONLINE_FEATURE_PREFIXES = (
    "delta_tau_cmd",
    "delta_qdot",
    "delta_e_q",
    "tau_cmd",
    "qdot",
    "e_q",
    "q",
)


def parse_online_feature_name(name: str, dof: int) -> tuple[str, int]:
    """Parse names like ``delta_qdot3`` into (block_name, zero_based_index)."""

    text = str(name)
    for prefix in ONLINE_FEATURE_PREFIXES:
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if not suffix.isdigit():
            break
        index = int(suffix) - 1
        if index < 0 or index >= int(dof):
            raise ValueError(f"feature {name!r} has index outside 1..{dof}")
        return prefix, index
    raise ValueError(
        f"Unsupported online feature name {name!r}. "
        "Online detector currently supports q/qdot/e_q/tau_cmd and their delta variants, "
        "but not residual torque features."
    )


class OnlineGRU:
    """Small runtime GRU matching contact_detection.models.GRUDetector."""

    def __init__(self, torch_module: Any, checkpoint: dict[str, Any]) -> None:
        nn = torch_module.nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden_dim = int(checkpoint["hidden_dim"])
                num_layers = int(checkpoint["num_layers"])
                dropout = float(checkpoint["dropout"])
                bidirectional = bool(checkpoint.get("bidirectional", False))
                gru_dropout = dropout if num_layers > 1 else 0.0
                self.gru = nn.GRU(
                    input_size=int(checkpoint["input_dim"]),
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=gru_dropout,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
                direction_multiplier = 2 if bidirectional else 1
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_dim * direction_multiplier, 1)

            def forward(self, inputs):
                outputs, _ = self.gru(inputs)
                last_hidden = outputs[:, -1, :]
                return self.head(self.dropout(last_hidden)).squeeze(-1)

        self.module = _Model()
        self.module.load_state_dict(checkpoint["state_dict"], strict=True)


class ContactDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("contact_detector_node")

        self.model_path = Path(str(self.declare_parameter("model_path", "").value)).expanduser()
        self.scaler_path = Path(str(self.declare_parameter("scaler_path", "").value)).expanduser()
        self.state_topic = str(self.declare_parameter("state_topic", "/motor_state_array").value)
        self.command_topic = str(self.declare_parameter("command_topic", "/motor_cmd_array_applied").value)
        self.probability_topic = str(self.declare_parameter("probability_topic", "/contact_probability").value)
        self.contact_state_topic = str(self.declare_parameter("contact_state_topic", "/contact_state").value)
        self.ready_topic = str(self.declare_parameter("ready_topic", "/contact_detector_ready").value)
        self.motor_ids = [int(value) for value in self.declare_parameter("motor_ids", [1, 2, 3, 4, 5, 6]).value]
        self.state_timeout_s = float(self.declare_parameter("state_timeout_s", 0.25).value)
        self.cmd_timeout_s = float(self.declare_parameter("cmd_timeout_s", 0.25).value)
        self.sync_tolerance_s = float(self.declare_parameter("sync_tolerance_s", 0.10).value)
        self.inference_hz = float(self.declare_parameter("inference_hz", 100.0).value)
        self.decision_threshold_param = float(self.declare_parameter("decision_threshold", -1.0).value)
        self.enable_hysteresis = bool(self.declare_parameter("enable_hysteresis", True).value)
        self.contact_on_threshold_param = float(self.declare_parameter("contact_on_threshold", -1.0).value)
        self.contact_off_threshold_param = float(self.declare_parameter("contact_off_threshold", -1.0).value)
        self.device_name = str(self.declare_parameter("device", "cpu").value)
        self.rate_log_period_s = float(self.declare_parameter("rate_log_period_s", 3.0).value)
        self.debug_log_enabled = bool(self.declare_parameter("debug_log_enabled", False).value)
        self.debug_log_period_s = float(self.declare_parameter("debug_log_period_s", 1.0).value)
        self.apply_tau_clipping = bool(self.declare_parameter("apply_tau_clipping", True).value)
        default_limit_json = json.dumps(DEFAULT_TAU_LIMIT_BY_MOTOR, sort_keys=True)
        self.tau_limit_by_motor = parse_float_map_json(
            str(self.declare_parameter("tau_limit_by_motor_json", default_limit_json).value),
            "tau_limit_by_motor_json",
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"model_path does not exist: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"scaler_path does not exist: {self.scaler_path}")
        if not self.motor_ids or len(self.motor_ids) != len(set(self.motor_ids)):
            raise ValueError(f"motor_ids must be unique and non-empty: {self.motor_ids}")

        self.torch = self._import_torch()
        self.device = self._select_device(self.device_name)
        self.checkpoint = self._load_checkpoint(self.model_path)
        self.scaler_mean, self.scaler_std = self._load_scaler(self.scaler_path)
        self.model = OnlineGRU(self.torch, self.checkpoint).module.to(self.device)
        self.model.eval()

        self.dof = len(self.motor_ids)
        self.input_dim = int(self.checkpoint["input_dim"])
        self.window_length = int(self.checkpoint["window_length"])
        self.use_delta_features = bool(self.checkpoint.get("use_delta_features", self.input_dim == 7 * self.dof))
        self.feature_mode = str(self.checkpoint.get("feature_mode", "original_42"))
        self.feature_names = list(self.checkpoint.get("feature_names", []))
        if not self.feature_names:
            self.feature_names = original_42_feature_names(self.dof, self.use_delta_features)
        self.online_feature_names = list(self.feature_names)
        self.feature_layout = [parse_online_feature_name(name, self.dof) for name in self.online_feature_names]
        self._validate_feature_contract()

        self.decision_threshold = self._resolve_decision_threshold()
        self.contact_on_threshold = (
            self.decision_threshold
            if self.contact_on_threshold_param < 0.0
            else float(self.contact_on_threshold_param)
        )
        self.contact_off_threshold = (
            0.7 * self.decision_threshold
            if self.contact_off_threshold_param < 0.0
            else float(self.contact_off_threshold_param)
        )
        if self.contact_off_threshold > self.contact_on_threshold:
            raise ValueError("contact_off_threshold must be <= contact_on_threshold")

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.state_sub = self.create_subscription(MotorStateArray, self.state_topic, self.on_state_array, qos)
        self.cmd_sub = self.create_subscription(MotorCMDArray, self.command_topic, self.on_cmd_array, qos)
        self.prob_pub = self.create_publisher(Float32, self.probability_topic, qos)
        self.state_pub = self.create_publisher(Bool, self.contact_state_topic, qos)
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, qos)

        self.state_by_motor: dict[int, StateSample] = {}
        self.cmd_by_motor: dict[int, CommandSample] = {}
        self.feature_window: deque[np.ndarray] = deque(maxlen=self.window_length)
        self.prev_e_q: np.ndarray | None = None
        self.prev_qdot: np.ndarray | None = None
        self.prev_tau_cmd: np.ndarray | None = None
        self.last_processed_stamp_pair: tuple[float, float] | None = None
        self.contact_state = False
        self.log_last_s: dict[str, float] = {}
        self.debug_last_feature_stats: dict[str, float | int] = {}

        self.timer = self.create_timer(max(1.0 / max(self.inference_hz, 1.0), 1.0e-4), self.on_timer)
        self.get_logger().warn(SAFETY_NOTE)
        self.get_logger().info(
            "contact_detector_node initialized: "
            f"model={self.model_path} scaler={self.scaler_path} command_topic={self.command_topic} "
            f"input_dim={self.input_dim} window_length={self.window_length} "
            f"feature_mode={self.feature_mode} threshold={self.decision_threshold:.4f} "
            f"debug_log={self.debug_log_enabled}"
        )

    def _import_torch(self):
        try:
            import torch

            return torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for contact_detector_node") from exc

    def _select_device(self, requested: str):
        request = str(requested).strip().lower()
        if request.startswith("cuda") and self.torch.cuda.is_available():
            return self.torch.device(request)
        if request.startswith("cuda"):
            self.get_logger().warn("CUDA requested but unavailable; using CPU.")
        return self.torch.device("cpu")

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        checkpoint = self.torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("checkpoint must be a dict with state_dict")
        for key in ("input_dim", "hidden_dim", "num_layers", "dropout", "window_length"):
            if key not in checkpoint:
                raise ValueError(f"checkpoint missing {key!r}")
        return checkpoint

    def _load_scaler(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        mean = np.asarray(payload["mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(payload["std"], dtype=np.float64).reshape(-1)
        if mean.shape != std.shape:
            raise ValueError(f"scaler mean/std mismatch: {mean.shape} vs {std.shape}")
        return mean, np.maximum(std, 1.0e-8)

    def _validate_feature_contract(self) -> None:
        if self.scaler_mean.shape[0] != self.input_dim or self.scaler_std.shape[0] != self.input_dim:
            raise ValueError(
                f"Scaler dimension mismatch: scaler={self.scaler_mean.shape[0]}, checkpoint={self.input_dim}"
            )
        if len(self.online_feature_names) != self.input_dim:
            raise ValueError(
                f"Online feature dimension mismatch: {len(self.online_feature_names)} vs {self.input_dim}"
            )
        if any("tau_ext" in name for name in self.online_feature_names):
            raise ValueError("tau_ext must never appear in online model feature_names.")

    def _resolve_decision_threshold(self) -> float:
        if self.decision_threshold_param >= 0.0:
            return float(self.decision_threshold_param)
        for key in ("decision_threshold", "decision_threshold_value", "validation_selected_threshold"):
            if key in self.checkpoint:
                return float(self.checkpoint[key])
        return 0.5

    def _node_time_s(self) -> float:
        now = self.get_clock().now().to_msg()
        return float(now.sec) + 1.0e-9 * float(now.nanosec)

    def _should_log(self, key: str, period_s: float) -> bool:
        now_s = self._node_time_s()
        last_s = self.log_last_s.get(key, float("-inf"))
        if now_s - last_s >= float(period_s):
            self.log_last_s[key] = now_s
            return True
        return False

    def on_state_array(self, msg: MotorStateArray) -> None:
        recv_s = self._node_time_s()
        stamp_s = stamp_to_sec(msg.stamp, recv_s)
        for state in msg.states:
            motor_id = int(state.motor_id)
            if motor_id in self.motor_ids:
                self.state_by_motor[motor_id] = StateSample(
                    q=float(state.q),
                    qdot=float(state.qd),
                    stamp_s=stamp_s,
                    recv_s=recv_s,
                )

    def on_cmd_array(self, msg: MotorCMDArray) -> None:
        recv_s = self._node_time_s()
        stamp_s = stamp_to_sec(msg.stamp, recv_s)
        for cmd in msg.commands:
            motor_id = int(cmd.motor_id)
            if motor_id in self.motor_ids:
                self.cmd_by_motor[motor_id] = CommandSample(
                    q_des=float(cmd.q_des),
                    qd_des=float(getattr(cmd, "qd_des", 0.0)),
                    kp=max(0.0, float(cmd.kp)),
                    kd=max(0.0, float(cmd.kd)),
                    tau_ff=float(getattr(cmd, "tau_ff", 0.0)),
                    stamp_s=stamp_s,
                    recv_s=recv_s,
                )

    def _complete_and_fresh(self) -> tuple[bool, str]:
        now_s = self._node_time_s()
        missing_state = [motor_id for motor_id in self.motor_ids if motor_id not in self.state_by_motor]
        missing_cmd = [motor_id for motor_id in self.motor_ids if motor_id not in self.cmd_by_motor]
        if missing_state or missing_cmd:
            return False, f"missing_state={missing_state} missing_cmd={missing_cmd}"
        stale_state = [
            motor_id for motor_id in self.motor_ids if now_s - self.state_by_motor[motor_id].recv_s > self.state_timeout_s
        ]
        stale_cmd = [
            motor_id for motor_id in self.motor_ids if now_s - self.cmd_by_motor[motor_id].recv_s > self.cmd_timeout_s
        ]
        if stale_state or stale_cmd:
            return False, f"stale_state={stale_state} stale_cmd={stale_cmd}"
        state_stamp = max(self.state_by_motor[motor_id].stamp_s for motor_id in self.motor_ids)
        cmd_stamp = max(self.cmd_by_motor[motor_id].stamp_s for motor_id in self.motor_ids)
        if abs(state_stamp - cmd_stamp) > self.sync_tolerance_s:
            return False, f"state/cmd timestamp mismatch={abs(state_stamp - cmd_stamp):.4f}s"
        return True, ""

    def _publish_safe_default(self, ready: bool = False) -> None:
        probability = Float32()
        probability.data = 0.0
        contact = Bool()
        contact.data = False
        ready_msg = Bool()
        ready_msg.data = bool(ready)
        self.prob_pub.publish(probability)
        self.state_pub.publish(contact)
        self.ready_pub.publish(ready_msg)
        if not ready:
            self.contact_state = False

    def _reset_temporal_state(self) -> None:
        self.feature_window.clear()
        self.prev_e_q = None
        self.prev_qdot = None
        self.prev_tau_cmd = None
        self.last_processed_stamp_pair = None

    def _build_raw_feature(self) -> np.ndarray | None:
        q = np.asarray([self.state_by_motor[motor_id].q for motor_id in self.motor_ids], dtype=np.float64)
        qdot = np.asarray([self.state_by_motor[motor_id].qdot for motor_id in self.motor_ids], dtype=np.float64)
        q_des = np.asarray([self.cmd_by_motor[motor_id].q_des for motor_id in self.motor_ids], dtype=np.float64)
        qd_des = np.asarray([self.cmd_by_motor[motor_id].qd_des for motor_id in self.motor_ids], dtype=np.float64)
        kp = np.asarray([self.cmd_by_motor[motor_id].kp for motor_id in self.motor_ids], dtype=np.float64)
        kd = np.asarray([self.cmd_by_motor[motor_id].kd for motor_id in self.motor_ids], dtype=np.float64)
        tau_ff = np.asarray([self.cmd_by_motor[motor_id].tau_ff for motor_id in self.motor_ids], dtype=np.float64)

        e_q = q_des - q
        tau_cmd = kp * e_q + kd * (qd_des - qdot) + tau_ff
        if self.apply_tau_clipping:
            limits = np.asarray([self.tau_limit_by_motor.get(motor_id, float("inf")) for motor_id in self.motor_ids])
            tau_cmd = np.asarray([clip_symmetric(value, limit) for value, limit in zip(tau_cmd, limits)])

        if self.prev_e_q is None or self.prev_qdot is None or self.prev_tau_cmd is None:
            delta_e_q = np.zeros_like(e_q)
            delta_qdot = np.zeros_like(qdot)
            delta_tau_cmd = np.zeros_like(tau_cmd)
        else:
            delta_e_q = e_q - self.prev_e_q
            delta_qdot = qdot - self.prev_qdot
            delta_tau_cmd = tau_cmd - self.prev_tau_cmd

        blocks_by_name = {
            "q": q,
            "qdot": qdot,
            "e_q": e_q,
            "tau_cmd": tau_cmd,
            "delta_e_q": delta_e_q,
            "delta_qdot": delta_qdot,
            "delta_tau_cmd": delta_tau_cmd,
        }
        self.prev_e_q = e_q.copy()
        self.prev_qdot = qdot.copy()
        self.prev_tau_cmd = tau_cmd.copy()

        max_e_idx = int(np.argmax(np.abs(e_q))) if e_q.size else 0
        max_tau_idx = int(np.argmax(np.abs(tau_cmd))) if tau_cmd.size else 0
        self.debug_last_feature_stats = {
            "e_q_norm": float(np.linalg.norm(e_q)),
            "delta_e_q_norm": float(np.linalg.norm(delta_e_q)),
            "qdot_norm": float(np.linalg.norm(qdot)),
            "delta_qdot_norm": float(np.linalg.norm(delta_qdot)),
            "tau_cmd_norm": float(np.linalg.norm(tau_cmd)),
            "delta_tau_cmd_norm": float(np.linalg.norm(delta_tau_cmd)),
            "max_abs_e_q": float(np.max(np.abs(e_q))) if e_q.size else 0.0,
            "max_abs_e_q_motor": int(self.motor_ids[max_e_idx]) if self.motor_ids else -1,
            "max_abs_tau_cmd": float(np.max(np.abs(tau_cmd))) if tau_cmd.size else 0.0,
            "max_abs_tau_cmd_motor": int(self.motor_ids[max_tau_idx]) if self.motor_ids else -1,
            "tau_ff_norm": float(np.linalg.norm(tau_ff)),
            "kp_max": float(np.max(kp)) if kp.size else 0.0,
            "kd_max": float(np.max(kd)) if kd.size else 0.0,
        }

        feature = np.asarray(
            [blocks_by_name[block_name][index] for block_name, index in self.feature_layout],
            dtype=np.float64,
        )
        if not np.isfinite(feature).all():
            self.get_logger().error("NaN/Inf found in online feature; skipping inference.")
            return None
        return feature

    def _log_debug_sample(self, *, probability: float, logit: float, normalized: np.ndarray) -> None:
        if not self.debug_log_enabled or not self._should_log("debug_sample", self.debug_log_period_s):
            return
        stats = self.debug_last_feature_stats
        z_abs_max = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        z_norm = float(np.linalg.norm(normalized)) if normalized.size else 0.0
        self.get_logger().info(
            "[contact_debug] "
            f"ready=true p={probability:.4f} logit={logit:.3f} "
            f"state={self.contact_state} th={self.decision_threshold:.4f} "
            f"on/off={self.contact_on_threshold:.4f}/{self.contact_off_threshold:.4f} "
            f"win={len(self.feature_window)}/{self.window_length} "
            f"||e_q||={float(stats.get('e_q_norm', 0.0)):.4f} "
            f"max|e_q|=j{int(stats.get('max_abs_e_q_motor', -1))}:{float(stats.get('max_abs_e_q', 0.0)):.4f} "
            f"||de_q||={float(stats.get('delta_e_q_norm', 0.0)):.4f} "
            f"||qdot||={float(stats.get('qdot_norm', 0.0)):.4f} "
            f"||dqdot||={float(stats.get('delta_qdot_norm', 0.0)):.4f} "
            f"||tau_cmd||={float(stats.get('tau_cmd_norm', 0.0)):.4f} "
            f"max|tau_cmd|=j{int(stats.get('max_abs_tau_cmd_motor', -1))}:{float(stats.get('max_abs_tau_cmd', 0.0)):.4f} "
            f"||dtau_cmd||={float(stats.get('delta_tau_cmd_norm', 0.0)):.4f} "
            f"||tau_ff||={float(stats.get('tau_ff_norm', 0.0)):.4f} "
            f"z_norm={z_norm:.2f} z_abs_max={z_abs_max:.2f}"
        )

    def on_timer(self) -> None:
        complete, reason = self._complete_and_fresh()
        if not complete:
            if self._should_log("not_ready", 1.0):
                self.get_logger().warn(f"detector not ready: {reason}")
            self._reset_temporal_state()
            self._publish_safe_default(False)
            return

        state_stamp = max(self.state_by_motor[motor_id].stamp_s for motor_id in self.motor_ids)
        cmd_stamp = max(self.cmd_by_motor[motor_id].stamp_s for motor_id in self.motor_ids)
        stamp_pair = (float(state_stamp), float(cmd_stamp))
        if self.last_processed_stamp_pair == stamp_pair:
            return

        feature = self._build_raw_feature()
        if feature is None or feature.shape[0] != self.input_dim:
            self._reset_temporal_state()
            self._publish_safe_default(False)
            return
        normalized = (feature - self.scaler_mean) / self.scaler_std
        if not np.isfinite(normalized).all():
            self.get_logger().error("NaN/Inf found after scaler; skipping inference.")
            self._reset_temporal_state()
            self._publish_safe_default(False)
            return

        self.last_processed_stamp_pair = stamp_pair
        self.feature_window.append(normalized.astype(np.float32))
        if len(self.feature_window) < self.window_length:
            if self._should_log("warmup", 1.0):
                self.get_logger().info(f"detector warming up: {len(self.feature_window)}/{self.window_length}")
            self._publish_safe_default(False)
            return

        window = np.stack(list(self.feature_window), axis=0)[None, :, :]
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(window, dtype=self.torch.float32, device=self.device)
            logit = float(self.model(tensor).detach().cpu().numpy().reshape(-1)[0])
        probability = sigmoid(logit)

        if self.enable_hysteresis:
            if probability > self.contact_on_threshold:
                self.contact_state = True
            elif probability < self.contact_off_threshold:
                self.contact_state = False
        else:
            self.contact_state = probability > self.decision_threshold

        self._log_debug_sample(probability=probability, logit=logit, normalized=normalized)

        prob_msg = Float32()
        prob_msg.data = float(probability)
        state_msg = Bool()
        state_msg.data = bool(self.contact_state)
        ready_msg = Bool()
        ready_msg.data = True
        self.prob_pub.publish(prob_msg)
        self.state_pub.publish(state_msg)
        self.ready_pub.publish(ready_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ContactDetectorNode()
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
