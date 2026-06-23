"""External-force estimation via torque residual with hysteresis state machine.

Pipeline per tick:
    default:
        τ_expected[m] = kp·(q_des-q) + kd·(qd_des-qd) + τ_ff
    optional gravity_plus_ff mode:
        τ_expected[m] = τ_gravity[m] + kp·(q_des-q) + kd·(qd_des-qd) + τ_ff
    τ_residual[m] = τ_measured[m] - τ_expected[m]
    τ_external[m] = τ_residual[m] - offset[m]                # offset from calibrate()

Status state machine driven by ext_norm = max(|τ_external[active motors]|):
    clear  ──[ext > slow,  for N ticks]──→ slow
    slow   ──[ext > pause, for N ticks]──→ pause
    *      ──[ext > abort, immediate]───→ abort  (sticky; caller must reset())
    higher → lower transitions also require N-tick confirmation
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .robot_model import RobotModel


class ExternalForceEstimator:
    """Per-motor external-torque estimator + safety status emitter.

    Motor 7 (gripper) is disabled by default — its tau_limit is 1.6 Nm so
    residual SNR is too low for reliable force detection.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        *,
        threshold_slow: float = 2.0,
        threshold_pause: float = 5.0,
        threshold_abort: float = 15.0,
        confirm_ticks: int = 5,
        velocity_threshold_gain: float = 0.5,
        disabled_motors: frozenset[int] = frozenset({7}),
        expected_torque_mode: str = "command_total",
    ):
        if not (0.0 <= threshold_slow <= threshold_pause <= threshold_abort):
            raise ValueError("thresholds must satisfy 0 ≤ slow ≤ pause ≤ abort")
        if confirm_ticks < 1:
            raise ValueError("confirm_ticks must be >= 1")

        self.robot = robot_model
        self.threshold_slow = float(threshold_slow)
        self.threshold_pause = float(threshold_pause)
        self.threshold_abort = float(threshold_abort)
        self.confirm_ticks = int(confirm_ticks)
        self.velocity_threshold_gain = float(velocity_threshold_gain)
        self.disabled_motors = frozenset(int(m) for m in disabled_motors)
        self.expected_torque_mode = str(expected_torque_mode).strip()
        if self.expected_torque_mode not in {"command_total", "gravity_plus_ff"}:
            raise ValueError("expected_torque_mode must be 'command_total' or 'gravity_plus_ff'")

        self.offset: dict[int, float] = {m: 0.0 for m in robot_model.ordered_motor_ids}
        self._active_motor_ids = tuple(
            m for m in robot_model.ordered_motor_ids if m not in self.disabled_motors
        )

        self._status: str = "clear"
        self._counts: dict[str, int] = {"clear": confirm_ticks, "slow": 0, "pause": 0}

    @property
    def status(self) -> str:
        return self._status

    @property
    def active_motor_ids(self) -> tuple[int, ...]:
        return self._active_motor_ids

    def reset(self) -> None:
        """Reset state machine to 'clear'. Required to recover from 'abort'."""
        self._status = "clear"
        self._counts = {"clear": self.confirm_ticks, "slow": 0, "pause": 0}

    def calibrate(self, residual_samples: list[dict[int, float]]) -> None:
        """Set per-motor offset from a list of residual samples taken at rest.

        Each sample is a per-motor dict of (τ_measured - τ_expected) values
        collected while the robot is stationary. The mean per motor becomes
        the offset subtracted in subsequent ``estimate()`` calls.
        """
        if not residual_samples:
            raise ValueError("calibration requires at least one sample")
        for motor_id in self.robot.ordered_motor_ids:
            values = [s.get(motor_id, 0.0) for s in residual_samples]
            self.offset[motor_id] = float(np.mean(values))

    def estimate(
        self,
        *,
        q_by_motor: Mapping[int, float],
        qd_by_motor: Mapping[int, float],
        tau_measured_by_motor: Mapping[int, float],
        last_cmd_by_motor: Mapping[int, Mapping[str, float]],
    ) -> tuple[dict[int, float], str]:
        """Compute per-motor external torque + update status.

        Args:
            q_by_motor: current joint positions (rad)
            qd_by_motor: current joint velocities (rad/s)
            tau_measured_by_motor: motor torque readings (Nm)
            last_cmd_by_motor: motor_id → dict with keys ``q_des, qd_des, kp,
                kd, tau_ff`` (the most recently published command)

        Returns:
            (tau_external_by_motor, status) where status is one of
            ``"clear" | "slow" | "pause" | "abort"``.
        """
        gravity = self.robot.gravity_torque(q_by_motor) if self.expected_torque_mode == "gravity_plus_ff" else {}
        tau_ext: dict[int, float] = {}
        for motor_id in self.robot.ordered_motor_ids:
            cmd = last_cmd_by_motor.get(motor_id, {})
            q = float(q_by_motor[motor_id])
            qd = float(qd_by_motor[motor_id])
            q_des = float(cmd.get("q_des", q))
            qd_des = float(cmd.get("qd_des", qd))
            kp = float(cmd.get("kp", 0.0))
            kd = float(cmd.get("kd", 0.0))
            tau_ff = float(cmd.get("tau_ff", 0.0))

            tau_expected = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff
            if self.expected_torque_mode == "gravity_plus_ff":
                tau_expected += gravity[motor_id]
            tau_meas = float(tau_measured_by_motor.get(motor_id, 0.0))
            tau_ext[motor_id] = tau_meas - tau_expected - self.offset[motor_id]

        if not self._active_motor_ids:
            return tau_ext, self._status

        ext_norm = max(abs(tau_ext[m]) for m in self._active_motor_ids)
        qd_max = max(abs(qd_by_motor.get(m, 0.0)) for m in self._active_motor_ids)
        slow_thr = self.threshold_slow + self.velocity_threshold_gain * qd_max
        pause_thr = self.threshold_pause + self.velocity_threshold_gain * qd_max

        self._update_status(ext_norm, slow_thr, pause_thr)
        return tau_ext, self._status

    def _update_status(self, ext_norm: float, slow_thr: float, pause_thr: float) -> None:
        # abort is immediate and only cleared via reset()
        if ext_norm >= self.threshold_abort:
            self._status = "abort"
            return
        if self._status == "abort":
            return

        if ext_norm >= pause_thr:
            target = "pause"
        elif ext_norm >= slow_thr:
            target = "slow"
        else:
            target = "clear"

        for state in self._counts:
            if state == target:
                self._counts[state] = min(self.confirm_ticks, self._counts[state] + 1)
            else:
                self._counts[state] = max(0, self._counts[state] - 1)

        # Promote to highest-severity state that has confirmed.
        for state in ("pause", "slow", "clear"):
            if self._counts[state] >= self.confirm_ticks:
                self._status = state
                return
