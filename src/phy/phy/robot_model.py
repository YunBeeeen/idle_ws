"""Shared Pinocchio robot model — single source for FK / Jacobian / dynamics.

Wraps one ``pin.Model`` + ``pin.Data`` and exposes motor-id-keyed APIs so
callers don't deal with pinocchio's internal q/v indices. Pre-allocates the
joint-configuration buffer to avoid per-tick numpy allocations.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pinocchio as pin

# Gripper crank-slider geometry (measured from CAD)
# 0 rad = fully open (finger_r prismatic = 0 mm)
# +1.4099 rad = fully closed (finger_r prismatic ≈ 42 mm measured, 40.74 mm theoretical)
_GRIPPER_R = 0.0325   # crank radius [m] — motor shaft center to crank pin
_GRIPPER_L = 0.045    # connecting rod length [m] — crank pin to slider pin
_GRIPPER_THETA_CLOSED = 1.4099  # motor angle at fully closed [rad]


def _gripper_disp(theta: float) -> float:
    r, l = _GRIPPER_R, _GRIPPER_L
    sin_t = math.sin(theta)
    discriminant = l * l - r * r * sin_t * sin_t
    if discriminant < 0.0:
        discriminant = 0.0
    return (r + l) - (r * math.cos(theta) + math.sqrt(discriminant))


@dataclass(frozen=True)
class JointBinding:
    """Resolved motor-to-joint index binding."""

    motor_id: int
    joint_name: str
    q_index: int
    v_index: int


class RobotModel:
    """Pinocchio-backed kinematics/dynamics for the configured motor set.

    All public methods accept and return values keyed by ``motor_id`` (not
    pinocchio internal ``q_index``). Joints not in ``motor_joint_map`` remain
    at their neutral configuration during computations.
    """

    def __init__(self, urdf_path: str | Path, motor_joint_map: Mapping[int, str]):
        path = Path(urdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"URDF not found: {path}")
        if not motor_joint_map:
            raise ValueError("motor_joint_map must not be empty")

        self.urdf_path = path
        self.model = pin.buildModelFromUrdf(str(path))
        self.data = self.model.createData()

        bindings: dict[int, JointBinding] = {}
        for motor_id, joint_name in sorted(motor_joint_map.items()):
            joint_id = self.model.getJointId(joint_name)
            if joint_id <= 0:
                raise ValueError(
                    f"joint not found in URDF: {joint_name} (motor_id={motor_id})"
                )
            if self.model.nqs[joint_id] != 1 or self.model.nvs[joint_id] != 1:
                raise ValueError(
                    f"joint must be 1-DoF: {joint_name} "
                    f"(nq={self.model.nqs[joint_id]}, nv={self.model.nvs[joint_id]})"
                )
            bindings[int(motor_id)] = JointBinding(
                motor_id=int(motor_id),
                joint_name=joint_name,
                q_index=int(self.model.idx_qs[joint_id]),
                v_index=int(self.model.idx_vs[joint_id]),
            )

        self.bindings = bindings
        self.ordered_motor_ids = tuple(sorted(bindings.keys()))
        self._q_indices = [self.bindings[mid].q_index for mid in self.ordered_motor_ids]
        self._v_indices = [self.bindings[mid].v_index for mid in self.ordered_motor_ids]

        self._q_neutral = pin.neutral(self.model)
        self._q_buf = self._q_neutral.copy()
        self._lock = threading.Lock()

    def _fill_q_buf(self, q_by_motor: Mapping[int, float]) -> None:
        self._q_buf[:] = self._q_neutral
        for motor_id in self.ordered_motor_ids:
            if motor_id not in q_by_motor:
                raise KeyError(f"missing q for motor_id={motor_id}")
            self._q_buf[self.bindings[motor_id].q_index] = float(q_by_motor[motor_id])

    def _resolve_frame_id(self, frame_name: str) -> int:
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"frame not found in URDF: {frame_name}")
        return int(frame_id)

    def gravity_torque(self, q_by_motor: Mapping[int, float]) -> dict[int, float]:
        """Gravity-compensation torques keyed by motor_id."""
        with self._lock:
            self._fill_q_buf(q_by_motor)
            tau = pin.computeGeneralizedGravity(self.model, self.data, self._q_buf)
            return {
                motor_id: float(tau[self.bindings[motor_id].v_index])
                for motor_id in self.ordered_motor_ids
            }

    def inertia_ff_torque(
        self,
        q_by_motor: Mapping[int, float],
        qd_by_motor: Mapping[int, float],
        qdd_by_motor: Mapping[int, float],
    ) -> dict[int, float]:
        """Inertia + Coriolis feedforward: RNEA(q,qd,qdd) − gravity(q)."""
        nv = self.model.nv
        qd_model = np.zeros(nv)
        qdd_model = np.zeros(nv)
        for motor_id in self.ordered_motor_ids:
            vi = self.bindings[motor_id].v_index
            qd_model[vi]  = float(qd_by_motor[motor_id])
            qdd_model[vi] = float(qdd_by_motor[motor_id])
        with self._lock:
            self._fill_q_buf(q_by_motor)
            tau_rnea = pin.rnea(self.model, self.data, self._q_buf, qd_model, qdd_model)
            tau_g    = pin.computeGeneralizedGravity(self.model, self.data, self._q_buf)
            return {
                motor_id: float(tau_rnea[self.bindings[motor_id].v_index]
                                - tau_g[self.bindings[motor_id].v_index])
                for motor_id in self.ordered_motor_ids
            }

    def forward_kinematics(
        self, q_by_motor: Mapping[int, float], frame_name: str
    ) -> pin.SE3:
        """World-frame placement (``SE3``) of the named frame."""
        frame_id = self._resolve_frame_id(frame_name)
        with self._lock:
            self._fill_q_buf(q_by_motor)
            pin.forwardKinematics(self.model, self.data, self._q_buf)
            pin.updateFramePlacement(self.model, self.data, frame_id)
            return self.data.oMf[frame_id].copy()

    def jacobian(
        self, q_by_motor: Mapping[int, float], frame_name: str
    ) -> np.ndarray:
        """6×N Jacobian (LOCAL_WORLD_ALIGNED) with columns ordered by ``ordered_motor_ids``."""
        frame_id = self._resolve_frame_id(frame_name)
        with self._lock:
            self._fill_q_buf(q_by_motor)
            pin.computeJointJacobians(self.model, self.data, self._q_buf)
            pin.updateFramePlacements(self.model, self.data)
            J_full = pin.getFrameJacobian(
                self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            return np.asarray(J_full)[:, self._v_indices].copy()

    def mass_matrix(self, q_by_motor: Mapping[int, float]) -> np.ndarray:
        """N×N joint-space mass matrix for controlled motors."""
        with self._lock:
            self._fill_q_buf(q_by_motor)
            pin.crba(self.model, self.data, self._q_buf)
            M_full = np.asarray(self.data.M)
            return M_full[np.ix_(self._v_indices, self._v_indices)].copy()

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` joint position limits ordered by ``ordered_motor_ids``.

        Non-finite limits (joints with no URDF ``<limit>``) become ``±inf``.
        """
        lower_full = np.asarray(self.model.lowerPositionLimit, dtype=float)
        upper_full = np.asarray(self.model.upperPositionLimit, dtype=float)
        lower = lower_full[self._q_indices]
        upper = upper_full[self._q_indices]
        lower = np.where(np.isfinite(lower), lower, -np.inf)
        upper = np.where(np.isfinite(upper), upper, np.inf)
        return lower, upper

    def gripper_width(self, motor_angle: float) -> float:
        """Gripper motor angle [rad] → finger_r prismatic displacement [m].

        Crank-slider kinematics: r=32.5 mm, l=45 mm.
        0 rad = open (0 mm), +1.4099 rad = closed (~40.7 mm theoretical / 42 mm measured).
        """
        theta = max(0.0, min(float(motor_angle), _GRIPPER_THETA_CLOSED))
        return _gripper_disp(theta)

    def gripper_motor_from_width(self, width_m: float) -> float:
        """Finger_r prismatic displacement [m] → gripper motor angle [rad].

        Inverse of gripper_width via bisection. Result clamped to [0, 1.4099] rad.
        """
        d_max = _gripper_disp(_GRIPPER_THETA_CLOSED)
        d = max(0.0, min(float(width_m), d_max))
        lo, hi = 0.0, _GRIPPER_THETA_CLOSED
        for _ in range(52):
            mid = (lo + hi) * 0.5
            if _gripper_disp(mid) < d:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5
