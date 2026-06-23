"""Pinocchio-based gravity compensation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pinocchio as pin

from idle_common.motor_map import (
    DEFAULT_MOTOR_JOINT_MAP,
    DEFAULT_TAU_LIMIT_BY_MOTOR,
    parse_float_map_json,
    parse_motor_joint_map_json,
)

__all__ = [
    "DEFAULT_MOTOR_JOINT_MAP",
    "DEFAULT_TAU_LIMIT_BY_MOTOR",
    "GravityCompensator",
    "JointBinding",
    "parse_float_map_json",
    "parse_motor_joint_map_json",
]


@dataclass(frozen=True)
class JointBinding:
    """Resolved motor-to-joint index binding."""

    motor_id: int
    joint_name: str
    q_index: int
    v_index: int


class GravityCompensator:
    """Gravity torque calculator for selected URDF joints."""

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
                raise ValueError(f"joint not found in URDF: {joint_name} (motor_id={motor_id})")
            if self.model.nqs[joint_id] != 1 or self.model.nvs[joint_id] != 1:
                raise ValueError(
                    f"joint must be 1-DoF: {joint_name} "
                    f"(nq={self.model.nqs[joint_id]}, nv={self.model.nvs[joint_id]})"
                )
            bindings[motor_id] = JointBinding(
                motor_id=int(motor_id),
                joint_name=joint_name,
                q_index=int(self.model.idx_qs[joint_id]),
                v_index=int(self.model.idx_vs[joint_id]),
            )

        self.bindings = bindings
        self.ordered_motor_ids = tuple(sorted(bindings.keys()))

    def compute_gravity_by_motor(self, q_by_motor: Mapping[int, float]) -> dict[int, float]:
        """Compute gravity torques keyed by motor id."""

        q_model = pin.neutral(self.model)
        for motor_id in self.ordered_motor_ids:
            if motor_id not in q_by_motor:
                raise KeyError(f"missing q for motor_id={motor_id}")
            binding = self.bindings[motor_id]
            q_model[binding.q_index] = float(q_by_motor[motor_id])

        tau_model = pin.computeGeneralizedGravity(self.model, self.data, q_model)
        out: dict[int, float] = {}
        for motor_id in self.ordered_motor_ids:
            binding = self.bindings[motor_id]
            out[motor_id] = float(tau_model[binding.v_index])
        return out
