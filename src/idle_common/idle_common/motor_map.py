"""Shared motor-to-joint mapping and torque-limit defaults."""

from __future__ import annotations

import json


DEFAULT_MOTOR_JOINT_MAP: dict[int, str] = {
    1: "j1",
    2: "j2",
    3: "j3",
    4: "j4",
    5: "j5",
    6: "j6",
}

# Motor 7 (gripper) is the smallest — external force estimation has poor SNR there.
DEFAULT_TAU_LIMIT_BY_MOTOR: dict[int, float] = {
    1: 6.0,
    2: 28.0,
    3: 10.0,
    4: 6.0,
    5: 5.0,
    6: 5.0,
    7: 1.6,
}


def _parse_object_json(text: str, field_name: str) -> dict[object, object]:
    text_stripped = text.strip()
    if not text_stripped:
        return {}
    try:
        raw = json.loads(text_stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON object: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return raw


def parse_motor_joint_map_json(text: str, field_name: str = "motor_joint_map_json") -> dict[int, str]:
    """Parse motor_id->joint_name JSON mapping."""

    raw = _parse_object_json(text, field_name)
    out: dict[int, str] = {}
    for key, value in raw.items():
        try:
            motor_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}: invalid motor id key '{key}'") from exc
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}: motor {motor_id} joint must be non-empty string")
        out[motor_id] = value.strip()
    return out


def parse_float_map_json(text: str, field_name: str) -> dict[int, float]:
    """Parse motor_id->float JSON mapping."""

    raw = _parse_object_json(text, field_name)
    out: dict[int, float] = {}
    for key, value in raw.items():
        try:
            motor_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}: invalid motor id key '{key}'") from exc
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name}: motor {motor_id} value must be numeric")
        out[motor_id] = float(value)
    return out
