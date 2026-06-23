"""Control tuning workflow and gating utilities."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .param_store import (
    control_original_path,
    control_tuned_path,
    initialize_tuned_from_original_if_missing,
    load_control_effective,
    load_control_tuned,
    lock_path,
    save_control_tuned,
)

STATE_FILE_ENV = "IDLE_CONTROL_GATE_STATE"
DEFAULT_STATE_FILE = "/tmp/idle_control_gate_state.json"

CONTROL_TUNING_KEYS = {
    "kp",
    "kd",
    "q_des",
    "qd_des",
    "tau_ff",
    "gravity_scale",
    "gravity_bias",
    "inertia_ff_scale",
    "v_max",
    "a_max",
    "profile_sharpness",
    "goal_mode",
}


def state_file_path() -> Path:
    return Path(os.environ.get(STATE_FILE_ENV, DEFAULT_STATE_FILE))


def _read_state() -> dict[str, Any]:
    path = state_file_path()
    try:
        with path.open("r", encoding="utf-8") as fp:
            obj = json.load(fp)
    except FileNotFoundError:
        obj = {}
    except json.JSONDecodeError as exc:
        print(f"warning: gate state file corrupt at {path}: {exc}", file=sys.stderr)
        obj = {}
    except Exception:
        obj = {}

    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("control_active", False)
    obj.setdefault("updated_at", 0.0)
    return obj


def _write_state(state: dict[str, Any]):
    path = state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(state, fp, ensure_ascii=True, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def set_control_active(active: bool):
    state = _read_state()
    state["control_active"] = bool(active)
    state["updated_at"] = time.time()
    _write_state(state)


def is_control_active() -> bool:
    return bool(_read_state().get("control_active", False))


def assert_control_inactive():
    if is_control_active():
        raise RuntimeError("control is active: stop control before modifying/saving parameters")


@contextmanager
def control_session():
    set_control_active(True)
    try:
        yield
    finally:
        set_control_active(False)


def _validate_control_doc(doc: dict[str, Any]):
    if not isinstance(doc, dict):
        raise ValueError("control params must be a dict")

    defaults = doc.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a dict")
    motors = doc.get("motors", {})
    if not isinstance(motors, dict):
        raise ValueError("motors must be a dict")

    for dkey, dval in defaults.items():
        if dkey in CONTROL_TUNING_KEYS:
            _validate_tuning_value(dkey, dval)

    for motor_id, motor_cfg in motors.items():
        int(motor_id, 10)
        if not isinstance(motor_cfg, dict):
            raise ValueError(f"motors.{motor_id} must be a dict")
        for key, value in motor_cfg.items():
            if key not in CONTROL_TUNING_KEYS:
                continue
            _validate_tuning_value(key, value)


def _validate_tuning_value(key: str, value: Any):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    f = float(value)
    if key in {"kp", "kd"} and f < 0:
        raise ValueError(f"{key} must be >= 0")


def set_control_tuning(motor_ids: list[int], updates: dict[str, float]) -> Path:
    if not updates:
        raise ValueError("no updates provided")
    bad_keys = [k for k in updates if k not in CONTROL_TUNING_KEYS]
    if bad_keys:
        raise ValueError(f"unsupported control keys: {bad_keys}")

    assert_control_inactive()
    initialize_tuned_from_original_if_missing()
    tuned_path = control_tuned_path()
    lock_file = tuned_path.with_suffix(tuned_path.suffix + ".lock")

    with lock_path(lock_file):
        doc = load_control_tuned()
        if not doc:
            doc = load_control_effective()

        motors = doc.setdefault("motors", {})
        if not isinstance(motors, dict):
            raise ValueError("motors must be a dict")

        for motor_id in motor_ids:
            mcfg = motors.setdefault(str(int(motor_id)), {})
            if not isinstance(mcfg, dict):
                raise ValueError(f"motors.{motor_id} must be a dict")
            for key, value in updates.items():
                _validate_tuning_value(key, value)
                mcfg[key] = float(value)

        _validate_control_doc(doc)
        save_control_tuned(doc)
    return tuned_path


def save_control_tuning() -> Path:
    assert_control_inactive()
    initialize_tuned_from_original_if_missing()
    tuned_path = control_tuned_path()
    lock_file = tuned_path.with_suffix(tuned_path.suffix + ".lock")
    with lock_path(lock_file):
        doc = load_control_tuned()
        if not doc:
            doc = load_control_effective()
        _validate_control_doc(doc)
        save_control_tuned(doc)
    return tuned_path


_EFFECTIVE_CACHE: dict[str, Any] | None = None
_EFFECTIVE_CACHE_MTIME: tuple[float, float] = (0.0, 0.0)


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _get_cached_effective() -> dict[str, Any]:
    global _EFFECTIVE_CACHE, _EFFECTIVE_CACHE_MTIME
    mtimes = (
        _file_mtime(control_original_path()),
        _file_mtime(control_tuned_path()),
    )
    if _EFFECTIVE_CACHE is None or mtimes != _EFFECTIVE_CACHE_MTIME:
        _EFFECTIVE_CACHE = load_control_effective()
        _EFFECTIVE_CACHE_MTIME = mtimes
    return _EFFECTIVE_CACHE


def control_params_for_motor(motor_id: int) -> dict[str, float]:
    effective = _get_cached_effective()
    defaults = effective.get("defaults", {})
    motors = effective.get("motors", {})

    out: dict[str, float] = {}
    if isinstance(defaults, dict):
        for key in CONTROL_TUNING_KEYS:
            if key in defaults and isinstance(defaults[key], (int, float)):
                out[key] = float(defaults[key])

    if isinstance(motors, dict):
        mcfg = motors.get(str(int(motor_id)))
        if isinstance(mcfg, dict):
            for key in CONTROL_TUNING_KEYS:
                if key in mcfg and isinstance(mcfg[key], (int, float)):
                    out[key] = float(mcfg[key])
    return out
