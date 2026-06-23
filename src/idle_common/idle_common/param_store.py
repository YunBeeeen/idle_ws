"""Shared parameter store utilities for driver/control split files."""

from __future__ import annotations

import copy
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

PARAM_ROOT_ENV = "IDLE_PARAM_ROOT"
DEFAULT_SCAN_MIN_ID = 1
DEFAULT_SCAN_MAX_ID = 10
DEFAULT_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]

DRIVER_ORIGINAL_REL = Path("original/driver_params.yaml")
CONTROL_ORIGINAL_REL = Path("original/control_params.yaml")
CONTROL_TUNED_REL = Path("tuned/control_params.yaml")


def _repo_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    out: list[Path] = []
    for base in [cwd, *cwd.parents, here.parent, *here.parents]:
        cand = base / "param"
        if cand not in out:
            out.append(cand)
    return out


def resolve_param_root() -> Path:
    root = os.environ.get(PARAM_ROOT_ENV, "").strip()
    if root:
        return Path(root).expanduser().resolve()

    for cand in _repo_candidates():
        if cand.is_dir():
            return cand

    return (Path(__file__).resolve().parents[3] / "param").resolve()


def driver_original_path() -> Path:
    return resolve_param_root() / DRIVER_ORIGINAL_REL


def control_original_path() -> Path:
    return resolve_param_root() / CONTROL_ORIGINAL_REL


def control_tuned_path() -> Path:
    return resolve_param_root() / CONTROL_TUNED_REL


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _empty_control() -> dict[str, Any]:
    return {
        "version": 1,
        "scan": {"min_id": DEFAULT_SCAN_MIN_ID, "max_id": DEFAULT_SCAN_MAX_ID},
        "defaults": {
            "kp": 0.0,
            "kd": 0.0,
            "q_des": 0.0,
            "qd_des": 0.0,
            "tau_ff": 0.0,
            "gravity_scale": 1.0,
            "gravity_bias": 0.0,
        },
        "motors": {str(i): {} for i in DEFAULT_MOTOR_IDS},
    }


def _empty_driver() -> dict[str, Any]:
    return {
        "version": 1,
        "motors": {str(i): {} for i in DEFAULT_MOTOR_IDS},
    }


def atomic_write_yaml(path: Path, data: dict[str, Any]):
    _ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            yaml.safe_dump(data, fp, sort_keys=True, allow_unicode=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def load_yaml(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(default) if default is not None else {}
    with path.open("r", encoding="utf-8") as fp:
        doc = yaml.safe_load(fp)
    if isinstance(doc, dict):
        return doc
    return copy.deepcopy(default) if default is not None else {}


def ensure_base_files():
    dpath = driver_original_path()
    cpath = control_original_path()
    if not dpath.exists():
        atomic_write_yaml(dpath, _empty_driver())
    if not cpath.exists():
        atomic_write_yaml(cpath, _empty_control())


@contextmanager
def lock_path(lock_file: Path):
    _ensure_parent(lock_file)
    with lock_file.open("a+", encoding="utf-8") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_control_original() -> dict[str, Any]:
    ensure_base_files()
    return load_yaml(control_original_path(), default=_empty_control())


def load_control_tuned() -> dict[str, Any]:
    return load_yaml(control_tuned_path(), default={})


def load_control_effective() -> dict[str, Any]:
    return _merge_dict(load_control_original(), load_control_tuned())


def save_control_tuned(doc: dict[str, Any]):
    atomic_write_yaml(control_tuned_path(), doc)


def initialize_tuned_from_original_if_missing() -> bool:
    ensure_base_files()
    tpath = control_tuned_path()
    if tpath.exists():
        return False
    atomic_write_yaml(tpath, load_control_original())
    return True


def load_driver_original() -> dict[str, Any]:
    ensure_base_files()
    return load_yaml(driver_original_path(), default=_empty_driver())
