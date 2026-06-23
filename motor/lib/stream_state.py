"""mit_stream 실행 상태 공유(진단 모니터용)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

STATE_FILE_ENV = "MOTOR_MIT_STREAM_STATE"
DEFAULT_STATE_FILE = "/tmp/motor_mit_stream_state.json"


def _state_path() -> Path:
    return Path(os.environ.get(STATE_FILE_ENV, DEFAULT_STATE_FILE))


def _load_state() -> dict:
    path = _state_path()
    try:
        with path.open("r", encoding="utf-8") as fp:
            obj = json.load(fp)
            if isinstance(obj, dict):
                return obj
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return {"streams": {}, "updated_at": 0.0}


def _write_state(state: dict):
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(state, fp, ensure_ascii=True, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def publish_mit_stream(can_ids: list[int], *, hz: float, channel: str, host_id: int):
    now = time.time()
    state = _load_state()
    streams = state.get("streams")
    if not isinstance(streams, dict):
        streams = {}
        state["streams"] = streams

    for can_id in can_ids:
        streams[str(int(can_id))] = {
            "hz": float(hz),
            "channel": str(channel),
            "host_id": int(host_id),
            "updated_at": now,
        }

    state["updated_at"] = now
    _write_state(state)


def clear_mit_stream(can_ids: list[int]):
    state = _load_state()
    streams = state.get("streams")
    if not isinstance(streams, dict):
        return

    for can_id in can_ids:
        streams.pop(str(int(can_id)), None)

    state["updated_at"] = time.time()
    _write_state(state)


def load_expected_hz_map(
    *,
    can_ids: set[int] | None = None,
    channel: str | None = None,
    max_age_sec: float | None = 3.0,
) -> dict[int, float]:
    now = time.time()
    state = _load_state()
    streams = state.get("streams")
    if not isinstance(streams, dict):
        return {}

    out: dict[int, float] = {}
    for key, meta in streams.items():
        if not isinstance(meta, dict):
            continue

        try:
            can_id = int(key, 10)
        except Exception:
            continue
        if can_ids is not None and can_id not in can_ids:
            continue

        hz = meta.get("hz")
        if not isinstance(hz, (int, float)) or hz <= 0:
            continue

        if channel is not None and meta.get("channel") != channel:
            continue

        updated_at = meta.get("updated_at")
        if max_age_sec is not None:
            if not isinstance(updated_at, (int, float)):
                continue
            if now - float(updated_at) > max_age_sec:
                continue

        out[can_id] = float(hz)
    return out
