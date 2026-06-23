"""Compatibility wrapper for shared idle_common.param_store."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_idle_common_importable():
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "src" / "idle_common"
    if candidate.is_dir():
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


_ensure_idle_common_importable()

from idle_common.param_store import *  # noqa: F401,F403,E402
