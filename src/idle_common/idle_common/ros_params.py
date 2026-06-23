"""Typed ROS2 parameter declaration helper."""

from __future__ import annotations

from typing import Any, Callable


def declare_typed(
    node: Any,
    name: str,
    default: Any,
    cast: Callable[[Any], Any] | None = None,
) -> Any:
    """Declare a ROS parameter and return its value cast via ``cast``.

    If ``cast`` is None, it defaults to ``type(default)`` which covers the
    common cases (float/int/bool/str). For transforms like trimming whitespace,
    pass an explicit callable, e.g. ``lambda v: str(v).strip()``.
    """

    if cast is None:
        cast = type(default)
    return cast(node.declare_parameter(name, default).value)
