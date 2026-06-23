"""Small shared helpers for image/depth ROS messages."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


def stamp_ns(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def depth_values_to_meters(
    values: np.ndarray,
    *,
    encoding: str,
    depth_scale: float,
) -> np.ndarray:
    if values.dtype == np.uint16 or "16U" in encoding.upper():
        return values.astype(np.float32) * float(depth_scale)
    return values.astype(np.float32)
