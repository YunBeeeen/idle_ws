"""HSV color presets and contour quality checks for colored box detection."""

from __future__ import annotations

import json
from typing import Optional

import cv2
import numpy as np

HsvRange = tuple[tuple[int, int, int], tuple[int, int, int]]


HSV_PRESETS: dict[str, list[HsvRange]] = {
    "red": [((0, 130, 80), (12, 255, 190)), ((170, 130, 80), (180, 255, 190))],
    "orange": [((5, 80, 40), (25, 255, 255))],
    "yellow": [((20, 70, 50), (38, 255, 255))],
    "green": [((28, 90, 65), (58, 230, 150))],
    "blue": [((85, 130, 110), (112, 255, 190))],
    "purple": [((128, 50, 35), (165, 255, 255))],
    "white": [((0, 0, 165), (180, 70, 255))],
    "black": [((0, 0, 0), (180, 255, 70))],
}

AUTO_TARGET_COLORS = ("red", "green", "blue")


def _hsv_mask(
    hsv: np.ndarray,
    ranges: list[HsvRange],
) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
    return mask


def color_ratio_mask(color_bgr: np.ndarray, color_key: str) -> np.ndarray:
    """Return a mask for pixels whose B/G/R channel ratios match a color.

    HSV handles hue ranges; this normalized BGR check makes the detector less
    sensitive to brightness changes and rejects many floor/background pixels.
    """
    key = color_key.lower()
    if key not in AUTO_TARGET_COLORS:
        return np.full(color_bgr.shape[:2], 255, dtype=np.uint8)

    bgr = color_bgr.astype(np.float32)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    total = np.maximum(b + g + r, 1.0)
    b_ratio = b / total
    g_ratio = g / total
    r_ratio = r / total

    if key == "red":
        keep = (r_ratio >= 0.45) & ((r - g) >= 20.0) & ((r - b) >= 20.0)
    elif key == "green":
        keep = (g_ratio >= 0.33) & ((g - b) >= 5.0) & ((g - r) >= 2.0)
    else:
        keep = (b_ratio >= 0.38) & ((b - r) >= 20.0) & ((b - g) >= -20.0)

    return np.where(keep, 255, 0).astype(np.uint8)


def make_color_mask(
    color_bgr: np.ndarray,
    hsv: np.ndarray,
    color_key: str,
    ranges: list[HsvRange],
    *,
    use_ratio: bool = True,
) -> np.ndarray:
    mask = _hsv_mask(hsv, ranges)
    if use_ratio:
        mask = cv2.bitwise_and(mask, color_ratio_mask(color_bgr, color_key))
    return mask


def contour_color_stats(
    color_bgr: np.ndarray,
    hsv: np.ndarray,
    contour,
) -> Optional[dict[str, float]]:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    hsv_values = hsv[mask > 0]
    bgr_values = color_bgr[mask > 0]
    if hsv_values.size == 0 or bgr_values.size == 0:
        return None

    hsv_median = np.median(hsv_values, axis=0)
    bgr_median = np.median(bgr_values, axis=0)
    b, g, r = (float(v) for v in bgr_median)
    total = max(b + g + r, 1.0)
    return {
        "h_median": float(hsv_median[0]),
        "s_median": float(hsv_median[1]),
        "v_median": float(hsv_median[2]),
        "b_median": b,
        "g_median": g,
        "r_median": r,
        "b_ratio": b / total,
        "g_ratio": g / total,
        "r_ratio": r / total,
    }


def passes_color_quality(color_key: str, stats: Optional[dict[str, float]]) -> bool:
    """Reject low-quality color blobs after broad HSV masking.

    The masks are intentionally a little wider so the box contour stays connected.
    This median check keeps broad masks from accepting reflections/background.
    """
    if stats is None:
        return False
    key = color_key.lower()
    s = stats["s_median"]
    b = stats["b_median"]
    g = stats["g_median"]
    r = stats["r_median"]

    if key == "blue":
        return (
            85.0 <= stats["h_median"] <= 112.0
            and 150.0 <= s <= 255.0
            and 115.0 <= stats["v_median"] <= 190.0
            and stats["b_ratio"] >= 0.35
            and (b - r) >= 15.0
            and (b - g) >= -25.0
        )
    if key == "green":
        return (
            28.0 <= stats["h_median"] <= 58.0
            and 100.0 <= s <= 230.0
            and 65.0 <= stats["v_median"] <= 150.0
            and stats["g_ratio"] >= 0.32
            and (g - b) >= 3.0
            and (g - r) >= 0.0
        )
    if key == "red":
        hue = stats["h_median"]
        return (
            (hue <= 18.0 or hue >= 162.0)
            and 145.0 <= s <= 255.0
            and 80.0 <= stats["v_median"] <= 190.0
            and stats["r_ratio"] >= 0.42
            and (r - g) >= 15.0
            and (r - b) >= 15.0
        )
    return True


def parse_hsv_ranges_json(text: str) -> list[HsvRange]:
    """Return HSV ranges as [((h,s,v), (h,s,v)), ...]."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"hsv_ranges_json parse failed: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ValueError("hsv_ranges_json must be a non-empty JSON list")

    ranges = []
    for item in raw:
        if (
            isinstance(item, list)
            and len(item) == 6
            and all(isinstance(x, (int, float)) for x in item)
        ):
            lower = tuple(int(x) for x in item[:3])
            upper = tuple(int(x) for x in item[3:])
        elif (
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(part, list) and len(part) == 3 for part in item)
        ):
            lower = tuple(int(x) for x in item[0])
            upper = tuple(int(x) for x in item[1])
        else:
            raise ValueError(
                "each HSV range must be [h0,s0,v0,h1,s1,v1] "
                "or [[h0,s0,v0],[h1,s1,v1]]"
            )
        ranges.append((lower, upper))
    return ranges


def parse_hsv_ranges(
    *,
    target_color: str,
    hsv_ranges_json: str,
) -> list[HsvRange]:
    if hsv_ranges_json.strip():
        return parse_hsv_ranges_json(hsv_ranges_json)

    key = target_color.strip().lower()
    if key in ("", "auto"):
        key = AUTO_TARGET_COLORS[0]
    if key not in HSV_PRESETS:
        known = ", ".join(["auto", "all"] + sorted(HSV_PRESETS.keys()))
        raise ValueError(f"unknown target_color '{target_color}'. Known: {known}")
    return HSV_PRESETS[key]


def resolve_detector_ranges(
    *,
    target_color: str,
    hsv_ranges_json: str,
) -> dict[str, list[HsvRange]]:
    if hsv_ranges_json.strip():
        key = target_color.strip().lower()
        if key in ("", "auto", "all"):
            key = "custom"
        return {key: parse_hsv_ranges_json(hsv_ranges_json)}

    key = target_color.strip().lower()
    if key in ("", "auto"):
        return {color: HSV_PRESETS[color] for color in AUTO_TARGET_COLORS}
    if key == "all":
        return dict(HSV_PRESETS)
    if key not in HSV_PRESETS:
        known = ", ".join(["auto", "all"] + sorted(HSV_PRESETS.keys()))
        raise ValueError(f"unknown target_color '{target_color}'. Known: {known}")
    return {key: HSV_PRESETS[key]}
