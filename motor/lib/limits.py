"""Robot-side position limits for decoded motor feedback."""

import math

from .frames import mit_ranges_for
from .parse import Feedback

POSITION_LIMIT_BY_MOTOR = {
    1: (-math.pi, math.pi),
    2: (-1.78, 1.78),
    3: (-math.pi, math.pi),
    4: (-math.pi, math.pi),
    5: (-math.pi, math.pi),
    6: (-math.pi, math.pi),
    7: (0.0, 1.3599),
}


def position_limits_for_motor(motor_id: int) -> tuple[float, float]:
    spec = mit_ranges_for(motor_id)
    q_min, q_max = POSITION_LIMIT_BY_MOTOR.get(int(motor_id), (spec.P_MIN, spec.P_MAX))
    q_min = max(spec.P_MIN, float(q_min))
    q_max = min(spec.P_MAX, float(q_max))
    if q_min > q_max:
        return spec.P_MIN, spec.P_MAX
    return q_min, q_max


def feedback_position_violation(fb: Feedback, eps: float = 1e-3) -> tuple[float, float, float] | None:
    q_min, q_max = position_limits_for_motor(fb.motor_id)
    q_limited = min(max(fb.pos, q_min), q_max)
    if abs(q_limited - fb.pos) <= eps:
        return None
    return q_limited, q_min, q_max
