"""Quintic (minimum-jerk style) joint trajectory helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_JERK_VEL_FACTOR = 1.875
MIN_JERK_ACC_FACTOR = 5.7735


@dataclass(frozen=True)
class QuinticPlan:
    """Planned quintic trajectory coefficients for each joint."""

    duration: float
    coeffs: np.ndarray  # shape: [dof, 6], coefficients a0..a5
    q_start: np.ndarray
    q_goal: np.ndarray


def _safe_positive(values: np.ndarray, default_value: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = default_value
    out[out <= 0.0] = default_value
    return out


def estimate_minimum_jerk_duration(
    delta_q: np.ndarray,
    v_max: np.ndarray,
    a_max: np.ndarray,
    min_duration: float = 0.05,
) -> float:
    """Estimate a conservative duration satisfying velocity/acceleration bounds."""

    dq = np.abs(np.asarray(delta_q, dtype=float))
    vmax = _safe_positive(np.asarray(v_max, dtype=float), 1.0)
    amax = _safe_positive(np.asarray(a_max, dtype=float), 1.0)

    t_vel = MIN_JERK_VEL_FACTOR * dq / vmax
    t_acc = np.sqrt(MIN_JERK_ACC_FACTOR * dq / amax)
    duration = float(np.max(np.maximum(t_vel, t_acc)))
    return max(float(min_duration), duration)


def plan_quintic(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    v_start: np.ndarray,
    v_goal: np.ndarray,
    v_max: np.ndarray,
    a_max: np.ndarray,
    min_duration: float = 0.05,
) -> QuinticPlan:
    """Create a synchronized multi-joint quintic trajectory."""

    q0 = np.asarray(q_start, dtype=float)
    qf = np.asarray(q_goal, dtype=float)
    v0 = np.asarray(v_start, dtype=float)
    vf = np.asarray(v_goal, dtype=float)

    if q0.shape != qf.shape or q0.shape != v0.shape or q0.shape != vf.shape:
        raise ValueError("q_start, q_goal, v_start, v_goal must share shape")

    duration = estimate_minimum_jerk_duration(qf - q0, v_max, a_max, min_duration=min_duration)
    t = duration
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    # Boundary conditions: q(0)=q0, q(T)=qf, qd(0)=v0, qd(T)=vf, qdd(0)=qdd(T)=0.
    a0 = q0
    a1 = v0
    a2 = np.zeros_like(q0)
    a3 = (20.0 * (qf - q0) - (8.0 * vf + 12.0 * v0) * t) / (2.0 * t3)
    a4 = (30.0 * (q0 - qf) + (14.0 * vf + 16.0 * v0) * t) / (2.0 * t4)
    a5 = (12.0 * (qf - q0) - (6.0 * vf + 6.0 * v0) * t) / (2.0 * t5)
    coeffs = np.stack([a0, a1, a2, a3, a4, a5], axis=1)
    return QuinticPlan(duration=duration, coeffs=coeffs, q_start=q0, q_goal=qf)


def sample_quintic(plan: QuinticPlan, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sample planned trajectory at elapsed time."""

    t = float(np.clip(elapsed_s, 0.0, plan.duration))
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    a0 = plan.coeffs[:, 0]
    a1 = plan.coeffs[:, 1]
    a2 = plan.coeffs[:, 2]
    a3 = plan.coeffs[:, 3]
    a4 = plan.coeffs[:, 4]
    a5 = plan.coeffs[:, 5]

    q = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
    qd = a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3 + 5.0 * a5 * t4
    done = elapsed_s >= plan.duration
    return q, qd, done
