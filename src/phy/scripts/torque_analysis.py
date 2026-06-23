"""동적 토크 분석 — 워크스페이스 대표 자세에서 정적/동적 토크 측정.

실행:
    source ~/idle_ws/install/setup.bash
    python3 src/phy/scripts/torque_analysis.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_WS = Path(__file__).resolve().parents[3]
for _pkg in ("phy", "idle_common"):
    for _p in (_WS / "install" / _pkg / "lib").glob("python3*"):
        _dp = _p / "dist-packages"
        if _dp.exists() and str(_dp) not in sys.path:
            sys.path.insert(0, str(_dp))

URDF_PATH = str(_WS / "install/sim/share/sim/urdf/robot.urdf")

from phy.ik import IKConfig, IKSolver
from phy.plan import top_down_R
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP, DEFAULT_TAU_LIMIT_BY_MOTOR
import pinocchio as pin

joints = tuple(DEFAULT_MOTOR_JOINT_MAP[m] for m in sorted(DEFAULT_MOTOR_JOINT_MAP))
ik = IKSolver(URDF_PATH, IKConfig(target_frame="gripper", controlled_joints=joints))
model = ik.model
data = ik.data

TAU_LIMITS = np.array([DEFAULT_TAU_LIMIT_BY_MOTOR[m] for m in sorted(DEFAULT_MOTOR_JOINT_MAP)])

IK_TOL = 0.005

def _q_model(q_ord):
    return ik._ordered_to_model_q(ik.clip_to_limits(q_ord))

def static_torque(q_ord):
    """중력 토크 (정적, q_dot=0, q_ddot=0)."""
    qm = _q_model(q_ord)
    tau = pin.computeGeneralizedGravity(model, data, qm)
    return np.array([tau[vi] for vi in ik.v_indices])

def dynamic_torque(q_ord, qd_ord, qdd_ord):
    """RNEA 동적 토크."""
    qm = _q_model(q_ord)
    dof = model.nv
    qd_m = np.zeros(dof)
    qdd_m = np.zeros(dof)
    for i, vi in enumerate(ik.v_indices):
        qd_m[vi] = qd_ord[i]
        qdd_m[vi] = qdd_ord[i]
    tau = pin.rnea(model, data, qm, qd_m, qdd_m)
    return np.array([tau[vi] for vi in ik.v_indices])

def utilization(tau):
    return np.max(np.abs(tau) / TAU_LIMITS)

# 워크스페이스 대표 타겟
targets = [
    (0.25, 0.00, 0.025, "near"),
    (0.35, 0.00, 0.025, "mid"),
    (0.45, 0.00, 0.025, "far"),
    (0.35, 0.15, 0.025, "left"),
    (0.35, 0.00, 0.150, "high"),
]

rng = np.random.default_rng(42)
R_down = top_down_R(0.0)

hdr = f"{'타겟':>22}  " + "  ".join(f"{j:>6}" for j in joints) + f"  {'worst':>6}  flag"
print(hdr)
print("-" * len(hdr))

for x, y, z, label in targets:
    xyz = np.array([x, y, z])
    R = top_down_R(math.atan2(y, x))

    seeds = ik.heuristic_seeds_from_target(xyz)
    if not seeds:
        seeds = [np.zeros(6)]
    best = None
    for s in seeds:
        res = ik.solve_pose(xyz, R, s)
        if best is None or res.residual_norm < best.residual_norm:
            best = res
    if best is None or best.residual_norm > IK_TOL:
        print(f"  {label:>22}  IK failed")
        continue
    q = best.q

    # 동적 토크: RNEA with peak quintic velocity/acceleration
    dq_norm = np.linalg.norm(q)
    if dq_norm > 0:
        q_dot  = q / dq_norm * 0.5   # v_max = 0.5 rad/s
        q_ddot = q / dq_norm * 1.0   # a_max = 1.0 rad/s²
    else:
        q_dot = q_ddot = np.zeros(6)

    tau_d = dynamic_torque(q, q_dot, q_ddot)

    # 관절별 토크 리미트 대비 비율
    ratio = np.abs(tau_d) / TAU_LIMITS
    worst_idx = np.argmax(ratio)
    worst_joint = joints[worst_idx]
    max_ratio = ratio[worst_idx]

    flag = " !!!" if max_ratio > 0.8 else ("  ! " if max_ratio > 0.5 else "    ")
    ratio_str = "  ".join(f"{r:>5.1%}" for r in ratio)
    print(f"  {label:>4} ({x:.2f},{y:.2f},{z:.3f})  {ratio_str}  {worst_joint:>6}  {flag}")

print()
print(f"tau limits [Nm]: {list(zip(joints, TAU_LIMITS.round(1)))}")
print()
print("판단 기준 (토크 리미트 대비 비율):")
print("  < 50%: 여유 충분")
print("  50~80%: 모니터링, cost 추가 고려")
print("  > 80%: cost 추가 강력 권장")
