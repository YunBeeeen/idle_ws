"""IK seed strategy comparison — ROS 불필요.

기존 24 random seed vs 새 structured seed (G1~G6) 비교:
  - 수렴률 (feasible IK 비율)
  - 평균 manipulability
  - 평균 계획 시간

실행:
    source ~/idle_ws/install/setup.bash
    python3 src/phy/scripts/ik_seed_analysis.py
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_WS = Path(__file__).resolve().parents[3]
for _pkg in ("phy", "idle_common"):
    for _p in (_WS / "install" / _pkg / "lib").glob("python3*"):
        _dp = _p / "dist-packages"
        if _dp.exists() and str(_dp) not in sys.path:
            sys.path.insert(0, str(_dp))

URDF_PATH = str(_WS / "install/sim/share/sim/urdf/robot.urdf")
SRDF_PATH = str(_WS / "install/sim/share/sim/srdf/robot.srdf")
SIM_SHARE_PARENT = str(_WS / "install/sim/share")

from phy.ik import IKConfig, IKSolver
from phy.plan import top_down_R
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP

# Workspace: pick-and-place 워크스페이스 대표 타겟
X_RANGE = np.arange(0.20, 0.50, 0.10)
Y_RANGE = np.arange(-0.25, 0.30, 0.10)
Z_PICK = 0.025   # 테이블 + 블록 절반 높이 (대략)
IK_TOL = 0.005   # 5 mm
N_TRIALS = 3     # 타겟당 반복 횟수 (타이밍 안정성)


@dataclass
class StrategyResult:
    name: str
    n_targets: int = 0
    n_feasible: int = 0
    mean_manip: float = 0.0
    mean_time_ms: float = 0.0
    _manip_sum: float = field(default=0.0, repr=False)
    _time_sum: float = field(default=0.0, repr=False)

    def record(self, feasible: bool, manip: float, elapsed_s: float) -> None:
        self.n_targets += 1
        if feasible:
            self.n_feasible += 1
            self._manip_sum += manip
        self._time_sum += elapsed_s * 1000.0

    def finalise(self) -> None:
        self.mean_manip = self._manip_sum / max(self.n_feasible, 1)
        self.mean_time_ms = self._time_sum / max(self.n_targets, 1)

    @property
    def convergence_pct(self) -> float:
        return 100.0 * self.n_feasible / max(self.n_targets, 1)


# ---------------------------------------------------------------------------
# Old strategy: 4 structured + 24 random (from plan.py before this PR)
# ---------------------------------------------------------------------------
def _solve_old(
    ik: IKSolver,
    target_xyz: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
    n_random: int = 24,
) -> tuple[bool, float]:
    yaw = math.atan2(float(target_xyz[1]), float(target_xyz[0]))
    seeds = [
        np.zeros(6),
        ik.clip_to_limits(np.array([yaw, 1.2, 2.2, 0., 0., 0.])),
        ik.clip_to_limits(np.array([yaw + math.pi, 1.2, 2.2, 0., 0., 0.])),
    ]
    for _ in range(n_random):
        seeds.append(rng.uniform(ik.lower_limits, ik.upper_limits))

    best = None
    for s in seeds:
        res = ik.solve_pose(target_xyz, R, s)
        if best is None or res.residual_norm < best.residual_norm:
            best = res
        if res.success or res.residual_norm <= IK_TOL:
            return True, ik.manipulability(res.q)
    return False, 0.0


# ---------------------------------------------------------------------------
# New strategy: structured G1-G6
# ---------------------------------------------------------------------------
def _solve_new(
    ik: IKSolver,
    target_xyz: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
    n_random: int = 12,
    w_min_manip: float = 0.02,
) -> tuple[bool, float]:
    """Analytical seed strategy matching updated plan.py _solve_ik_multistart."""
    half_pi = math.pi / 2.0
    lo, hi = ik.lower_limits, ik.upper_limits

    seeds: list[np.ndarray] = [np.zeros(6)]

    # G2/G3: analytical seeds from URDF geometry
    analytic = ik.heuristic_seeds_from_target(target_xyz)
    seeds.extend(analytic)

    # G4: shoulder ±π/6 of each analytical seed
    for base in analytic:
        for delta in (math.pi / 6, -math.pi / 6):
            s = base.copy(); s[0] = base[0] + delta
            seeds.append(ik.clip_to_limits(s))

    # G5: backward of each analytical seed
    for base in analytic:
        s = base.copy(); s[0] = base[0] + math.pi
        seeds.append(ik.clip_to_limits(s))

    # G6: biased random — J2*J3>0, J4 from empirical formula
    for _ in range(n_random):
        j2 = float(rng.uniform(float(lo[1]), float(hi[1])))
        j5_val = -half_pi if j2 >= 0.0 else half_pi
        j3_lo = max(float(lo[2]), 0.1) if j2 >= 0.0 else float(lo[2])
        j3_hi = float(hi[2]) if j2 >= 0.0 else min(float(hi[2]), -0.1)
        j3 = float(rng.uniform(j3_lo, j3_hi)) if j3_lo < j3_hi else j3_lo
        j4_center = -0.623 * j3 - 1.275 * (1.0 if j2 >= 0.0 else -1.0)
        j4 = float(np.clip(rng.normal(j4_center, 0.2), float(lo[3]), float(hi[3])))
        seeds.append(ik.clip_to_limits(np.array([
            float(rng.uniform(float(lo[0]), float(hi[0]))),
            j2, j3, j4, j5_val,
            float(rng.uniform(float(lo[5]), float(hi[5]))),
        ])))

    feasible = []
    for s in seeds:
        res = ik.solve_pose(target_xyz, R, s)
        if not (res.success or res.residual_norm <= IK_TOL):
            continue
        m = ik.manipulability(res.q)
        if m < w_min_manip:
            continue
        feasible.append((res, m))

    if not feasible:
        return False, 0.0
    _, best_m = max(feasible, key=lambda x: x[1])
    return True, best_m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("로봇 모델 로딩...", flush=True)
    joints = tuple(DEFAULT_MOTOR_JOINT_MAP[m] for m in sorted(DEFAULT_MOTOR_JOINT_MAP))
    ik = IKSolver(URDF_PATH, IKConfig(target_frame="gripper", controlled_joints=joints))

    targets = [
        np.array([x, y, Z_PICK])
        for x in X_RANGE
        for y in Y_RANGE
    ]
    n = len(targets)
    print(f"타겟 수: {n}  (Z={Z_PICK:.3f}m 고정, top-down 자세)\n")

    old_res = StrategyResult("기존 (4구조 + 24 random)")
    new_res = StrategyResult("신규 (G1~G6 structured + 12 biased)")

    rng = np.random.default_rng(42)
    R = top_down_R(0.0)

    for i, xyz in enumerate(targets, 1):
        yaw = math.atan2(float(xyz[1]), float(xyz[0]))
        R_t = top_down_R(yaw)

        # Old
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            ok_old, m_old = _solve_old(ik, xyz, R_t, rng)
        old_res.record(ok_old, m_old, (time.perf_counter() - t0) / N_TRIALS)

        # New
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            ok_new, m_new = _solve_new(ik, xyz, R_t, rng)
        new_res.record(ok_new, m_new, (time.perf_counter() - t0) / N_TRIALS)

        if i % 10 == 0:
            print(f"  [{i}/{n}]", flush=True)

    old_res.finalise()
    new_res.finalise()

    print("\n" + "=" * 60)
    print(f"{'전략':<32} {'수렴률':>8} {'평균Manip':>10} {'평균시간':>10}")
    print("=" * 60)
    for r in (old_res, new_res):
        print(
            f"{r.name:<32} {r.convergence_pct:>7.1f}% "
            f"{r.mean_manip:>10.4f} {r.mean_time_ms:>8.1f}ms"
        )
    print("=" * 60)

    # Delta
    d_conv = new_res.convergence_pct - old_res.convergence_pct
    d_manip = new_res.mean_manip - old_res.mean_manip
    d_time = new_res.mean_time_ms - old_res.mean_time_ms
    print(f"\n개선: 수렴률 {d_conv:+.1f}%  manipulability {d_manip:+.4f}  시간 {d_time:+.1f}ms")


if __name__ == "__main__":
    main()
