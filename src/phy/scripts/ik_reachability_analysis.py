"""IK 도달 가능성 분석 — ROS 불필요, launch/노드에 포함하지 말 것.

xyz+yaw 그리드를 쓸어가며 각 타겟을 분류:
  accepted          : 기본 시드(영벡터)로 IK 성공 + 충돌 없음
  needs_random_seed : 기본 시드 실패 → 랜덤 시드로 성공 + 충돌 없음
  collision_blocked : IK는 풀리지만 모든 해가 충돌
  ik_unreachable    : 모든 시드 실패 (진짜 도달 불가)

실행:
    source ~/idle_ws/install/setup.bash
    python3 src/phy/scripts/ik_reachability_analysis.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── 패키지 경로 (빌드된 install 기준) ─────────────────────────────────────────
_WS = Path(__file__).resolve().parents[3]
for _pkg in ("phy", "idle_common"):
    for _p in (_WS / "install" / _pkg / "lib").glob("python3*"):
        _dp = _p / "dist-packages"
        if _dp.exists() and str(_dp) not in sys.path:
            sys.path.insert(0, str(_dp))

URDF_PATH = str(_WS / "install/sim/share/sim/urdf/robot.urdf")
SRDF_PATH = str(_WS / "install/sim/share/sim/srdf/robot.srdf")
SIM_SHARE_PARENT = str(_WS / "install/sim/share")

from phy.collision import CollisionChecker
from phy.ik import IKConfig, IKSolver
from phy.plan import top_down_R
from phy.robot_model import RobotModel
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP

# ── 분석 파라미터 ──────────────────────────────────────────────────────────────
X_RANGE    = np.arange(0.10, 0.55, 0.10)
Y_RANGE    = np.arange(-0.35, 0.40, 0.10)
Z_RANGE    = np.arange(0.10, 0.65, 0.10)
YAW_DEGS   = [0.0, 45.0, 90.0]
N_RANDOM   = 40       # default 실패 시 추가 랜덤 시드 수
IK_TOL     = 1.0e-3   # residual 허용 한계 (m)
SEED_ZERO  = np.zeros(6)

# ── 결과 ──────────────────────────────────────────────────────────────────────
@dataclass
class Result:
    xyz: tuple[float, float, float]
    yaw_deg: float
    status: str   # accepted | needs_random_seed | collision_blocked | ik_unreachable
    q: np.ndarray | None = None
    residual: float = float("inf")
    seeds_tried: int = 0
    n_col_configs: int = 0

@dataclass
class Summary:
    accepted:          list[Result] = field(default_factory=list)
    needs_random_seed: list[Result] = field(default_factory=list)
    collision_blocked: list[Result] = field(default_factory=list)
    ik_unreachable:    list[Result] = field(default_factory=list)

    def all(self) -> list[Result]:
        return self.accepted + self.needs_random_seed + self.collision_blocked + self.ik_unreachable

    @property
    def total(self) -> int:
        return len(self.all())


# ── 분석 로직 ──────────────────────────────────────────────────────────────────
def _solve(ik: IKSolver, xyz: np.ndarray, R: np.ndarray, seed: np.ndarray):
    res = ik.solve_pose(xyz, R, seed)
    ok = res.success or res.residual_norm <= IK_TOL
    return ok, res.q, float(res.residual_norm)


def analyze(
    xyz: np.ndarray,
    yaw_deg: float,
    ik: IKSolver,
    col: CollisionChecker,
    motor_ids: list[int],
    rng: np.random.Generator,
) -> Result:
    R = top_down_R(math.radians(yaw_deg))

    def q_dict(q): return {m: float(q[i]) for i, m in enumerate(motor_ids)}

    # 1) 기본 시드
    ok, q, res = _solve(ik, xyz, R, SEED_ZERO)
    seeds = 1
    best_q, best_res = q.copy(), res
    n_col = 0

    if ok:
        if not col.check(q_dict(q)):
            return Result(tuple(xyz.tolist()), yaw_deg, "accepted", q.copy(), res, seeds)
        n_col += 1

    # 2) 랜덤 시드
    for _ in range(N_RANDOM):
        s = rng.uniform(ik.lower_limits, ik.upper_limits)
        ok_r, q_r, res_r = _solve(ik, xyz, R, s)
        seeds += 1
        if ok_r:
            if not col.check(q_dict(q_r)):
                return Result(tuple(xyz.tolist()), yaw_deg, "needs_random_seed", q_r.copy(), res_r, seeds)
            n_col += 1
        elif res_r < best_res:
            best_q, best_res = q_r.copy(), res_r

    if n_col > 0:
        return Result(tuple(xyz.tolist()), yaw_deg, "collision_blocked",
                      seeds_tried=seeds, n_col_configs=n_col)
    return Result(tuple(xyz.tolist()), yaw_deg, "ik_unreachable",
                  residual=best_res, seeds_tried=seeds)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("로봇 모델 로딩...", flush=True)
    robot = RobotModel(URDF_PATH, dict(DEFAULT_MOTOR_JOINT_MAP))
    motor_ids = list(robot.ordered_motor_ids)
    joints = tuple(robot.bindings[m].joint_name for m in motor_ids)
    ik = IKSolver(URDF_PATH, IKConfig(target_frame="gripper", controlled_joints=joints))
    col = CollisionChecker(robot, srdf_path=SRDF_PATH, package_dirs=[SIM_SHARE_PARENT])

    print("\n=== Joint limits ===")
    for i, name in enumerate(joints):
        lo, hi = math.degrees(ik.lower_limits[i]), math.degrees(ik.upper_limits[i])
        print(f"  {name}: [{lo:.1f}°, {hi:.1f}°]")

    total = len(X_RANGE) * len(Y_RANGE) * len(Z_RANGE) * len(YAW_DEGS)
    print(f"\n{total}개 타겟 분석 시작\n", flush=True)

    summary = Summary()
    rng = np.random.default_rng(42)
    t0 = time.time()
    done = 0

    for x in X_RANGE:
        for y in Y_RANGE:
            for z in Z_RANGE:
                for yaw in YAW_DEGS:
                    r = analyze(np.array([x, y, z]), yaw, ik, col, motor_ids, rng)
                    getattr(summary, r.status).append(r)
                    done += 1
                    if done % 100 == 0:
                        print(f"  [{done}/{total}] {time.time()-t0:.0f}s | "
                              f"OK={len(summary.accepted)} "
                              f"seed={len(summary.needs_random_seed)} "
                              f"col={len(summary.collision_blocked)} "
                              f"unreachable={len(summary.ik_unreachable)}", flush=True)

    print(f"\n완료 ({time.time()-t0:.1f}s)\n")

    # ── 전체 요약 ────────────────────────────────────────────────────────────
    n = summary.total
    print("=" * 60)
    print("=== 결과 요약 ===")
    print("=" * 60)
    for attr, label in [
        ("accepted",          "기본 시드 성공    "),
        ("needs_random_seed", "랜덤 시드 필요    "),
        ("collision_blocked", "IK OK, 충돌 막힘  "),
        ("ik_unreachable",    "진짜 불가         "),
    ]:
        lst = getattr(summary, attr)
        print(f"  {label}: {len(lst):4d} / {n}  ({100*len(lst)/n:.1f}%)")

    # ── 성공 케이스 joint 분포 ───────────────────────────────────────────────
    good = summary.accepted + summary.needs_random_seed
    if good:
        q_arr = np.array([r.q for r in good])
        print("\n=== 성공 케이스 joint 분포 (degrees) ===")
        for i, name in enumerate(joints):
            v = np.degrees(q_arr[:, i])
            print(f"  {name}: mean={v.mean():+7.1f}°  std={v.std():5.1f}°"
                  f"  [{v.min():+7.1f}°, {v.max():+7.1f}°]")

    # ── '랜덤 시드 필요' 케이스 — 기본 시드 대비 왜 다른지 ──────────────────
    if summary.needs_random_seed:
        print(f"\n=== 랜덤 시드 필요 케이스 샘플 (최대 15개) ===")
        print(f"  {'xyz':25s}  {'yaw':>6s}  {'seeds':>5s}  {'j5':>7s}  {'j6':>7s}")
        for r in summary.needs_random_seed[:15]:
            q_deg = np.degrees(r.q)
            print(f"  ({r.xyz[0]:.2f},{r.xyz[1]:+.2f},{r.xyz[2]:.2f})  "
                  f"{r.yaw_deg:>5.0f}°  {r.seeds_tried:>5d}  "
                  f"{q_deg[4]:>+7.1f}°  {q_deg[5]:>+7.1f}°")

    # ── 충돌 막힌 케이스 ─────────────────────────────────────────────────────
    if summary.collision_blocked:
        print(f"\n=== 충돌 막힌 케이스 샘플 (최대 10개) ===")
        for r in summary.collision_blocked[:10]:
            print(f"  ({r.xyz[0]:.2f},{r.xyz[1]:+.2f},{r.xyz[2]:.2f})  "
                  f"yaw={r.yaw_deg:.0f}°  충돌 configs={r.n_col_configs}")

    # ── 불가 영역 z층별 요약 ─────────────────────────────────────────────────
    if summary.ik_unreachable:
        print(f"\n=== IK 불가 영역 (z별) ===")
        by_z: dict = defaultdict(list)
        for r in summary.ik_unreachable:
            by_z[round(r.xyz[2], 2)].append(r)
        for z_val in sorted(by_z):
            grp = by_z[z_val]
            xs = sorted(set(round(r.xyz[0], 2) for r in grp))
            ys = sorted(set(round(r.xyz[1], 2) for r in grp))
            print(f"  z={z_val:.2f}  {len(grp):3d}개  "
                  f"x={[f'{v:.2f}' for v in xs]}  "
                  f"y=[{min(ys):.2f}~{max(ys):.2f}]")

    # ── yaw별 성공률 ─────────────────────────────────────────────────────────
    print("\n=== yaw별 도달 가능률 ===")
    for yaw in YAW_DEGS:
        grp = [r for r in summary.all() if r.yaw_deg == yaw]
        ok  = [r for r in grp if r.status in ("accepted", "needs_random_seed")]
        print(f"  yaw={yaw:5.1f}°: {len(ok)}/{len(grp)} ({100*len(ok)/len(grp):.1f}%)")


if __name__ == "__main__":
    main()
