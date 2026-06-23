"""오프라인 PRE_GRASP + GRASP_DESCEND 1000-타겟 실현 가능성 테스트.

ROS 없이 Planner를 직접 호출 — IK + trajectory self-collision + RNEA 토크 + manipulability 체크.

실행:
    source ~/idle_ws/install/setup.bash
    python3 src/phy/scripts/offline_pregrasp_test.py [--n N] [--seed S] [--v-max V] [--a-max A]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ── 경로 설정 (ROS 없이) ──────────────────────────────────────────────────────
_WS = Path(__file__).resolve().parents[3]
for _pkg in ("phy", "idle_common"):
    for _p in (_WS / "install" / _pkg / "lib").glob("python3*"):
        _dp = _p / "dist-packages"
        if _dp.exists() and str(_dp) not in sys.path:
            sys.path.insert(0, str(_dp))

URDF_PATH = str(_WS / "install/sim/share/sim/urdf/robot.urdf")
SRDF_PATH = str(_WS / "install/sim/share/sim/srdf/robot.srdf")
SIM_SHARE_PARENT = str((_WS / "install/sim/share/sim").parent)

import pinocchio as pin
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP, DEFAULT_TAU_LIMIT_BY_MOTOR
from phy.collision import CollisionChecker
from phy.ik import IKConfig, IKSolver
from phy.plan import Plan, Planner, PlannerConfig
from phy.robot_model import RobotModel
from phy.traj import sample_quintic

# ── 워크스페이스 / Z 상수 ─────────────────────────────────────────────────────
X_MIN, X_MAX = 0.22, 0.48
Y_MIN, Y_MAX = -0.25, 0.25

Z_TABLE    = 0.00
Z_BLOCK    = 0.050
Z_APPROACH = Z_TABLE + Z_BLOCK + 0.080   # 0.130 m  (PRE_GRASP)
Z_GRASP    = Z_TABLE + Z_BLOCK * 0.5     # 0.025 m  (GRASP_DESCEND)
GRASP_MIN_DURATION = 3.0

# ── 결과 분류 ─────────────────────────────────────────────────────────────────
BOTH_OK           = "BOTH_OK"
IK_FAIL_APPROACH  = "IK_FAIL_APPROACH"
COLL_APPROACH     = "COLL_APPROACH"
IK_FAIL_DESCENT   = "IK_FAIL_DESCENT"
COLL_DESCENT      = "COLL_DESCENT"

TAU_WARN_THRESH  = 0.80   # tau_limit 대비 80% 초과 시 경고
MANIP_WARN_THRESH = 0.05  # manipulability 이 값 미만 시 경고
SAFETY_SAMPLES   = 20     # 안전 체크용 trajectory 샘플 수


# ── 안전 체크 헬퍼 ────────────────────────────────────────────────────────────

def _qdd_at(plan: Plan, t: float) -> np.ndarray:
    """quintic 계수에서 해석적으로 qdd 계산."""
    c = plan.trajectory.coeffs  # [dof, 6]
    t2 = t * t
    t3 = t2 * t
    return (2.0 * c[:, 2]
            + 6.0  * c[:, 3] * t
            + 12.0 * c[:, 4] * t2
            + 20.0 * c[:, 5] * t3)


def _rnea_ratios(
    robot: RobotModel,
    q_vec: np.ndarray,
    qd_vec: np.ndarray,
    qdd_vec: np.ndarray,
    tau_limits: np.ndarray,
) -> np.ndarray:
    """RNEA 토크를 계산하고 각 관절의 tau_limit 대비 비율을 반환."""
    robot._fill_q_buf({m: float(q_vec[i])
                       for i, m in enumerate(robot.ordered_motor_ids)})
    dof = robot.model.nv
    qd_m = np.zeros(dof)
    qdd_m = np.zeros(dof)
    for i, m in enumerate(robot.ordered_motor_ids):
        vi = robot.bindings[m].v_index
        qd_m[vi]  = float(qd_vec[i])
        qdd_m[vi] = float(qdd_vec[i])
    tau = pin.rnea(robot.model, robot.data, robot._q_buf, qd_m, qdd_m)
    return np.array([abs(float(tau[robot.bindings[m].v_index])) / tau_limits[i]
                     for i, m in enumerate(robot.ordered_motor_ids)])


def check_plan_safety(
    plan: Plan,
    robot: RobotModel,
    ik: IKSolver,
    tau_limits: np.ndarray,
    n_samples: int = SAFETY_SAMPLES,
) -> tuple[float, int, float]:
    """
    trajectory 전체에서 RNEA 토크 비율과 도착 q의 manipulability 반환.

    - RNEA 토크: 경로 전체 샘플 (피크 토크가 어디서 나오는지 중요)
    - manipulability: 도착 q만 (경로 시작 q_zero가 특이점이어서 경로 최솟값은 항상 0)

    returns: (max_tau_ratio, worst_motor_idx, end_manipulability)
    """
    traj = plan.trajectory
    max_ratio = 0.0
    worst_motor = 0

    denom = max(n_samples - 1, 1)
    for i in range(n_samples):
        t = (i / denom) * traj.duration
        q_vec, qd_vec, _ = sample_quintic(traj, t)
        qdd_vec = _qdd_at(plan, t)

        ratios = _rnea_ratios(robot, q_vec, qd_vec, qdd_vec, tau_limits)
        idx = int(np.argmax(ratios))
        if ratios[idx] > max_ratio:
            max_ratio = float(ratios[idx])
            worst_motor = idx

    # manipulability는 도착 q(잡는 자세)에서만 의미 있음
    end_manip = ik.manipulability(plan.end_q)
    return max_ratio, worst_motor, end_manip


# ── Planner 초기화 ────────────────────────────────────────────────────────────

class TestContext:
    """플래너 + 안전 체크에 필요한 객체 묶음."""
    def __init__(self, v_max: float, a_max: float) -> None:
        motor_joint_map = dict(DEFAULT_MOTOR_JOINT_MAP)
        self.robot    = RobotModel(URDF_PATH, motor_joint_map)
        collision     = CollisionChecker(self.robot, srdf_path=SRDF_PATH,
                                         package_dirs=[SIM_SHARE_PARENT])
        controlled    = tuple(self.robot.bindings[m].joint_name
                               for m in self.robot.ordered_motor_ids)
        self.ik       = IKSolver(URDF_PATH, IKConfig(target_frame="gripper",
                                                      controlled_joints=controlled))
        self.planner  = Planner(self.robot, collision, self.ik,
                                PlannerConfig(v_max=v_max, a_max=a_max))
        self.tau_lims = np.array([DEFAULT_TAU_LIMIT_BY_MOTOR[m]
                                   for m in self.robot.ordered_motor_ids], dtype=float)
        self.q_zero   = np.zeros(len(motor_joint_map))
        self.joints   = [DEFAULT_MOTOR_JOINT_MAP[m] for m in self.robot.ordered_motor_ids]


# ── 단일 타겟 테스트 ──────────────────────────────────────────────────────────

def test_target(ctx: TestContext, x: float, y: float):
    """
    returns: (result, t_app_ms, t_des_ms, max_tau, worst_jname, min_manip)
    안전 수치는 BOTH_OK인 경우에만 채워짐.
    """
    yaw = math.atan2(y, x)
    q0  = ctx.q_zero

    # LEG 1: approach
    t0 = time.perf_counter()
    p_app = ctx.planner.plan_to_pose(np.array([x, y, Z_APPROACH]), yaw, start_q=q0)
    t_app = (time.perf_counter() - t0) * 1000

    if p_app is None:
        return IK_FAIL_APPROACH, t_app, 0.0, 0.0, "", 0.0
    if not p_app.collision_safe:
        return COLL_APPROACH, t_app, 0.0, 0.0, "", 0.0

    # LEG 2: descent
    t1 = time.perf_counter()
    p_des = ctx.planner.plan_to_pose(
        np.array([x, y, Z_GRASP]), yaw,
        start_q=p_app.end_q,
        min_duration=GRASP_MIN_DURATION,
    )
    t_des = (time.perf_counter() - t1) * 1000

    if p_des is None:
        return IK_FAIL_DESCENT, t_app, t_des, 0.0, "", 0.0
    if not p_des.collision_safe:
        return COLL_DESCENT, t_app, t_des, 0.0, "", 0.0

    # 안전 체크 (approach + descent 각각, 더 나쁜 값 선택)
    # manipulability: approach end_q와 descent end_q 중 작은 값
    tau_a, wi_a, man_a = check_plan_safety(p_app, ctx.robot, ctx.ik, ctx.tau_lims)
    tau_d, wi_d, man_d = check_plan_safety(p_des, ctx.robot, ctx.ik, ctx.tau_lims)
    max_tau   = max(tau_a, tau_d)
    worst_idx = wi_a if tau_a >= tau_d else wi_d
    min_manip = min(man_a, man_d)  # descent end_q(파지 자세)가 보통 더 낮음
    worst_j   = ctx.joints[worst_idx]

    return BOTH_OK, t_app, t_des, max_tau, worst_j, min_manip


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,   default=1000)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--v-max", type=float, default=0.5)
    parser.add_argument("--a-max", type=float, default=1.0)
    args = parser.parse_args()

    print(f"초기화 중… (v_max={args.v_max}, a_max={args.a_max})")
    ctx = TestContext(args.v_max, args.a_max)
    print("초기화 완료.\n")

    rng = np.random.default_rng(args.seed)
    counts: dict[str, int] = {k: 0 for k in
        [BOTH_OK, IK_FAIL_APPROACH, COLL_APPROACH, IK_FAIL_DESCENT, COLL_DESCENT]}
    times_app: list[float] = []
    times_des: list[float] = []
    tau_ratios: list[float] = []
    manips: list[float] = []
    tau_warns: list[tuple] = []
    manip_warns: list[tuple] = []
    failures: list[tuple] = []

    hdr = (f"{'#':>5}  {'x':>6} {'y':>6}  {'결과':<18}"
           f"  {'t_app':>7}  {'t_des':>7}  {'τ_max%':>7}  {'wj':>4}  {'μ_min':>7}")
    print(hdr)
    print("-" * len(hdr))

    t_wall = time.perf_counter()
    for i in range(args.n):
        x = float(rng.uniform(X_MIN, X_MAX))
        y = float(rng.uniform(Y_MIN, Y_MAX))

        res, ta, td, tau_r, wj, manip = test_target(ctx, x, y)
        counts[res] += 1
        times_app.append(ta)
        if td > 0:
            times_des.append(td)

        mark = "✓"
        extra = ""
        if res != BOTH_OK:
            mark = "✗"
            failures.append((x, y, res))
        else:
            tau_ratios.append(tau_r)
            manips.append(manip)
            if tau_r > TAU_WARN_THRESH:
                mark = "!"
                extra = " TAU"
                tau_warns.append((x, y, tau_r, wj))
            if manip < MANIP_WARN_THRESH:
                mark = "!"
                extra += " MANIP"
                manip_warns.append((x, y, manip))

        tau_s   = f"{tau_r*100:>6.1f}%" if res == BOTH_OK else "      -"
        manip_s = f"{manip:>7.4f}"       if res == BOTH_OK else "       -"
        print(f"{i+1:>5}  {x:>+6.3f} {y:>+6.3f}  {mark} {res:<16}"
              f"  {ta:>6.0f}ms  {td:>6.0f}ms  {tau_s}  {wj:>4}  {manip_s}{extra}")

    elapsed = time.perf_counter() - t_wall
    n = args.n
    n_ok = counts[BOTH_OK]

    print()
    print("=" * 72)
    print(f"결과 요약  (n={n}, seed={args.seed}, v_max={args.v_max}, a_max={args.a_max})")
    print("=" * 72)
    print(f"  전체 성공             : {n_ok:>4}/{n}  ({100*n_ok/n:5.1f}%)")
    print(f"  IK 실패 — approach    : {counts[IK_FAIL_APPROACH]:>4}/{n}  ({100*counts[IK_FAIL_APPROACH]/n:5.1f}%)")
    print(f"  충돌    — approach    : {counts[COLL_APPROACH]:>4}/{n}  ({100*counts[COLL_APPROACH]/n:5.1f}%)")
    print(f"  IK 실패 — descent     : {counts[IK_FAIL_DESCENT]:>4}/{n}  ({100*counts[IK_FAIL_DESCENT]/n:5.1f}%)")
    print(f"  충돌    — descent     : {counts[COLL_DESCENT]:>4}/{n}  ({100*counts[COLL_DESCENT]/n:5.1f}%)")
    print(f"  토크 경고 (>{TAU_WARN_THRESH*100:.0f}%)  : {len(tau_warns):>4}/{n_ok}")
    print(f"  조작성 경고 (<{MANIP_WARN_THRESH:.2f}) : {len(manip_warns):>4}/{n_ok}")

    print()
    if times_app:
        arr = np.array(times_app)
        print(f"  계획시간 approach  mean={arr.mean():.0f}ms  p95={np.percentile(arr,95):.0f}ms  max={arr.max():.0f}ms")
    if times_des:
        arr = np.array(times_des)
        print(f"  계획시간 descent   mean={arr.mean():.0f}ms  p95={np.percentile(arr,95):.0f}ms  max={arr.max():.0f}ms")
    if tau_ratios:
        arr = np.array(tau_ratios)
        print(f"  RNEA τ/τ_lim      mean={arr.mean()*100:.1f}%  p95={np.percentile(arr,95)*100:.1f}%  max={arr.max()*100:.1f}%")
    if manips:
        arr = np.array(manips)
        print(f"  manipulability    mean={arr.mean():.4f}  p5={np.percentile(arr,5):.4f}  min={arr.min():.4f}")
    print(f"\n  총 소요: {elapsed:.1f}s  ({elapsed/n*1000:.1f}ms/타겟)")

    if tau_warns:
        print(f"\n토크 경고 ({len(tau_warns)}건, τ>{TAU_WARN_THRESH*100:.0f}%):")
        for fx, fy, tr, wj in tau_warns[:20]:
            print(f"  ({fx:+.3f},{fy:+.3f})  τ={tr*100:.1f}%  worst={wj}")
    if manip_warns:
        print(f"\n조작성 경고 ({len(manip_warns)}건, μ<{MANIP_WARN_THRESH:.2f}):")
        for fx, fy, m in manip_warns[:20]:
            print(f"  ({fx:+.3f},{fy:+.3f})  μ={m:.4f}")
    if failures:
        print(f"\n실패 목록 ({len(failures)}건):")
        for fx, fy, fr in failures[:30]:
            print(f"  ({fx:+.3f},{fy:+.3f})  {fr}")
        if len(failures) > 30:
            print(f"  … 외 {len(failures)-30}건")


if __name__ == "__main__":
    main()
