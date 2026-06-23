#!/usr/bin/env python3
"""gravity_scale 진단: tau_measured vs tau_g_model 비율로 보정값 추천.

/motor_state_array를 구독해 피노키오 중력 토크와 실측 토크를 비교한다.
|tau_g_model| 이 너무 작은 샘플(중력 기여가 작은 자세)은 제외한다.

Usage:
    # ROS2 환경 소싱 후:
    python3 src/phy/scripts/gravity_scale_diag.py

    # 특정 모터만 보려면:
    python3 src/phy/scripts/gravity_scale_diag.py --motors 2 3

권장 자세:
    j2 교정 → j2 를 ±60° 이상으로 고정, j3=0
    j3 교정 → j3 를 ±60° 이상으로 고정, j2 는 적당히
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC / "phy"))
sys.path.insert(0, str(_SRC / "idle_common"))

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP
from msgs.msg import MotorStateArray
from phy.gravity import GravityCompensator

URDF_PATH = _SRC / "sim" / "urdf" / "robot.urdf"
TAU_G_MIN = 0.3      # 이 이하인 샘플은 비율 계산에서 제외 [Nm]
WINDOW = 200         # 이동 평균 윈도우 (샘플 수)
PRINT_EVERY = 50     # N 샘플마다 출력


class GravityScaleDiag(Node):
    def __init__(self, target_motors: list[int]) -> None:
        super().__init__("gravity_scale_diag")
        self.target_motors = set(target_motors)

        self.gc = GravityCompensator(URDF_PATH, DEFAULT_MOTOR_JOINT_MAP)
        self.motor_ids = self.gc.ordered_motor_ids

        # motor_id → deque of (tau_measured, tau_g_model)
        self.samples: dict[int, collections.deque[tuple[float, float]]] = {
            mid: collections.deque(maxlen=WINDOW) for mid in self.motor_ids
        }
        self.tick = 0
        self.latest_q: dict[int, float] = {mid: 0.0 for mid in self.motor_ids}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(MotorStateArray, "/motor_state_array", self._on_state, qos)
        self.get_logger().info(
            f"gravity_scale_diag 시작 — 대상 모터: {sorted(self.target_motors)}\n"
            f"  TAU_G_MIN={TAU_G_MIN} Nm  window={WINDOW}  print_every={PRINT_EVERY}"
        )

    def _on_state(self, msg: MotorStateArray) -> None:
        for s in msg.states:
            mid = int(s.motor_id)
            if mid in self.latest_q:
                self.latest_q[mid] = float(s.q)

        tau_g = self.gc.compute_gravity_by_motor(self.latest_q)

        for s in msg.states:
            mid = int(s.motor_id)
            if mid not in self.target_motors:
                continue
            tg = tau_g.get(mid, 0.0)
            tm = float(s.tau)
            if abs(tg) >= TAU_G_MIN:
                self.samples[mid].append((tm, tg))

        self.tick += 1
        if self.tick % PRINT_EVERY == 0:
            self._print_summary()

    def _print_summary(self) -> None:
        lines = [f"\n{'─'*56}"]
        lines.append(f"{'모터':>5}  {'샘플':>5}  {'tau_g평균':>10}  {'tm평균':>10}  {'추천 scale':>11}  {'편향 bias':>10}")
        lines.append(f"{'─'*56}")

        for mid in sorted(self.target_motors):
            buf = self.samples[mid]
            if len(buf) < 10:
                lines.append(f"  {mid:>3}   샘플 부족 ({len(buf)} / {WINDOW}) — 자세를 바꿔보세요")
                continue

            tms = [x[0] for x in buf]
            tgs = [x[1] for x in buf]
            tm_mean = sum(tms) / len(tms)
            tg_mean = sum(tgs) / len(tgs)

            # scale: tm = scale * tg  →  scale = mean(tm/tg)
            ratios = [x[0] / x[1] for x in buf]
            scale = sum(ratios) / len(ratios)

            # bias: tm = tg + bias  →  bias = mean(tm - tg)
            bias = tm_mean - tg_mean

            lines.append(
                f"  {mid:>3}   {len(buf):>5}  {tg_mean:>+10.3f}  {tm_mean:>+10.3f}"
                f"  {scale:>+11.4f}  {bias:>+10.3f}"
            )

        lines.append(f"{'─'*56}")
        lines.append("※ 추천값을 control_params.yaml의 gravity_scale / gravity_bias 에 적용하세요.")
        print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motors", type=int, nargs="+", default=[2, 3],
                    help="진단할 모터 ID (기본: 2 3)")
    args = ap.parse_args()

    rclpy.init()
    node = GravityScaleDiag(args.motors)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
