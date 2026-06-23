"""랜덤 XY 위치로 pre-grasp 이동 테스트.

먼저 시뮬을 실행하세요:
    ros2 launch idle_launch sim_pickplace.launch.py

그 다음 이 스크립트를 실행하세요:
    source ~/idle_ws/install/setup.bash
    python3 src/phy/scripts/sim_random_pregrasp.py [--n N] [--seed S]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from msgs.msg import EETarget
from std_msgs.msg import String

# Pre-grasp Z: 테이블 높이 + 블록 높이 + 접근 여유
Z_TABLE  = 0.00
Z_BLOCK  = 0.050
Z_APPROACH = Z_TABLE + Z_BLOCK + 0.080   # ≈ 0.13 m

# 워크스페이스 범위
X_MIN, X_MAX = 0.22, 0.48
Y_MIN, Y_MAX = -0.25, 0.25

TIMEOUT_S = 15.0   # 한 이동당 최대 대기 시간


def yaw_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class RandomPregrasper(Node):
    def __init__(self, n_targets: int, seed: int):
        super().__init__("sim_random_pregrasp")
        self._n = n_targets
        self._rng = np.random.default_rng(seed)
        self._status = "IDLE"
        self._done_event = False

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(EETarget, "/ee_target", 10)
        self._status_sub = self.create_subscription(
            String, "/plan/status", self._on_status, 10
        )

    def _on_status(self, msg: String) -> None:
        self._status = msg.data
        if msg.data in ("DONE", "FAIL"):
            self._done_event = True

    def _send(self, x: float, y: float, z: float) -> None:
        yaw = math.atan2(y, x)
        qx, qy, qz, qw = yaw_quat(yaw)
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "world"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        msg = EETarget()
        msg.pose = pose
        msg.duration_override_s = 0.0
        msg.use_safe_transit = False
        self._pub.publish(msg)

    def run(self) -> None:
        # 상태 수신 대기
        time.sleep(0.5)
        rclpy.spin_once(self, timeout_sec=0.5)

        results = []
        print(f"\n{'#':>3}  {'target':>22}  {'결과':>6}  {'시간(s)':>8}")
        print("-" * 50)

        for i in range(self._n):
            x = float(self._rng.uniform(X_MIN, X_MAX))
            y = float(self._rng.uniform(Y_MIN, Y_MAX))
            z = Z_APPROACH

            self._done_event = False
            self._status = "IDLE"
            t0 = time.time()
            self._send(x, y, z)

            # DONE or FAIL 대기
            while not self._done_event:
                rclpy.spin_once(self, timeout_sec=0.05)
                if time.time() - t0 > TIMEOUT_S:
                    self._status = "TIMEOUT"
                    break

            elapsed = time.time() - t0
            ok = self._status == "DONE"
            results.append(ok)
            mark = "✓" if ok else "✗"
            print(f"{i+1:>3}  ({x:+.3f}, {y:+.3f}, {z:.3f})  {mark:>6}  {elapsed:>7.2f}s  [{self._status}]")

            # 다음 이동 전 잠시 대기 (hold 상태에서 시작)
            time.sleep(0.5)

        n_ok = sum(results)
        print(f"\n{'='*50}")
        print(f"결과: {n_ok}/{self._n} 성공 ({100*n_ok/self._n:.0f}%)")
        if self._n - n_ok > 0:
            print(f"실패: {self._n - n_ok}건 (IK 실패 또는 timeout)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="이동 횟수 (기본: 10)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    rclpy.init()
    node = RandomPregrasper(args.n, args.seed)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
