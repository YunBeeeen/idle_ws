"""Record real robot ROS topics into the wide CSV format used by infer_real_log.py.

이 스크립트는 실제 로봇 preliminary check용 로그 수집 도구다.
/motor_state_array와 /motor_cmd_array를 motor_id 기준으로 정렬해
infer_real_log.py가 기대하는 wide CSV 형태로 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import rclpy
from msgs.msg import MotorCMDArray, MotorStateArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class RealLogRecorder(Node):
    """Subscribe to motor state/command topics and write synchronized wide rows."""

    def __init__(
        self,
        csv_path: Path,
        dof: int,
        sample_hz: float,
        duration_s: float,
        residual_offset_duration_s: float,
        contact_intervals: list[tuple[float, float]],
        progress_period_s: float,
        cmd_topic: str,
        state_topic: str,
    ) -> None:
        super().__init__("contact_real_log_recorder")
        self.csv_path = csv_path.expanduser().resolve()
        self.dof = int(dof)
        self.sample_hz = float(sample_hz)
        self.duration_s = float(duration_s)
        self.residual_offset_duration_s = max(0.0, float(residual_offset_duration_s))
        self.contact_intervals = [(float(start), float(end)) for start, end in contact_intervals if float(end) > float(start)]
        self.progress_period_s = max(0.0, float(progress_period_s))
        self.cmd_topic = str(cmd_topic)
        self.state_topic = str(state_topic)
        self.node_start_monotonic_s = time.monotonic()
        self.record_start_monotonic_s: float | None = None
        self.first_row_ros_time: float | None = None
        self.last_warn_s = float("-inf")
        self.last_progress_s = float("-inf")
        self.rows_written = 0
        self.stop_requested = False
        self.residual_offset_sum = [0.0 for _ in range(self.dof)]
        self.residual_offset_count = 0
        self.residual_offset = [0.0 for _ in range(self.dof)]

        # callback에서 받은 최신 state/cmd를 motor_id별 dict에 저장한다.
        # CSV row를 쓸 때 message 순서가 아니라 motor_id=1..6 순서로 재정렬한다.
        self.state_by_motor: dict[int, dict[str, float]] = {}
        self.cmd_by_motor: dict[int, dict[str, float]] = {}

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(self._header())
        self.csv_file.flush()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.state_sub = self.create_subscription(MotorStateArray, self.state_topic, self.on_state_array, qos)
        self.cmd_sub = self.create_subscription(MotorCMDArray, self.cmd_topic, self.on_cmd_array, qos)
        period_s = max(1.0 / self.sample_hz, 1.0e-4)
        self.timer = self.create_timer(period_s, self.on_timer)
        self.get_logger().info(
            f"recording real log to {self.csv_path} dof={self.dof} sample_hz={self.sample_hz:.1f} "
            f"duration_s={self.duration_s if self.duration_s > 0.0 else 'until Ctrl-C'} "
            f"state_topic={self.state_topic} cmd_topic={self.cmd_topic}"
        )
        if self.contact_intervals:
            self.get_logger().info(f"manual contact intervals enabled: {self.contact_intervals}")

    def _header(self) -> list[str]:
        # infer_real_log.py가 읽는 column convention.
        # tau_meas는 기록만 하고 모델 feature에는 쓰지 않는다.
        columns = ["time"]
        columns += [f"q{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"qdot{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"q_des{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"tau_cmd{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"tau_meas{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"tau_ff{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"kp{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"kd{idx}" for idx in range(1, self.dof + 1)]
        # 새 컬럼은 뒤에만 붙여 기존 CSV reader와 호환성을 유지한다.
        columns += [f"qdot_des{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"tau_residual{idx}" for idx in range(1, self.dof + 1)]
        columns += [f"tau_residual_corrected{idx}" for idx in range(1, self.dof + 1)]
        columns += ["contact_label"]
        return columns

    def on_state_array(self, msg: MotorStateArray) -> None:
        # MotorStateArray callback: q, qdot, measured tau를 최신값으로 저장한다.
        stamp_s = float(msg.stamp.sec) + float(msg.stamp.nanosec) * 1.0e-9
        for state in msg.states:
            motor_id = int(state.motor_id)
            if 1 <= motor_id <= self.dof:
                self.state_by_motor[motor_id] = {
                    "stamp": stamp_s,
                    "q": float(state.q),
                    "qdot": float(state.qd),
                    "tau_meas": float(state.tau),
                }

    def on_cmd_array(self, msg: MotorCMDArray) -> None:
        # MotorCMDArray callback: q_des, tau_ff, kp, kd를 저장하고 tau_cmd를 row 작성 시 재구성한다.
        stamp_s = float(msg.stamp.sec) + float(msg.stamp.nanosec) * 1.0e-9
        for cmd in msg.commands:
            motor_id = int(cmd.motor_id)
            if 1 <= motor_id <= self.dof:
                self.cmd_by_motor[motor_id] = {
                    "stamp": stamp_s,
                    "q_des": float(cmd.q_des),
                    "qd_des": float(cmd.qd_des),
                    "kp": float(cmd.kp),
                    "kd": float(cmd.kd),
                    "tau_ff": float(cmd.tau_ff),
                }

    def _ready(self) -> bool:
        required = range(1, self.dof + 1)
        return all(motor_id in self.state_by_motor for motor_id in required) and all(
            motor_id in self.cmd_by_motor for motor_id in required
        )

    def _warn_if_not_ready(self) -> None:
        now_s = time.monotonic()
        if (now_s - self.last_warn_s) < 1.0:
            return
        missing_state = [motor_id for motor_id in range(1, self.dof + 1) if motor_id not in self.state_by_motor]
        missing_cmd = [motor_id for motor_id in range(1, self.dof + 1) if motor_id not in self.cmd_by_motor]
        self.get_logger().warn(f"waiting for complete topics: missing_state={missing_state} missing_cmd={missing_cmd}")
        self.last_warn_s = now_s

    def _request_stop(self, reason: str) -> None:
        if self.stop_requested:
            return
        self.stop_requested = True
        self.get_logger().info(f"{reason}; wrote {self.rows_written} rows to {self.csv_path}")
        try:
            self.csv_file.flush()
        except Exception:
            pass
        try:
            self.timer.cancel()
        except Exception:
            pass
        rclpy.shutdown()

    def _contact_label_at(self, t_s: float) -> int:
        for start_s, end_s in self.contact_intervals:
            if start_s <= float(t_s) <= end_s:
                return 1
        return 0

    def _phase_text_at(self, t_s: float) -> str:
        if not self.contact_intervals:
            return "RECORD"
        for start_s, end_s in self.contact_intervals:
            if start_s <= float(t_s) <= end_s:
                return f"CONTACT until {end_s:.1f}s"
            if float(t_s) < start_s:
                return f"NO CONTACT; contact starts at {start_s:.1f}s"
        return "NO CONTACT; contact finished"

    def _print_progress(self, elapsed_s: float) -> None:
        if self.progress_period_s <= 0.0:
            return
        if elapsed_s - self.last_progress_s < self.progress_period_s:
            return
        self.last_progress_s = float(elapsed_s)
        total_text = "until Ctrl-C" if self.duration_s <= 0.0 else f"{self.duration_s:.1f}s"
        self.get_logger().info(
            f"[timer] t={elapsed_s:5.1f}s / {total_text} | {self._phase_text_at(elapsed_s)} | rows={self.rows_written}"
        )

    def on_timer(self) -> None:
        if self.stop_requested:
            return

        if not self._ready():
            self._warn_if_not_ready()
            return

        now_monotonic_s = time.monotonic()
        if self.record_start_monotonic_s is None:
            self.record_start_monotonic_s = now_monotonic_s
            self.get_logger().info("all topics ready; recording timer starts now")
        elapsed_record_s = now_monotonic_s - self.record_start_monotonic_s
        if self.duration_s > 0.0 and elapsed_record_s >= self.duration_s:
            self._request_stop(f"duration reached ({elapsed_record_s:.2f}s >= {self.duration_s:.2f}s)")
            return

        now_msg = self.get_clock().now().to_msg()
        ros_time_s = float(now_msg.sec) + float(now_msg.nanosec) * 1.0e-9
        if self.first_row_ros_time is None:
            self.first_row_ros_time = ros_time_s
        relative_time_s = ros_time_s - self.first_row_ros_time
        self._print_progress(relative_time_s)

        q: list[float] = []
        qdot: list[float] = []
        q_des: list[float] = []
        tau_cmd: list[float] = []
        tau_meas: list[float] = []
        tau_ff: list[float] = []
        kp: list[float] = []
        kd: list[float] = []
        qdot_des: list[float] = []
        tau_residual: list[float] = []
        tau_residual_corrected: list[float] = []

        for motor_id in range(1, self.dof + 1):
            state = self.state_by_motor[motor_id]
            cmd = self.cmd_by_motor[motor_id]
            q_i = float(state["q"])
            qdot_i = float(state["qdot"])
            q_des_i = float(cmd["q_des"])
            qd_des_i = float(cmd["qd_des"])
            kp_i = max(0.0, float(cmd["kp"]))
            kd_i = max(0.0, float(cmd["kd"]))
            tau_ff_i = float(cmd["tau_ff"])
            tau_cmd_i = kp_i * (q_des_i - q_i) + kd_i * (qd_des_i - qdot_i) + tau_ff_i
            tau_meas_i = float(state["tau_meas"])
            tau_residual_i = tau_meas_i - tau_cmd_i

            q.append(q_i)
            qdot.append(qdot_i)
            q_des.append(q_des_i)
            tau_cmd.append(tau_cmd_i)
            tau_meas.append(tau_meas_i)
            tau_ff.append(tau_ff_i)
            kp.append(kp_i)
            kd.append(kd_i)
            qdot_des.append(qd_des_i)
            tau_residual.append(tau_residual_i)

        if self.residual_offset_duration_s > 0.0 and relative_time_s <= self.residual_offset_duration_s:
            self.residual_offset_count += 1
            for idx, value in enumerate(tau_residual):
                self.residual_offset_sum[idx] += float(value)
                self.residual_offset[idx] = self.residual_offset_sum[idx] / max(self.residual_offset_count, 1)
        tau_residual_corrected = [
            float(value) - float(offset) for value, offset in zip(tau_residual, self.residual_offset)
        ]

        row = (
            [relative_time_s]
            + q
            + qdot
            + q_des
            + tau_cmd
            + tau_meas
            + tau_ff
            + kp
            + kd
            + qdot_des
            + tau_residual
            + tau_residual_corrected
            + [self._contact_label_at(relative_time_s)]
        )
        self.writer.writerow([f"{value:.9f}" if math.isfinite(float(value)) else "nan" for value in row])
        self.rows_written += 1
        if (self.rows_written % max(1, int(round(self.sample_hz)))) == 0:
            self.csv_file.flush()

    def destroy_node(self) -> bool:
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Output CSV path.")
    parser.add_argument("--dof", type=int, default=6, help="Number of motors/joints to record.")
    parser.add_argument("--sample-hz", type=float, default=100.0, help="CSV sampling rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to record. 0 means until Ctrl-C.")
    parser.add_argument(
        "--residual-offset-duration",
        type=float,
        default=2.0,
        help="Initial no-contact seconds used for online tau_residual offset correction.",
    )
    parser.add_argument(
        "--contact-intervals-json",
        default="[]",
        help="Manual contact intervals in recording time, e.g. '[[5.0, 8.0]]'. CSV contact_label becomes 1 inside intervals.",
    )
    parser.add_argument(
        "--progress-period",
        type=float,
        default=1.0,
        help="Seconds between terminal timer prints. Set 0 to disable.",
    )
    parser.add_argument(
        "--state-topic",
        default="/motor_state_array",
        help="Motor state topic to record.",
    )
    parser.add_argument(
        "--cmd-topic",
        default="/motor_cmd_array",
        help=(
            "Motor command topic to record. Use /motor_cmd_array_applied when can_bridge_node "
            "is the only command source and its internal home policy is active."
        ),
    )
    args = parser.parse_args()
    raw_intervals = json.loads(args.contact_intervals_json)
    contact_intervals = []
    for item in raw_intervals:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each contact interval must be [start_s, end_s]")
        contact_intervals.append((float(item[0]), float(item[1])))

    rclpy.init()
    node = RealLogRecorder(
        Path(args.csv),
        args.dof,
        args.sample_hz,
        args.duration,
        args.residual_offset_duration,
        contact_intervals,
        args.progress_period,
        args.cmd_topic,
        args.state_topic,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f"finished recording {node.rows_written} rows to {node.csv_path}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
