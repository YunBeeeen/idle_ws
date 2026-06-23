"""Motor 7 gripper controller with grasp-success and drop detection.

Grasp detection logic:
  - Closing command: q_des = q_closed_min - delta (tries to over-close).
  - Block present:   gripper stops between q_open and q_closed_min → grasped.
  - Empty grasp:     gripper reaches q_closed_min - delta → no block.
  - Drop detection:  during GRASPED state, q_actual moves toward q_closed_min
                     (resistance gone) OR tau_measured drops below threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

import rclpy
from idle_common.ros_params import declare_typed
from msgs.msg import MotorCMD, MotorCMDArray, MotorStateArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray
from std_srvs.srv import Trigger


class GripperState(Enum):
    OPEN = auto()
    CLOSING = auto()
    GRASPED = auto()
    OPENING = auto()
    FAIL = auto()


@dataclass
class MotorSample:
    q: float = 0.0
    qd: float = 0.0
    tau_measured: float = 0.0
    last_seen_s: float = float("-inf")


class GripperNode(Node):
    """Standalone Motor 7 controller — runs in parallel with plan_node.

    Topics published:
      /gripper/grasp_success  (Bool) — latched, True when block is confirmed grasped.
      /gripper/drop_detected  (Bool) — True when block is detected as dropped.
      /gripper/state          (Float32MultiArray) — [q_actual, tau_measured, state_id].

    Services:
      /gripper/open   (Trigger) — command open.
      /gripper/close  (Trigger) — command close (starts grasp sequence).
    """

    MOTOR_ID = 7

    def __init__(self) -> None:
        super().__init__("gripper_node")

        # Parameters — tune via YAML or CLI
        self.q_open = float(declare_typed(self, "q_open", 0.0))
        self.q_closed_min = float(declare_typed(self, "q_closed_min", 0.5))
        self.delta_overclose = float(declare_typed(self, "delta_overclose", 0.05))
        self.tau_drop_threshold = float(declare_typed(self, "tau_drop_threshold", 0.1))
        self.grasp_settle_ticks = int(declare_typed(self, "grasp_settle_ticks", 15))
        self.control_hz = float(declare_typed(self, "control_hz", 100.0))
        self.state_timeout_s = float(declare_typed(self, "state_timeout_s", 0.5))
        self.kp = float(declare_typed(self, "kp", 2.0))
        self.kd = float(declare_typed(self, "kd", 0.1))
        self.tau_limit = 1.6  # Motor 7 hardware limit [Nm]

        self._state = GripperState.OPEN
        self._motor = MotorSample()
        self._q_des = self.q_open
        self._settle_count = 0

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._state_sub = self.create_subscription(
            MotorStateArray, "/motor_state_array", self._on_state, qos_be
        )
        self._cmd_pub = self.create_publisher(MotorCMDArray, "/motor_cmd_array", qos_be)
        self._grasp_pub = self.create_publisher(Bool, "/gripper/grasp_success", 10)
        self._drop_pub = self.create_publisher(Bool, "/gripper/drop_detected", 10)
        self._status_pub = self.create_publisher(Float32MultiArray, "/gripper/state", 10)

        self.create_service(Trigger, "/gripper/open", self._srv_open)
        self.create_service(Trigger, "/gripper/close", self._srv_close)

        period_s = max(1.0 / self.control_hz, 1.0e-3)
        self.create_timer(period_s, self._on_timer)

        self.get_logger().info(
            f"gripper_node ready: q_open={self.q_open:.3f} "
            f"q_closed_min={self.q_closed_min:.3f} "
            f"delta={self.delta_overclose:.3f} kp={self.kp}"
        )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _on_timer(self) -> None:
        now_s = self._now_s()
        stale = (now_s - self._motor.last_seen_s) > self.state_timeout_s
        q = self._motor.q
        tau = self._motor.tau_measured

        if self._state == GripperState.CLOSING:
            # Wait for gripper to settle, then check grasp
            self._settle_count += 1
            if self._settle_count >= self.grasp_settle_ticks:
                q_cmd = self.q_closed_min - self.delta_overclose
                if q > (self.q_closed_min - 0.01):
                    # Fully closed → no block
                    self._state = GripperState.FAIL
                    self.get_logger().warn("grasp failed: gripper closed on air")
                    self._publish_grasp(False)
                else:
                    # Stopped before fully closed → block present
                    self._state = GripperState.GRASPED
                    self.get_logger().info(f"grasped: q_actual={q:.4f}")
                    self._publish_grasp(True)

        elif self._state == GripperState.GRASPED and not stale:
            # Drop detection: resistance gone → q moves toward q_cmd
            q_cmd = self.q_closed_min - self.delta_overclose
            block_gone = (q < q_cmd + 0.01) or (abs(tau) < self.tau_drop_threshold)
            if block_gone:
                self._state = GripperState.OPEN
                self.get_logger().warn("drop detected")
                msg = Bool()
                msg.data = True
                self._drop_pub.publish(msg)

        # Command generation
        self._send_cmd()
        self._publish_state()

    def _send_cmd(self) -> None:
        stamp = self.get_clock().now().to_msg()
        cmd = MotorCMD()
        cmd.stamp = stamp
        cmd.motor_id = self.MOTOR_ID
        cmd.q_des = float(self._q_des)
        cmd.qd_des = 0.0
        cmd.kp = self.kp
        cmd.kd = self.kd
        cmd.tau_ff = 0.0
        arr = MotorCMDArray()
        arr.stamp = stamp
        arr.commands = [cmd]
        self._cmd_pub.publish(arr)

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def _srv_open(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self._q_des = self.q_open
        self._state = GripperState.OPENING
        self._settle_count = 0
        self.get_logger().info("gripper: open commanded")
        res.success = True
        res.message = "opening"
        return res

    def _srv_close(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self._q_des = self.q_closed_min - self.delta_overclose
        self._state = GripperState.CLOSING
        self._settle_count = 0
        self.get_logger().info("gripper: close commanded")
        res.success = True
        res.message = "closing"
        return res

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_state(self, msg: MotorStateArray) -> None:
        stamp_s = float(msg.stamp.sec) + float(msg.stamp.nanosec) * 1.0e-9
        for s in msg.states:
            if int(s.motor_id) == self.MOTOR_ID:
                self._motor = MotorSample(
                    q=float(s.q),
                    qd=float(s.qd),
                    tau_measured=float(s.tau),
                    last_seen_s=stamp_s,
                )
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_grasp(self, success: bool) -> None:
        msg = Bool()
        msg.data = success
        self._grasp_pub.publish(msg)

    def _publish_state(self) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(self._motor.q),
            float(self._motor.tau_measured),
            float(self._state.value),
        ]
        self._status_pub.publish(msg)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GripperNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
