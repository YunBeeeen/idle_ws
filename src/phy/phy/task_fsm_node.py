"""Pick-and-place task FSM node.

Orchestrates plan_node and gripper_node to execute pick-and-place tasks.
Vision (object_detector_node) integration is stubbed with hardcoded positions
for control pipeline validation before the ML system is ready.

State machine:
  IDLE → PARSE → READY → DETECT_PICK → PRE_GRASP → GRASP_DESCEND →
  GRASP_CLOSE → VERIFY_GRASP → LIFT → DETECT_PLACE → PRE_PLACE →
  PLACE_DESCEND → GRASP_OPEN → RETRACT → DONE → IDLE

  Any state → RECOVERY → READY (retry, max 3) or FAIL.
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from idle_common.ros_params import declare_typed
from msgs.msg import EETarget, TaskCommand
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


# ---------------------------------------------------------------------------
# Hardcoded workspace positions (replace with vision output later)
# ---------------------------------------------------------------------------
_BLOCK_XY: dict[str, np.ndarray] = {
    "red":   np.array([0.35,  0.10]),
    "blue":  np.array([0.35,  0.00]),
    "green": np.array([0.35, -0.10]),
}
_BASKET_XY = np.array([0.25, 0.20])

# Z parameters [m] — measure and fill after robot calibration
_Z_TABLE: float = 0.00
_Z_BLOCK: float = 0.050
_Z_GRASP: float = _Z_TABLE + _Z_BLOCK * 0.5
_Z_APPROACH: float = _Z_TABLE + _Z_BLOCK + 0.080
_Z_LIFT: float = _Z_TABLE + 0.150
_GRASP_DURATION_S: float = 3.0   # slow descent for grasping
_PLACE_DURATION_S: float = 3.0   # slow descent for placing

# Ready pose (joint angles [rad]) — tune for your robot
_READY_Q_DEG = [0.0, 45.0, 120.0, 0.0, 90.0, 0.0]
_READY_Q_RAD = [math.radians(d) for d in _READY_Q_DEG]


class FSMState(Enum):
    IDLE = auto()
    PARSE = auto()
    READY = auto()
    DETECT_PICK = auto()
    PRE_GRASP = auto()
    GRASP_DESCEND = auto()
    GRASP_CLOSE = auto()
    VERIFY_GRASP = auto()
    LIFT = auto()
    DETECT_PLACE = auto()
    PRE_PLACE = auto()
    PLACE_DESCEND = auto()
    GRASP_OPEN = auto()
    RETRACT = auto()
    DONE = auto()
    RECOVERY = auto()
    FAIL = auto()


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (x,y,z,w) encoding a pure Z-rotation by yaw."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class TaskFSMNode(Node):
    """High-level task coordinator — drives plan_node and gripper_node."""

    def __init__(self) -> None:
        super().__init__("task_fsm_node")

        self.max_retries = int(declare_typed(self, "max_retries", 3))
        self.grasp_verify_ticks = int(declare_typed(self, "grasp_verify_ticks", 10))

        # FSM state
        self._state = FSMState.IDLE
        self._prev_state: Optional[FSMState] = None
        self._task_type: str = ""
        self._src_color: str = ""
        self._dst_color: str = ""
        self._is_basket: bool = False
        self._stack_count: int = 0
        self._retry_count: int = 0
        self._state_ticks: int = 0

        # Plan status tracking
        self._plan_status: str = "IDLE"
        self._grasp_success: Optional[bool] = None
        self._drop_detected: bool = False

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.create_subscription(String, "/plan/status", self._on_plan_status, 10)
        self.create_subscription(Bool, "/gripper/grasp_success", self._on_grasp_success, 10)
        self.create_subscription(Bool, "/gripper/drop_detected", self._on_drop_detected, 10)
        self.create_subscription(TaskCommand, "/task_command", self._on_task_command, 10)

        # Publishers
        self._ee_pub = self.create_publisher(EETarget, "/ee_target", 10)

        # Gripper service clients
        self._gripper_open = self.create_client(Trigger, "/gripper/open")
        self._gripper_close = self.create_client(Trigger, "/gripper/close")

        # FSM timer (10 Hz)
        self.create_timer(0.1, self._on_timer)

        self.get_logger().info("task_fsm_node ready — waiting for TaskCommand on /task_command")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_task_command(self, msg: TaskCommand) -> None:
        if self._state not in (FSMState.IDLE, FSMState.DONE, FSMState.FAIL):
            self.get_logger().warn("task command received while busy — ignoring")
            return
        self._task_type = msg.task_type
        self._src_color = msg.src_color
        self._dst_color = msg.dst_color
        self._is_basket = bool(msg.is_basket)
        self._stack_count = 0
        self._retry_count = 0
        self._transition(FSMState.PARSE)
        self.get_logger().info(
            f"task: {self._task_type} src={self._src_color} dst={self._dst_color}"
        )

    def _on_plan_status(self, msg: String) -> None:
        self._plan_status = msg.data

    def _on_grasp_success(self, msg: Bool) -> None:
        self._grasp_success = bool(msg.data)

    def _on_drop_detected(self, msg: Bool) -> None:
        if msg.data:
            self._drop_detected = True

    # ------------------------------------------------------------------
    # FSM timer
    # ------------------------------------------------------------------

    def _on_timer(self) -> None:
        self._state_ticks += 1
        s = self._state

        if s == FSMState.PARSE:
            self._transition(FSMState.READY)

        elif s == FSMState.READY:
            self._send_ready_pose()
            self._transition(FSMState.DETECT_PICK)

        elif s == FSMState.DETECT_PICK:
            # Vision stub: use hardcoded position
            self._transition(FSMState.PRE_GRASP)

        elif s == FSMState.PRE_GRASP:
            if self._state_ticks == 1:
                self._send_approach(self._src_color, _Z_APPROACH, 0.0,
                                    use_safe_transit=True)
            elif self._plan_done():
                self._transition(FSMState.GRASP_DESCEND)

        elif s == FSMState.GRASP_DESCEND:
            if self._state_ticks == 1:
                self._send_approach(self._src_color, _Z_GRASP, _GRASP_DURATION_S)
            elif self._plan_done():
                self._transition(FSMState.GRASP_CLOSE)

        elif s == FSMState.GRASP_CLOSE:
            if self._state_ticks == 1:
                self._grasp_success = None
                self._call_gripper(close=True)
            elif self._grasp_success is not None:
                if self._grasp_success:
                    self._transition(FSMState.VERIFY_GRASP)
                else:
                    self._transition(FSMState.RECOVERY)

        elif s == FSMState.VERIFY_GRASP:
            if self._state_ticks >= self.grasp_verify_ticks:
                if self._grasp_success:
                    self._transition(FSMState.LIFT)
                else:
                    self._transition(FSMState.RECOVERY)

        elif s == FSMState.LIFT:
            if self._state_ticks == 1:
                self._drop_detected = False
                self._send_approach(self._src_color, _Z_LIFT, 0.0)
            elif self._drop_detected:
                self.get_logger().warn("drop detected during lift")
                self._transition(FSMState.RECOVERY)
            elif self._plan_done():
                self._transition(FSMState.DETECT_PLACE)

        elif s == FSMState.DETECT_PLACE:
            # Vision stub: use hardcoded position
            self._transition(FSMState.PRE_PLACE)

        elif s == FSMState.PRE_PLACE:
            if self._state_ticks == 1:
                z_pre = self._place_z_approach()
                self._send_place_approach(z_pre, 0.0, use_safe_transit=True)
            elif self._drop_detected:
                self._transition(FSMState.RECOVERY)
            elif self._plan_done():
                self._transition(FSMState.PLACE_DESCEND)

        elif s == FSMState.PLACE_DESCEND:
            if self._state_ticks == 1:
                self._send_place_approach(self._place_z_goal(), _PLACE_DURATION_S)
            elif self._plan_done():
                self._transition(FSMState.GRASP_OPEN)

        elif s == FSMState.GRASP_OPEN:
            if self._state_ticks == 1:
                self._call_gripper(close=False)
            elif self._state_ticks >= 5:
                self._transition(FSMState.RETRACT)

        elif s == FSMState.RETRACT:
            if self._state_ticks == 1:
                self._send_place_approach(_Z_LIFT, 0.0)
            elif self._plan_done():
                self._stack_count += 1
                self._transition(FSMState.DONE)

        elif s == FSMState.RECOVERY:
            if self._state_ticks == 1:
                self._call_gripper(close=False)
                self._retry_count += 1
                if self._retry_count >= self.max_retries:
                    self.get_logger().error("max retries reached — FAIL")
                    self._transition(FSMState.FAIL)
                    return
                self.get_logger().warn(f"recovery {self._retry_count}/{self.max_retries}")
            elif self._state_ticks >= 3:
                self._transition(FSMState.READY)

        elif s == FSMState.DONE:
            self.get_logger().info("task complete")
            self._transition(FSMState.IDLE)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: FSMState) -> None:
        self.get_logger().info(f"FSM: {self._state.name} → {new_state.name}")
        self._prev_state = self._state
        self._state = new_state
        self._state_ticks = 0
        self._plan_status = "IDLE"

    def _plan_done(self) -> bool:
        return self._plan_status == "DONE"

    def _place_z_approach(self) -> float:
        n = self._stack_count
        return _Z_TABLE + n * _Z_BLOCK + 0.080

    def _place_z_goal(self) -> float:
        n = self._stack_count
        return _Z_TABLE + n * _Z_BLOCK + _Z_BLOCK * 0.5

    def _send_approach(self, color: str, z: float, duration_s: float,
                       use_safe_transit: bool = False) -> None:
        xy = _BLOCK_XY.get(color, np.array([0.35, 0.0]))
        self._send_ee_target(float(xy[0]), float(xy[1]), z, yaw=0.0,
                             duration_s=duration_s, use_safe_transit=use_safe_transit)

    def _send_place_approach(self, z: float, duration_s: float,
                             use_safe_transit: bool = False) -> None:
        if self._is_basket:
            xy = _BASKET_XY
        else:
            xy = _BLOCK_XY.get(self._dst_color, np.array([0.35, 0.0]))
        self._send_ee_target(float(xy[0]), float(xy[1]), z, yaw=0.0,
                             duration_s=duration_s, use_safe_transit=use_safe_transit)

    def _send_ready_pose(self) -> None:
        # Ready pose sent as a fixed joint-space config would require a different interface.
        # For now, send a fixed Cartesian position above workspace center.
        self._send_ee_target(0.30, 0.0, _Z_LIFT + 0.05, yaw=0.0, duration_s=0.0)

    def _send_ee_target(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        duration_s: float,
        use_safe_transit: bool = False,
    ) -> None:
        qx, qy, qz, qw = _yaw_quat(yaw)
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
        msg.duration_override_s = float(duration_s)
        msg.use_safe_transit = use_safe_transit
        self._ee_pub.publish(msg)

    def _call_gripper(self, close: bool) -> None:
        if close:
            if self._gripper_close.service_is_ready():
                self._gripper_close.call_async(Trigger.Request())
            else:
                self.get_logger().warn("gripper/close service not ready")
        else:
            if self._gripper_open.service_is_ready():
                self._gripper_open.call_async(Trigger.Request())
            else:
                self.get_logger().warn("gripper/open service not ready")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TaskFSMNode()
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
