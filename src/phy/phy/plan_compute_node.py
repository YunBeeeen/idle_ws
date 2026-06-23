"""Planning-only node — IK + collision + trajectory computation.

Subscribes to /ee_target and /motor_state_array.
Publishes /computed_plan when a valid plan is ready.

Runs as a separate process from ctrl_node so GIL contention during
IK/collision never starves the 250 Hz control loop.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from msgs.msg import ComputedPlan, EETarget, MotorStateArray
from std_msgs.msg import String
from idle_common.control_tuning import control_params_for_motor
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP
from idle_common.paths import resolve_share_file
from idle_common.ros_params import declare_typed
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from phy.collision import CollisionChecker
from phy.ik import IKConfig, IKSolver
from phy.plan import Planner, PlannerConfig
from phy.robot_model import RobotModel


def _quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


@dataclass
class _MotorSample:
    q: float = 0.0
    last_seen_s: float = float("-inf")


class PlanComputeNode(Node):
    def __init__(self) -> None:
        super().__init__("plan_compute_node")

        strip_str = lambda v: str(v).strip()
        self.state_timeout_s = declare_typed(self, "state_timeout_s", 0.2)
        self.target_frame = declare_typed(self, "target_frame", "gripper", cast=strip_str)
        v_max = declare_typed(self, "planner_v_max", 1.0)
        a_max = declare_typed(self, "planner_a_max", 1.0)
        min_traj_duration = declare_typed(self, "planner_min_traj_duration", 1.5)
        urdf_path_text = declare_typed(self, "urdf_path", "", cast=strip_str)

        urdf_path = resolve_share_file("sim", "urdf/robot.urdf", urdf_path_text)
        srdf_path = resolve_share_file("sim", "srdf/robot.srdf", "")
        from ament_index_python.packages import get_package_share_directory
        sim_share_parent = str(
            __import__("pathlib").Path(get_package_share_directory("sim")).parent
        )

        motor_joint_map = dict(DEFAULT_MOTOR_JOINT_MAP)
        self.robot = RobotModel(urdf_path, motor_joint_map)
        self.motor_ids = self.robot.ordered_motor_ids

        self.collision = CollisionChecker(
            self.robot, srdf_path=srdf_path, package_dirs=[sim_share_parent],
        )
        self.declare_parameter("cage_collision", False)
        self.add_on_set_parameters_callback(self._on_parameters)

        controlled_joints = tuple(
            self.robot.bindings[m].joint_name for m in self.motor_ids
        )
        self.ik = IKSolver(
            urdf_path,
            IKConfig(target_frame=self.target_frame, controlled_joints=controlled_joints),
        )
        self.planner = Planner(
            self.robot, self.collision, self.ik,
            PlannerConfig(v_max=v_max, a_max=a_max, min_traj_duration=min_traj_duration),
        )

        self.state_by_motor: dict[int, _MotorSample] = {
            m: _MotorSample() for m in self.motor_ids
        }

        self._plan_lock = threading.Lock()
        self._plan_serial: int = 0

        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.state_sub = self.create_subscription(
            MotorStateArray, "/motor_state_array", self.on_state_array, qos_state
        )
        self.target_sub = self.create_subscription(
            PoseStamped, "/ee_target_pose", self.on_target, 10
        )
        self.ee_target_sub = self.create_subscription(
            EETarget, "/ee_target", self.on_ee_target, 10
        )
        self.plan_pub = self.create_publisher(ComputedPlan, "/computed_plan", 10)
        self.status_pub = self.create_publisher(String, "/plan/status", 10)

        self.get_logger().info(
            f"plan_compute_node initialized: target_frame={self.target_frame} "
            f"v_max={v_max} a_max={a_max} motors={list(self.motor_ids)}"
        )

    # ------------------------------------------------------------------
    # Parameter callbacks
    # ------------------------------------------------------------------

    def _on_parameters(self, params: list) -> "SetParametersResult":
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "cage_collision":
                enabled = bool(p.value)
                self.collision.set_cage_enabled(enabled)
                self.get_logger().info(f"cage_collision → {enabled}")
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def on_state_array(self, msg: MotorStateArray) -> None:
        stamp_s = float(msg.stamp.sec) + float(msg.stamp.nanosec) * 1.0e-9
        for state in msg.states:
            motor_id = int(state.motor_id)
            if motor_id in self.state_by_motor:
                self.state_by_motor[motor_id].q = float(state.q)
                self.state_by_motor[motor_id].last_seen_s = stamp_s

    def on_target(self, msg: PoseStamped) -> None:
        target_xyz = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        yaw = _quaternion_to_yaw(
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        )
        self._start_planning(target_xyz, yaw, 0.0)

    def on_ee_target(self, msg: EETarget) -> None:
        target_xyz = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=float,
        )
        yaw = _quaternion_to_yaw(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
        )
        self._start_planning(target_xyz, yaw, float(msg.duration_override_s))

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _start_planning(self, target_xyz: np.ndarray, yaw: float, duration_override_s: float) -> None:
        start_q = self._current_q()
        if start_q is None:
            self.get_logger().warn("target received before fresh state — ignoring")
            return

        with self._plan_lock:
            self._plan_serial += 1
            my_serial = self._plan_serial

        self._publish_status("PLANNING")
        self.get_logger().info(
            f"planning [{my_serial}]: xyz={target_xyz.tolist()} yaw={math.degrees(yaw):+.1f}°"
        )
        threading.Thread(
            target=self._bg_plan,
            args=(target_xyz, yaw, start_q, my_serial, duration_override_s),
            daemon=True,
        ).start()

    def _bg_plan(
        self,
        target_xyz: np.ndarray,
        target_yaw: float,
        start_q: np.ndarray,
        my_serial: int,
        duration_override_s: float,
    ) -> None:
        min_dur = duration_override_s if duration_override_s > 0.0 else None
        cfg = self.planner.cfg

        v_max_arr = np.array([
            float(control_params_for_motor(m).get("v_max", cfg.v_max))
            for m in self.motor_ids
        ])
        a_max_arr = np.array([
            float(control_params_for_motor(m).get("a_max", cfg.a_max))
            for m in self.motor_ids
        ])

        result = self.planner.plan_motion(
            target_xyz=target_xyz, target_yaw=target_yaw,
            start_q=start_q, v_max=v_max_arr, a_max=a_max_arr,
            min_duration=min_dur,
        )

        # Discard stale (newer target arrived)
        with self._plan_lock:
            if self._plan_serial != my_serial:
                self.get_logger().info(f"[{my_serial}] stale plan discarded")
                return

        if result is None:
            self.get_logger().warn(f"[{my_serial}] IK unreachable — discarded")
            self._publish_status("FAIL")
            return

        n_dof = len(self.motor_ids)
        msg = ComputedPlan()
        msg.serial = int(my_serial)
        msg.n_dof = int(n_dof)

        if isinstance(result, tuple):
            leg1, leg2 = result
            if not (leg1.collision_safe and leg2.collision_safe):
                self.get_logger().warn(f"[{my_serial}] fold plan collision — discarded")
                self._publish_status("FAIL")
                return
            msg.start_q = start_q.tolist()
            msg.coeffs = leg1.trajectory.coeffs.flatten().tolist()
            msg.duration = float(leg1.duration_s)
            msg.end_q = leg1.end_q.tolist()
            msg.target_xyz = leg1.target_xyz.tolist()
            msg.target_yaw = float(leg1.target_yaw)
            msg.has_leg2 = True
            msg.leg2_coeffs = leg2.trajectory.coeffs.flatten().tolist()
            msg.leg2_duration = float(leg2.duration_s)
            msg.leg2_end_q = leg2.end_q.tolist()
            msg.leg2_target_xyz = leg2.target_xyz.tolist()
            msg.leg2_target_yaw = float(leg2.target_yaw)
            self.get_logger().info(
                f"[{my_serial}] fold-and-rotate: leg1={leg1.duration_s:.2f}s leg2={leg2.duration_s:.2f}s"
            )
        else:
            plan = result
            if not plan.collision_safe:
                self.get_logger().warn(
                    f"[{my_serial}] collision at sample {plan.collision_first_sample} — discarded"
                )
                self._publish_status("FAIL")
                return
            msg.start_q = start_q.tolist()
            msg.coeffs = plan.trajectory.coeffs.flatten().tolist()
            msg.duration = float(plan.duration_s)
            msg.end_q = plan.end_q.tolist()
            msg.target_xyz = plan.target_xyz.tolist()
            msg.target_yaw = float(plan.target_yaw)
            msg.has_leg2 = False
            self.get_logger().info(
                f"[{my_serial}] plan ready: duration={plan.duration_s:.2f}s"
            )

        self.plan_pub.publish(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_q(self) -> Optional[np.ndarray]:
        now_s = self._now_s()
        for sample in self.state_by_motor.values():
            if (
                not math.isfinite(sample.last_seen_s)
                or (now_s - sample.last_seen_s) > self.state_timeout_s
            ):
                return None
        return np.array([self.state_by_motor[m].q for m in self.motor_ids], dtype=float)

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = PlanComputeNode()
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
