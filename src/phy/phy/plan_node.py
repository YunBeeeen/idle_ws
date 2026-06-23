"""Control-only node — 250 Hz trajectory execution.

Receives pre-computed plans from plan_compute_node via /computed_plan and
executes them as quintic trajectories with time-warp and hold logic.
No IK / collision in this process — GIL pressure from planning never
interrupts the control loop.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from msgs.msg import ComputedPlan, MotorCMDArray, MotorStateArray
from std_msgs.msg import String
from idle_common.control_tuning import control_params_for_motor
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP, DEFAULT_TAU_LIMIT_BY_MOTOR
from idle_common.paths import resolve_share_file
from idle_common.ros_params import declare_typed
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from msgs.msg import MotorCMD
from phy.plan import Plan, PlannerConfig
from phy.robot_model import RobotModel
from phy.traj import QuinticPlan, plan_quintic, sample_quintic


@dataclass
class MotorSample:
    q: float = 0.0
    qd: float = 0.0
    tau_measured: float = 0.0
    last_seen_s: float = float("-inf")
    qdd_est: float = 0.0


@dataclass
class _ActiveTrajectory:
    plan: Plan
    start_time_s: float


class PlanNode(Node):
    """Trajectory execution node (ctrl). Receives plans from plan_compute_node."""

    def __init__(self) -> None:
        super().__init__("plan_node")

        strip_str = lambda v: str(v).strip()
        self.control_hz = declare_typed(self, "control_hz", 250.0)
        self.state_timeout_s = declare_typed(self, "state_timeout_s", 0.2)
        self.kp_max = declare_typed(self, "kp_max", 50.0)
        self.kd_max = declare_typed(self, "kd_max", 10.0)
        v_max = declare_typed(self, "planner_v_max", 1.0)
        a_max = declare_typed(self, "planner_a_max", 1.0)
        min_traj_duration = declare_typed(self, "planner_min_traj_duration", 1.5)
        disable_gravity = declare_typed(self, "disable_gravity", False)
        self.unlimited_tau = bool(declare_typed(self, "unlimited_tau", False))
        urdf_path_text = declare_typed(self, "urdf_path", "", cast=strip_str)

        self.rewarp_threshold_rad = float(declare_typed(self, "rewarp_threshold_rad", 0.15))
        self.warp_q_lo_rad = float(declare_typed(self, "warp_q_lo_rad", 0.12))
        self.warp_q_hi_rad = float(declare_typed(self, "warp_q_hi_rad", 0.40))

        # Default v/a for rewarp (per-joint YAML overrides these)
        self._planner_cfg = PlannerConfig(v_max=v_max, a_max=a_max, min_traj_duration=min_traj_duration)

        urdf_path = resolve_share_file("sim", "urdf/robot.urdf", urdf_path_text)
        motor_joint_map = dict(DEFAULT_MOTOR_JOINT_MAP)
        self.robot = RobotModel(urdf_path, motor_joint_map)
        self.disable_gravity = bool(disable_gravity)
        if self.disable_gravity:
            import pinocchio as pin
            self.robot.model.gravity = pin.Motion.Zero()
            self.get_logger().warn("disable_gravity=True — gravity zeroed")

        self.motor_ids = self.robot.ordered_motor_ids
        self.tau_limit_by_motor = {
            m: float(DEFAULT_TAU_LIMIT_BY_MOTOR.get(m, float("inf"))) for m in self.motor_ids
        }
        _q_lo, _q_hi = self.robot.joint_limits()
        self.q_min_by_motor = {m: float(_q_lo[i]) for i, m in enumerate(self.motor_ids)}
        self.q_max_by_motor = {m: float(_q_hi[i]) for i, m in enumerate(self.motor_ids)}
        self.state_by_motor = {m: MotorSample() for m in self.motor_ids}
        self.active: Optional[_ActiveTrajectory] = None
        self._hold_q: Optional[dict[int, float]] = None       # 명령용 (actual_q 기반)
        self._hold_target_q: Optional[dict[int, float]] = None  # 로그용 (q_final 기반)

        self._plan_lock = threading.Lock()
        self._pending_plan: Optional[Plan] = None
        self._queued_leg2: Optional[Plan] = None

        self._vt_elapsed_s: float = 0.0
        self._vt_last_wall_s: float = float("-inf")
        self._prev_max_err: float = 0.0

        self.traj_stall_timeout_s = float(declare_typed(self, "traj_stall_timeout_s", 10.0))
        self._warp_stall_s: float = 0.0

        self._warn_times: dict[str, float] = {}

        self._duration_override_s: float = 0.0

        self._hold_log_start_s: float = 0.0
        self._hold_log_count: int = 3

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.state_sub = self.create_subscription(
            MotorStateArray, "/motor_state_array", self.on_state_array, qos_state
        )
        self.computed_plan_sub = self.create_subscription(
            ComputedPlan, "/computed_plan", self.on_computed_plan, 10
        )
        self.cmd_pub = self.create_publisher(MotorCMDArray, "/motor_cmd_array", qos_cmd)
        self.status_pub = self.create_publisher(String, "/plan/status", 10)

        period_s = max(1.0 / self.control_hz, 1.0e-4)
        self.control_timer = self.create_timer(period_s, self.on_timer)

        self.get_logger().info(
            f"plan_node (ctrl) initialized: hz={self.control_hz:.1f} "
            f"motors={list(self.motor_ids)} "
            f"warp_lo={self.warp_q_lo_rad:.3f} warp_hi={self.warp_q_hi_rad:.3f} rad"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def on_state_array(self, msg: MotorStateArray) -> None:
        stamp_s = float(msg.stamp.sec) + float(msg.stamp.nanosec) * 1.0e-9
        for state in msg.states:
            motor_id = int(state.motor_id)
            if motor_id not in self.state_by_motor:
                continue
            prev = self.state_by_motor[motor_id]
            new_qd = float(state.qd)
            dt = stamp_s - prev.last_seen_s
            qdd_est = (new_qd - prev.qd) / dt if (0.0 < dt < 0.1) else prev.qdd_est
            self.state_by_motor[motor_id] = MotorSample(
                q=float(state.q),
                qd=new_qd,
                tau_measured=float(state.tau),
                last_seen_s=stamp_s,
                qdd_est=qdd_est,
            )

    def on_computed_plan(self, msg: ComputedPlan) -> None:
        """Deserialize ComputedPlan and deposit into pending slot."""
        n = int(msg.n_dof)
        start_q = np.array(msg.start_q, dtype=float)
        coeffs = np.array(msg.coeffs, dtype=float).reshape(n, 6)
        end_q = np.array(msg.end_q, dtype=float)
        target_xyz = np.array(msg.target_xyz, dtype=float)

        traj = QuinticPlan(
            duration=float(msg.duration),
            coeffs=coeffs,
            q_start=start_q,
            q_goal=end_q,
        )
        plan = Plan(
            trajectory=traj,
            start_q=start_q,
            end_q=end_q,
            duration_s=float(msg.duration),
            collision_safe=True,
            collision_first_sample=-1,
            target_xyz=target_xyz,
            target_yaw=float(msg.target_yaw),
            metadata={"serial": int(msg.serial)},
        )

        leg2: Optional[Plan] = None
        if msg.has_leg2:
            leg2_coeffs = np.array(msg.leg2_coeffs, dtype=float).reshape(n, 6)
            leg2_end_q = np.array(msg.leg2_end_q, dtype=float)
            leg2_traj = QuinticPlan(
                duration=float(msg.leg2_duration),
                coeffs=leg2_coeffs,
                q_start=end_q,
                q_goal=leg2_end_q,
            )
            leg2 = Plan(
                trajectory=leg2_traj,
                start_q=end_q,
                end_q=leg2_end_q,
                duration_s=float(msg.leg2_duration),
                collision_safe=True,
                collision_first_sample=-1,
                target_xyz=np.array(msg.leg2_target_xyz, dtype=float),
                target_yaw=float(msg.leg2_target_yaw),
                metadata={"serial": int(msg.serial), "leg": 2},
            )

        with self._plan_lock:
            self._pending_plan = plan
            self._queued_leg2 = leg2

        self.get_logger().info(
            f"[{msg.serial}] plan received: duration={msg.duration:.2f}s"
            + (f" + leg2={msg.leg2_duration:.2f}s" if msg.has_leg2 else "")
        )

    # ------------------------------------------------------------------
    # Control timer
    # ------------------------------------------------------------------

    def on_timer(self) -> None:
        now_s = self._now_s()
        if not self._state_fresh(now_s):
            return

        q_by_motor = {m: self.state_by_motor[m].q for m in self.motor_ids}
        try:
            tau_g_by_motor = self.robot.gravity_torque(q_by_motor)
        except Exception as exc:
            self.get_logger().warn(f"gravity computation failed: {exc}; skipping tick")
            return

        pending: Optional[Plan] = None
        with self._plan_lock:
            if self._pending_plan is not None:
                pending = self._pending_plan
                self._pending_plan = None
        if pending is not None:
            self._commit_plan(pending, now_s)

        if self.active is not None:
            dt_wall = self._now_s() - self._vt_last_wall_s
            self._vt_last_wall_s = self._now_s()
            warp = self._compute_warp(self._prev_max_err)
            self._vt_elapsed_s += dt_wall * warp

            if warp < 0.05:
                self._warp_stall_s += dt_wall
                if self._warp_stall_s >= self.traj_stall_timeout_s:
                    self.get_logger().warn(
                        f"[SAFETY] trajectory stalled {self._warp_stall_s:.1f}s — discarding"
                    )
                    q_stall, _, _ = self.active.plan.sample(self._vt_elapsed_s)
                    self._hold_q = {m: float(q_stall[i]) for i, m in enumerate(self.motor_ids)}
                    self.active = None
                    self._prev_max_err = 0.0
                    self._warp_stall_s = 0.0
                    self._publish_status("FAIL")
                    self._publish(self._hold_cmds(tau_g_by_motor))
                    return
            else:
                self._warp_stall_s = 0.0

            if self._vt_elapsed_s >= self.active.plan.duration_s:
                q_final, _, _ = self.active.plan.sample(self.active.plan.duration_s)
                # _hold_q는 actual_q로 설정해 다음 궤적 시작 시 q_des 점프를 방지.
                # _hold_target_q는 q_final로 설정해 hold 오차 로그에 사용.
                self._hold_q = {m: self.state_by_motor[m].q for m in self.motor_ids}
                self._hold_target_q = {m: float(q_final[i]) for i, m in enumerate(self.motor_ids)}
                self.active = None
                self._prev_max_err = 0.0
                self._warp_stall_s = 0.0

                with self._plan_lock:
                    leg2 = self._queued_leg2
                    self._queued_leg2 = None
                if leg2 is not None:
                    self.get_logger().info("two-leg plan leg1 done — starting leg2")
                    actual_qd = np.array(
                        [self.state_by_motor[m].qd for m in self.motor_ids], dtype=float
                    )
                    self._commit_plan(leg2, now_s, start_qd=actual_qd)
                    cmd_values, _ = self._trajectory_cmds(0.0, 1.0, tau_g_by_motor)
                else:
                    self.get_logger().info("trajectory complete — holding final pose")
                    self._publish_status("DONE")
                    self._hold_log_start_s = now_s
                    self._hold_log_count = 0
                    cmd_values = self._hold_cmds(tau_g_by_motor)
            else:
                cmd_values, max_err = self._trajectory_cmds(
                    self._vt_elapsed_s, warp, tau_g_by_motor
                )
                self._prev_max_err = max_err
        else:
            self._warp_stall_s = 0.0
            cmd_values = self._hold_cmds(tau_g_by_motor)
            if self._hold_target_q is not None and self._hold_log_count < 3:
                elapsed = now_s - self._hold_log_start_s
                if elapsed >= (self._hold_log_count + 1) * 1.0:
                    errs = [
                        self.state_by_motor[m].q - self._hold_target_q[m]
                        for m in self.motor_ids
                    ]
                    parts = "  ".join(
                        f"j{i+1}={errs[i]:+.4f}" for i in range(len(self.motor_ids))
                    )
                    self.get_logger().info(
                        f"[hold {self._hold_log_count + 1}/3 +{elapsed:.1f}s] "
                        f"err(rad): {parts}  max={max(abs(e) for e in errs):.4f}"
                    )
                    self._hold_log_count += 1

        self._publish(cmd_values)

    # ------------------------------------------------------------------
    # Plan commit — simplified rewarp without collision check
    # ------------------------------------------------------------------

    def _commit_plan(
        self,
        pending: Plan,
        now_s: float,
        start_qd: Optional[np.ndarray] = None,
    ) -> None:
        actual_q = np.array([self.state_by_motor[m].q for m in self.motor_ids], dtype=float)
        drift = float(np.linalg.norm(actual_q - pending.start_q))

        if drift > self.rewarp_threshold_rad:
            self.get_logger().warn(
                f"commit drift {drift:.3f} rad > threshold {self.rewarp_threshold_rad:.3f} — rewarping"
            )

        if drift > 1e-4:
            plan = self._rewarp(pending, actual_q, v_start=start_qd)
        else:
            plan = pending

        self.active = _ActiveTrajectory(plan=plan, start_time_s=now_s)
        self._vt_elapsed_s = 0.0
        self._vt_last_wall_s = self._now_s()
        self._prev_max_err = 0.0
        self._hold_q = None
        self._hold_target_q = None
        self._publish_status("EXECUTING")
        self.get_logger().info(
            f"plan committed: xyz={plan.target_xyz.tolist()} "
            f"yaw={math.degrees(plan.target_yaw):+.1f}° "
            f"duration={plan.duration_s:.2f}s drift={drift:.4f}rad"
        )

    def _rewarp(
        self,
        pending: Plan,
        actual_q: np.ndarray,
        v_start: Optional[np.ndarray] = None,
    ) -> Plan:
        """Rebuild quintic from actual_q to end_q (no collision check)."""
        cfg = self._planner_cfg
        n = len(actual_q)
        zeros = np.zeros(n)
        v_s = np.asarray(v_start, dtype=float) if v_start is not None else zeros
        v_max_arr = np.array([
            float(control_params_for_motor(m).get("v_max", cfg.v_max))
            for m in self.motor_ids
        ])
        a_max_arr = np.array([
            float(control_params_for_motor(m).get("a_max", cfg.a_max))
            for m in self.motor_ids
        ])
        traj = plan_quintic(
            q_start=actual_q,
            q_goal=pending.end_q,
            v_start=v_s,
            v_goal=zeros,
            v_max=v_max_arr,
            a_max=a_max_arr,
            min_duration=pending.duration_s,
        )
        return Plan(
            trajectory=traj,
            start_q=actual_q.copy(),
            end_q=pending.end_q.copy(),
            duration_s=traj.duration,
            collision_safe=True,
            collision_first_sample=-1,
            target_xyz=pending.target_xyz.copy(),
            target_yaw=pending.target_yaw,
            metadata={**pending.metadata, "rewarped_ctrl": True},
        )

    # ------------------------------------------------------------------
    # Command generation
    # ------------------------------------------------------------------

    def _trajectory_cmds(
        self,
        vt_s: float,
        warp: float,
        tau_g_by_motor: dict[int, float],
    ) -> tuple[dict[int, dict[str, float]], float]:
        assert self.active is not None
        traj = self.active.plan.trajectory
        q_des_vec, qd_des_vec, _ = self.active.plan.sample(vt_s)

        t = float(np.clip(vt_s, 0.0, traj.duration))
        t2, t3 = t * t, t * t * t
        c = traj.coeffs
        qdd_des_vec = (
            2.0 * c[:, 2]
            + 6.0 * c[:, 3] * t
            + 12.0 * c[:, 4] * t2
            + 20.0 * c[:, 5] * t3
        )

        tuning_list = [control_params_for_motor(m) for m in self.motor_ids]
        goal_modes = [bool(tuning_list[i].get("goal_mode", 0)) for i in range(len(self.motor_ids))]

        q_rnea: dict[int, float] = {}
        qd_rnea: dict[int, float] = {}
        qdd_rnea: dict[int, float] = {}
        for i, m in enumerate(self.motor_ids):
            if goal_modes[i]:
                s = self.state_by_motor[m]
                q_rnea[m] = float(s.q)
                qd_rnea[m] = float(s.qd)
                qdd_rnea[m] = 0.0
            else:
                q_rnea[m] = float(q_des_vec[i])
                qd_rnea[m] = float(qd_des_vec[i])
                qdd_rnea[m] = float(qdd_des_vec[i])

        tau_iff: dict[int, float] = {}
        if warp > 0.05:
            try:
                tau_iff = self.robot.inertia_ff_torque(q_rnea, qd_rnea, qdd_rnea)
            except Exception:
                tau_iff = {}

        max_err = 0.0
        out: dict[int, dict[str, float]] = {}
        end_q = self.active.plan.end_q
        for idx, motor_id in enumerate(self.motor_ids):
            tuning = tuning_list[idx]
            kp = float(tuning.get("kp", 0.0))
            kd = float(tuning.get("kd", 0.0))
            gscale = float(tuning.get("gravity_scale", 1.0))
            gbias = float(tuning.get("gravity_bias", 0.0))
            iff_scale = float(tuning.get("inertia_ff_scale", 0.0))

            tau_ff = gscale * tau_g_by_motor[motor_id] + gbias
            if iff_scale > 0.0 and motor_id in tau_iff:
                tau_ff += iff_scale * tau_iff[motor_id]

            if goal_modes[idx]:
                q_cmd = float(end_q[idx])
                qd_cmd = 0.0
            else:
                q_cmd = float(q_des_vec[idx])
                qd_cmd = float(qd_des_vec[idx]) * warp
                q_err = abs(self.state_by_motor[motor_id].q - q_cmd)
                if q_err > max_err:
                    max_err = q_err

            out[motor_id] = {
                "q_des": q_cmd,
                "qd_des": qd_cmd,
                "kp": kp,
                "kd": kd,
                "tau_ff": tau_ff,
            }
        return out, max_err

    def _hold_cmds(self, tau_g_by_motor: dict[int, float]) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for motor_id in self.motor_ids:
            tuning = control_params_for_motor(motor_id)
            kp = float(tuning.get("kp", 0.0))
            kd = float(tuning.get("kd", 0.0))
            gscale = float(tuning.get("gravity_scale", 1.0))
            gbias = float(tuning.get("gravity_bias", 0.0))
            tau_ff = gscale * tau_g_by_motor[motor_id] + gbias
            q_des = (
                self._hold_q[motor_id]
                if self._hold_q is not None
                else self.state_by_motor[motor_id].q
            )
            out[motor_id] = {
                "q_des": q_des, "qd_des": 0.0,
                "kp": kp, "kd": kd, "tau_ff": tau_ff,
            }
        return out

    def _compute_warp(self, max_err: float) -> float:
        lo, hi = self.warp_q_lo_rad, self.warp_q_hi_rad
        if hi <= lo or max_err <= lo:
            return 1.0
        if max_err >= hi:
            return 0.0
        return 1.0 - (max_err - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _warn_throttle(self, key: str, msg: str, interval_s: float = 2.0) -> None:
        now_s = self._now_s()
        if now_s - self._warn_times.get(key, float("-inf")) >= interval_s:
            self._warn_times[key] = now_s
            self.get_logger().warn(msg)

    def _publish(self, cmd_values: dict[int, dict[str, float]]) -> None:
        stamp = self.get_clock().now().to_msg()
        msg = MotorCMDArray()
        msg.stamp = stamp
        commands = []
        for motor_id in sorted(cmd_values.keys()):
            v = cmd_values[motor_id]
            kp = max(0.0, min(self.kp_max, v["kp"]))
            kd = max(0.0, min(self.kd_max, v["kd"]))
            tau_limit = self.tau_limit_by_motor.get(motor_id, float("inf"))
            tau_ff_raw = v["tau_ff"]
            tau_ff = tau_ff_raw
            if not self.unlimited_tau and math.isfinite(tau_limit) and tau_limit > 0:
                tau_ff = max(-tau_limit, min(tau_limit, tau_ff_raw))
                if abs(tau_ff - tau_ff_raw) > 1e-4:
                    self._warn_throttle(
                        f"tau_clamp_{motor_id}",
                        f"[SAFETY] motor {motor_id} tau_ff clamped: "
                        f"{tau_ff_raw:.3f} → {tau_ff:.3f} Nm (limit=±{tau_limit:.1f})",
                    )
            for name, val in (("q_des", v["q_des"]), ("qd_des", v["qd_des"]), ("tau_ff", tau_ff)):
                if not math.isfinite(val):
                    self.get_logger().warn(
                        f"[SAFETY] NaN/Inf in {name} for motor {motor_id} — dropping publish"
                    )
                    return
            cmd = MotorCMD()
            cmd.stamp = stamp
            cmd.motor_id = int(motor_id)
            q_des_raw = v["q_des"]
            q_des = max(self.q_min_by_motor[motor_id],
                        min(self.q_max_by_motor[motor_id], q_des_raw))
            if abs(q_des - q_des_raw) > 1e-4:
                self._warn_throttle(
                    f"q_clamp_{motor_id}",
                    f"[SAFETY] motor {motor_id} q_des clamped: {q_des_raw:.4f} → {q_des:.4f} rad",
                )
            cmd.q_des = float(q_des)
            cmd.qd_des = float(v["qd_des"])
            cmd.kp = float(kp)
            cmd.kd = float(kd)
            cmd.tau_ff = float(tau_ff)
            commands.append(cmd)
        msg.commands = commands
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_fresh(self, now_s: float) -> bool:
        for sample in self.state_by_motor.values():
            if (
                not math.isfinite(sample.last_seen_s)
                or (now_s - sample.last_seen_s) > self.state_timeout_s
            ):
                return False
        return True

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = PlanNode()
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
