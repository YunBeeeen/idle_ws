"""Standalone MuJoCo preview for dataset IK free-motion waypoints.

This script does not publish ROS topics and does not run MuJoCo physics.  It
only generates IK waypoints with the same safety filters used for dataset
free-motion and mirrors the planned joint positions into MuJoCo for visual
inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from ament_index_python.packages import get_package_share_directory
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP, DEFAULT_TAU_LIMIT_BY_MOTOR
from idle_common.paths import resolve_share_file

from phy.collision import CollisionChecker
from phy.gravity import GravityCompensator
from phy.ik import IKConfig, IKPolicyConfig, IKSolver
from phy.robot_model import RobotModel
from phy.traj import plan_quintic, sample_quintic
from sim.viewer_node import load_model_with_workaround


def _parse_json_object(text: str, field_name: str) -> dict[str, object]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON object: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return raw


def _parse_center_q(text: str, motor_ids: tuple[int, ...]) -> dict[int, float]:
    out = {motor_id: 0.0 for motor_id in motor_ids}
    raw = _parse_json_object(text, "center-q-json")
    for key, value in raw.items():
        motor_id = int(key)
        if motor_id in out:
            out[motor_id] = float(value)
    return out


def _parse_float_triplet(text: str, field_name: str) -> np.ndarray:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON list: {exc}") from exc
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{field_name} must be [x, y, z]")
    return np.asarray([float(raw[0]), float(raw[1]), float(raw[2])], dtype=float)


def _parse_int_list(text: str, field_name: str) -> list[int]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON list: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a JSON list")
    return [int(item) for item in raw]


def _sign_crosses_zero(a: float, b: float, eps: float = 1.0e-4) -> bool:
    return abs(a) > eps and abs(b) > eps and ((a > 0.0 and b < 0.0) or (a < 0.0 and b > 0.0))


def _joint_limits_by_motor(robot: RobotModel) -> tuple[dict[int, float], dict[int, float]]:
    lower, upper = robot.joint_limits()
    lower_by_motor = {motor_id: float(lower[idx]) for idx, motor_id in enumerate(robot.ordered_motor_ids)}
    upper_by_motor = {motor_id: float(upper[idx]) for idx, motor_id in enumerate(robot.ordered_motor_ids)}
    return lower_by_motor, upper_by_motor


def _validate_q(
    q_by_motor: dict[int, float],
    *,
    motor_ids: tuple[int, ...],
    lower_by_motor: dict[int, float],
    upper_by_motor: dict[int, float],
    collision: CollisionChecker | None,
    gravity: GravityCompensator,
    max_abs_joint_rad: float,
    check_gravity_load: bool,
    label: str,
) -> bool:
    for motor_id in motor_ids:
        q = float(q_by_motor[motor_id])
        if not math.isfinite(q) or abs(q) > float(max_abs_joint_rad) + 1.0e-6:
            return False
        if q < lower_by_motor[motor_id] - 1.0e-6 or q > upper_by_motor[motor_id] + 1.0e-6:
            return False
    if collision is not None and collision.check(q_by_motor):
        return False
    if check_gravity_load:
        tau_g = gravity.compute_gravity_by_motor(q_by_motor)
        for motor_id, limit in DEFAULT_TAU_LIMIT_BY_MOTOR.items():
            if motor_id in tau_g and abs(float(tau_g[motor_id])) > float(limit):
                return False
    return True


def _sample_segment(start: dict[int, float], end: dict[int, float], motor_ids: tuple[int, ...], count: int):
    for idx in range(max(2, int(count))):
        u = idx / max(int(count) - 1, 1)
        alpha = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        yield {
            motor_id: float(start[motor_id]) + (float(end[motor_id]) - float(start[motor_id])) * alpha
            for motor_id in motor_ids
        }


def _segment_safe(
    start: dict[int, float],
    end: dict[int, float],
    *,
    motor_ids: tuple[int, ...],
    lower_by_motor: dict[int, float],
    upper_by_motor: dict[int, float],
    collision: CollisionChecker | None,
    gravity: GravityCompensator,
    max_abs_joint_rad: float,
    check_gravity_load: bool,
    samples: int,
) -> bool:
    for q in _sample_segment(start, end, motor_ids, samples):
        if not _validate_q(
            q,
            motor_ids=motor_ids,
            lower_by_motor=lower_by_motor,
            upper_by_motor=upper_by_motor,
            collision=collision,
            gravity=gravity,
            max_abs_joint_rad=max_abs_joint_rad,
            check_gravity_load=check_gravity_load,
            label="segment_sample",
        ):
            return False
    return True


def _ordered(q_by_motor: dict[int, float], motor_ids: tuple[int, ...]) -> np.ndarray:
    return np.asarray([q_by_motor[motor_id] for motor_id in motor_ids], dtype=float)


def _full_with_controlled(
    base: dict[int, float],
    controlled_motor_ids: tuple[int, ...],
    q_controlled: np.ndarray,
) -> dict[int, float]:
    out = dict(base)
    for idx, motor_id in enumerate(controlled_motor_ids):
        out[motor_id] = float(q_controlled[idx])
    return out


def _insert_zero_crossings(
    waypoints: list[dict[int, float]],
    motor_ids: tuple[int, ...],
) -> list[dict[int, float]]:
    if not waypoints:
        return []
    out = [waypoints[0]]
    for target in waypoints[1:]:
        start = out[-1]
        crossing = [
            motor_id
            for motor_id in motor_ids
            if _sign_crosses_zero(float(start[motor_id]), float(target[motor_id]))
        ]
        if crossing:
            mid = {}
            for motor_id in motor_ids:
                mid[motor_id] = 0.0 if motor_id in crossing else 0.5 * (start[motor_id] + target[motor_id])
            out.append(mid)
            print(f"[preview] inserted zero-crossing waypoint for motors={crossing}")
        out.append(target)
    return out


def _qpos_indices(model: mujoco.MjModel) -> dict[int, int]:
    qpos: dict[int, int] = {}
    for motor_id, joint_name in DEFAULT_MOTOR_JOINT_MAP.items():
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if jid < 0:
            raise ValueError(f"joint not found in MuJoCo model: {joint_name}")
        qpos[int(motor_id)] = int(model.jnt_qposadr[jid])
    return qpos


def _apply_q(model: mujoco.MjModel, data: mujoco.MjData, qpos_idx: dict[int, int], q_by_motor: dict[int, float]) -> None:
    for motor_id, q_idx in qpos_idx.items():
        data.qpos[q_idx] = float(q_by_motor.get(motor_id, 0.0))
    mujoco.mj_forward(model, data)


def _generate_waypoints(args: argparse.Namespace) -> tuple[list[dict[int, float]], tuple[int, ...]]:
    motor_map = dict(DEFAULT_MOTOR_JOINT_MAP)
    urdf_path = resolve_share_file("sim", "urdf/robot.urdf", args.urdf)
    srdf_path = resolve_share_file("sim", "srdf/robot.srdf", args.srdf)
    robot = RobotModel(urdf_path, motor_map)
    gravity = GravityCompensator(urdf_path, motor_map)
    motor_ids = robot.ordered_motor_ids
    lower_by_motor, upper_by_motor = _joint_limits_by_motor(robot)

    collision: CollisionChecker | None = None
    if not args.no_self_collision_check:
        sim_share_parent = str(Path(get_package_share_directory("sim")).parent)
        collision = CollisionChecker(robot, srdf_path=srdf_path, package_dirs=[sim_share_parent])

    controlled_motor_ids = tuple(
        motor_id for motor_id in _parse_int_list(args.controlled_motor_ids_json, "controlled-motor-ids-json")
        if motor_id in motor_ids
    )
    if not controlled_motor_ids:
        raise ValueError("controlled-motor-ids-json has no valid motor ids")
    controlled_joint_names = tuple(motor_map[motor_id] for motor_id in controlled_motor_ids)

    ik = IKSolver(
        urdf_path,
        IKConfig(
            target_frame=args.target_frame,
            target_offset=tuple(_parse_float_triplet(args.target_offset_xyz, "target-offset-xyz")),
            controlled_joints=controlled_joint_names,
            max_iterations=int(args.ik_max_iterations),
            tolerance=float(args.ik_tolerance),
            damping=float(args.ik_damping),
            step_scale=float(args.ik_step_scale),
        ),
    )
    policy = IKPolicyConfig(
        max_ik_residual_accept_m=float(args.max_ik_residual_accept_m),
        ik_random_restarts=int(args.ik_random_restarts),
        ik_seed_default_span=float(args.ik_seed_default_span),
        max_joint_jump_rad=float(args.max_joint_jump_rad),
        use_heuristic_seed=True,
    )

    center_q = _parse_center_q(args.center_q_json, motor_ids)
    center_xyz = np.asarray(robot.forward_kinematics(center_q, args.target_frame).translation, dtype=float)
    center_xyz = center_xyz + _parse_float_triplet(args.workspace_bias_xyz, "workspace-bias-xyz")
    delta_xyz = _parse_float_triplet(args.workspace_delta_xyz, "workspace-delta-xyz")
    rng = np.random.default_rng(int(args.seed))

    start_q = {motor_id: 0.0 for motor_id in motor_ids}
    waypoints = [start_q]
    if not _validate_q(
        center_q,
        motor_ids=motor_ids,
        lower_by_motor=lower_by_motor,
        upper_by_motor=upper_by_motor,
        collision=collision,
        gravity=gravity,
        max_abs_joint_rad=float(args.max_abs_joint_rad),
        check_gravity_load=not args.no_gravity_load_check,
        label="center_q",
    ):
        raise ValueError("center-q-json is not safe under the current joint/collision/load checks")
    if not _segment_safe(
        start_q,
        center_q,
        motor_ids=motor_ids,
        lower_by_motor=lower_by_motor,
        upper_by_motor=upper_by_motor,
        collision=collision,
        gravity=gravity,
        max_abs_joint_rad=float(args.max_abs_joint_rad),
        check_gravity_load=not args.no_gravity_load_check,
        samples=int(args.segment_check_samples),
    ):
        raise ValueError("zero to center-q-json segment is not safe")
    waypoints.append(dict(center_q))
    current = dict(center_q)
    accepted = 0
    attempts = 0
    while accepted < int(args.waypoint_count) and attempts < int(args.max_attempts):
        attempts += 1
        goal_xyz = center_xyz + rng.uniform(-delta_xyz, delta_xyz)
        q_ref = _ordered(current, controlled_motor_ids)
        res = ik.solve_with_policy(goal_xyz, q_ref=q_ref, q_measured=q_ref, policy=policy, rng=rng)
        if not res.accepted or res.q_goal is None:
            continue
        candidate = _full_with_controlled(current, controlled_motor_ids, res.q_goal)
        if not _validate_q(
            candidate,
            motor_ids=motor_ids,
            lower_by_motor=lower_by_motor,
            upper_by_motor=upper_by_motor,
            collision=collision,
            gravity=gravity,
            max_abs_joint_rad=float(args.max_abs_joint_rad),
            check_gravity_load=not args.no_gravity_load_check,
            label=f"candidate_{accepted + 1}",
        ):
            continue
        step_norm = float(np.linalg.norm(_ordered(candidate, controlled_motor_ids) - _ordered(current, controlled_motor_ids)))
        if step_norm < float(args.min_joint_step_norm_rad):
            continue
        if not _segment_safe(
            current,
            candidate,
            motor_ids=motor_ids,
            lower_by_motor=lower_by_motor,
            upper_by_motor=upper_by_motor,
            collision=collision,
            gravity=gravity,
            max_abs_joint_rad=float(args.max_abs_joint_rad),
            check_gravity_load=not args.no_gravity_load_check,
            samples=int(args.segment_check_samples),
        ):
            continue
        accepted += 1
        waypoints.append(candidate)
        current = candidate

    if accepted < int(args.waypoint_count):
        raise ValueError(f"generated only {accepted}/{args.waypoint_count} waypoints after {attempts} attempts")
    print(f"[preview] generated {accepted} IK waypoints after {attempts} attempts")
    return _insert_zero_crossings(waypoints, motor_ids), motor_ids


def _preview(args: argparse.Namespace, waypoints: list[dict[int, float]], motor_ids: tuple[int, ...]) -> None:
    if args.software_gl:
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
        os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
        print("[preview] software GL fallback enabled")

    model_xml = resolve_share_file("sim", "robot.xml", args.model_xml)
    model, _used_workaround = load_model_with_workaround(str(model_xml))
    data = mujoco.MjData(model)
    qpos_idx = _qpos_indices(model)
    viewer = None
    if not args.no_viewer:
        import mujoco.viewer as mj_viewer

        viewer = mj_viewer.launch_passive(
            model,
            data,
            show_left_ui=not args.hide_left_ui,
            show_right_ui=not args.hide_right_ui,
        )

    dt = 1.0 / max(float(args.viewer_sync_hz), 1.0)
    q_zero = {motor_id: 0.0 for motor_id in motor_ids}
    _apply_q(model, data, qpos_idx, q_zero)
    if viewer is not None:
        viewer.sync()

    try:
        for idx in range(len(waypoints) - 1):
            start = waypoints[idx]
            end = waypoints[idx + 1]
            q0 = _ordered(start, motor_ids)
            q1 = _ordered(end, motor_ids)
            duration = float(args.initial_duration_s) if idx == 0 else float(args.segment_duration_s)
            plan = plan_quintic(
                q0,
                q1,
                np.zeros_like(q0),
                np.zeros_like(q0),
                np.full_like(q0, float(args.v_max)),
                np.full_like(q0, float(args.a_max)),
                min_duration=max(0.5, duration),
            )
            print(f"[preview] segment {idx + 1}/{len(waypoints) - 1} duration={plan.duration:.2f}s")
            t0 = time.monotonic()
            next_sync_s = t0
            while True:
                now_s = time.monotonic()
                elapsed = now_s - t0
                q_vec, _qd_vec, done = sample_quintic(plan, elapsed)
                q = {motor_id: float(q_vec[j]) for j, motor_id in enumerate(motor_ids)}
                _apply_q(model, data, qpos_idx, q)
                if viewer is not None:
                    viewer.sync()
                    if not viewer.is_running():
                        return
                if done:
                    break
                next_sync_s += dt
                time.sleep(max(0.0, next_sync_s - time.monotonic()))
        print("[preview] complete; holding final pose. Ctrl-C to exit.")
        while viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(dt)
    finally:
        if viewer is not None:
            viewer.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-xml", default="")
    parser.add_argument("--urdf", default="")
    parser.add_argument("--srdf", default="")
    parser.add_argument("--target-frame", default="gripper")
    parser.add_argument("--target-offset-xyz", default="[0.0, 0.0, 0.0]")
    parser.add_argument("--controlled-motor-ids-json", default="[1, 2, 3, 4, 5, 6]")
    parser.add_argument("--center-q-json", default='{"1": 0.0, "2": -0.70, "3": -1.20, "4": 0.0, "5": 0.0, "6": 0.0}')
    parser.add_argument("--workspace-delta-xyz", default="[0.06, 0.08, 0.05]")
    parser.add_argument("--workspace-bias-xyz", default="[0.0, 0.0, 0.0]")
    parser.add_argument("--waypoint-count", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--initial-duration-s", type=float, default=10.0)
    parser.add_argument("--segment-duration-s", type=float, default=8.0)
    parser.add_argument("--v-max", type=float, default=0.35)
    parser.add_argument("--a-max", type=float, default=0.70)
    parser.add_argument("--max-abs-joint-rad", type=float, default=math.pi)
    parser.add_argument("--segment-check-samples", type=int, default=16)
    parser.add_argument("--min-joint-step-norm-rad", type=float, default=0.08)
    parser.add_argument("--max-ik-residual-accept-m", type=float, default=0.010)
    parser.add_argument("--ik-random-restarts", type=int, default=32)
    parser.add_argument("--ik-seed-default-span", type=float, default=math.pi)
    parser.add_argument("--max-joint-jump-rad", type=float, default=0.75)
    parser.add_argument("--ik-max-iterations", type=int, default=220)
    parser.add_argument("--ik-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--ik-damping", type=float, default=1.0e-6)
    parser.add_argument("--ik-step-scale", type=float, default=1.0)
    parser.add_argument("--viewer-sync-hz", type=float, default=60.0)
    parser.add_argument("--software-gl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-self-collision-check", action="store_true")
    parser.add_argument("--no-gravity-load-check", action="store_true")
    args = parser.parse_args(argv)

    waypoints, motor_ids = _generate_waypoints(args)
    _preview(args, waypoints, motor_ids)


if __name__ == "__main__":
    main()
