"""Lightweight MuJoCo viewer for random no-contact joint motion preview.

This tool bypasses ROS topics and the MIT/physics controller path. It directly
sets MuJoCo qpos with the same smooth 5th-order interpolation used by the
hardware motion node, so it is useful for visually checking whether random
waypoints look reasonable before running the heavier ROS sim or the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP
from idle_common.paths import resolve_share_file

from sim.viewer_node import load_model_with_workaround


def _smoothstep5(u: float) -> tuple[float, float]:
    u_clamped = max(0.0, min(1.0, float(u)))
    alpha = 10.0 * u_clamped**3 - 15.0 * u_clamped**4 + 6.0 * u_clamped**5
    dalpha_du = 30.0 * u_clamped**2 - 60.0 * u_clamped**3 + 30.0 * u_clamped**4
    return alpha, dalpha_du


def _parse_motor_ids(text: str) -> list[int]:
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("--motor-ids-json must be a JSON list")
    motor_ids = [int(item) for item in raw]
    if not motor_ids:
        raise ValueError("--motor-ids-json must contain at least one motor id")
    return motor_ids


def _parse_ranges(text: str, motor_ids: list[int]) -> dict[int, tuple[float, float]]:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("--range-json must be a JSON object")
    ranges: dict[int, tuple[float, float]] = {}
    for motor_id in motor_ids:
        value = raw.get(str(motor_id), raw.get(motor_id))
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"--range-json motor {motor_id} must be [lower, upper]")
        lower = float(value[0])
        upper = float(value[1])
        if lower >= upper:
            raise ValueError(f"--range-json motor {motor_id}: lower must be < upper")
        ranges[motor_id] = (lower, upper)
    return ranges


def _parse_body_name_set(text: str) -> set[str]:
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("--floor-exclude-body-names-json must be a JSON list")
    return {str(item).strip() for item in raw if str(item).strip()}


def _parse_fixed_q(text: str) -> dict[int, float]:
    raw = json.loads(text) if text.strip() else {}
    if not isinstance(raw, dict):
        raise ValueError("--fixed-q-json must be a JSON object")
    fixed: dict[int, float] = {}
    for key, value in raw.items():
        motor_id = int(key)
        if motor_id not in DEFAULT_MOTOR_JOINT_MAP:
            raise ValueError(f"--fixed-q-json motor {motor_id} is not in the 6-DOF motor map")
        fixed[motor_id] = float(value)
    return fixed


def _parse_waypoints(raw: object, field_name: str) -> list[dict[int, float]]:
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a JSON list")
    waypoints: list[dict[int, float]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{idx}] must be a JSON object")
        waypoint: dict[int, float] = {}
        for key, value in item.items():
            motor_id = int(key)
            if motor_id not in DEFAULT_MOTOR_JOINT_MAP:
                raise ValueError(f"{field_name}[{idx}] motor {motor_id} is not in the 6-DOF motor map")
            waypoint[motor_id] = float(value)
        if waypoint:
            waypoints.append(waypoint)
    if not waypoints:
        raise ValueError(f"{field_name} must contain at least one non-empty waypoint")
    return waypoints


def _load_waypoints_json(waypoints_json: str, waypoints_path: str) -> list[dict[int, float]]:
    if waypoints_json.strip():
        return _parse_waypoints(json.loads(waypoints_json), "--waypoints-json")
    if waypoints_path.strip():
        path = Path(waypoints_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        with path.open("r", encoding="utf-8") as waypoint_file:
            return _parse_waypoints(json.load(waypoint_file), f"--waypoints-path={path}")
    return []


def _qpos_and_qvel_indices(model: mujoco.MjModel) -> tuple[dict[int, int], dict[int, int]]:
    qpos: dict[int, int] = {}
    qvel: dict[int, int] = {}
    for motor_id, joint_name in sorted(DEFAULT_MOTOR_JOINT_MAP.items()):
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if jid < 0:
            raise ValueError(f"joint not found in MuJoCo model: {joint_name} (motor_id={motor_id})")
        qpos[motor_id] = int(model.jnt_qposadr[jid])
        qvel[motor_id] = int(model.jnt_dofadr[jid])
    return qpos, qvel


def _sign_crosses_zero(start_q: float, end_q: float, eps: float) -> bool:
    if abs(start_q) <= eps or abs(end_q) <= eps:
        return False
    return (start_q > 0.0 and end_q < 0.0) or (start_q < 0.0 and end_q > 0.0)


def _insert_zero_crossing_segments(
    segments: list[dict[str, object]],
    motor_ids: list[int],
    eps: float,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for segment in segments:
        start = dict(segment["start"])
        end = dict(segment["end"])
        duration_s = float(segment["duration_s"])
        label = str(segment["label"])
        crossing = [
            motor_id
            for motor_id in motor_ids
            if _sign_crosses_zero(float(start[motor_id]), float(end[motor_id]), eps)
        ]
        if not crossing:
            out.append(segment)
            continue
        midpoint = dict(start)
        for motor_id in motor_ids:
            if motor_id in crossing:
                midpoint[motor_id] = 0.0
            else:
                midpoint[motor_id] = 0.5 * (float(start[motor_id]) + float(end[motor_id]))
        half_duration_s = max(0.5 * duration_s, 0.5)
        print(f"[preview] {label}: zero-crossing split for motors {crossing}", flush=True)
        out.append(
            {
                "start": start,
                "end": midpoint,
                "duration_s": half_duration_s,
                "label": f"{label}_to_zero",
            }
        )
        out.append(
            {
                "start": midpoint,
                "end": end,
                "duration_s": half_duration_s,
                "label": f"{label}_from_zero",
            }
        )
    return out


def _generate_waypoints(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_idx_by_motor: Mapping[int, int],
    qvel_idx_by_motor: Mapping[int, int],
    motor_ids: list[int],
    ranges: Mapping[int, tuple[float, float]],
    count: int,
    seed: int,
    min_step_norm_rad: float,
    max_step_norm_rad: float,
    max_abs_joint_rad: float,
    max_attempts: int,
    floor_filter_enabled: bool,
    min_floor_clearance_m: float,
    floor_samples_per_candidate: int,
    floor_excluded_body_names: set[str],
    fixed_q: Mapping[int, float],
) -> list[dict[int, float]]:
    rng = random.Random(int(seed))
    waypoints: list[dict[int, float]] = []
    previous = {motor_id: 0.0 for motor_id in DEFAULT_MOTOR_JOINT_MAP}
    attempts = 0
    while len(waypoints) < count and attempts < max_attempts:
        attempts += 1
        candidate = {
            motor_id: rng.uniform(float(ranges[motor_id][0]), float(ranges[motor_id][1]))
            for motor_id in motor_ids
        }
        if any(abs(q_rad) > max_abs_joint_rad for q_rad in candidate.values()):
            continue
        full_q = dict(previous)
        full_q.update(fixed_q)
        full_q.update(candidate)
        if waypoints:
            step_norm = math.sqrt(
                sum(
                    (candidate[motor_id] - previous.get(motor_id, 0.0)) ** 2
                    for motor_id in motor_ids
                )
            )
            if step_norm < min_step_norm_rad or step_norm > max_step_norm_rad:
                continue
        if floor_filter_enabled and not _segment_floor_clearance_ok(
            model,
            data,
            qpos_idx_by_motor,
            qvel_idx_by_motor,
            previous,
            full_q,
            min_floor_clearance_m,
            max(2, int(floor_samples_per_candidate)),
            floor_excluded_body_names,
        ):
            continue
        waypoints.append(candidate)
        previous = full_q
    if len(waypoints) < count:
        raise ValueError(f"generated only {len(waypoints)}/{count} waypoints after {attempts} attempts")
    print(f"[preview] generated {len(waypoints)} waypoints seed={seed} ranges={dict(ranges)}", flush=True)
    return waypoints


def _generate_sweep_waypoints(
    sweep_motor_id: int,
    high_rad: float,
    low_rad: float,
    cycles: int,
    max_abs_joint_rad: float,
) -> list[dict[int, float]]:
    if sweep_motor_id not in DEFAULT_MOTOR_JOINT_MAP:
        raise ValueError(f"--sweep-motor-id must be one of {sorted(DEFAULT_MOTOR_JOINT_MAP)}")
    if abs(high_rad) > max_abs_joint_rad or abs(low_rad) > max_abs_joint_rad:
        raise ValueError(
            f"sweep target exceeds +/-{max_abs_joint_rad:.3f} rad: "
            f"high={high_rad:.3f}, low={low_rad:.3f}"
        )
    values = [float(high_rad)]
    for _idx in range(max(1, int(cycles))):
        values.extend([float(low_rad), float(high_rad)])
    print(
        f"[preview] generated j{sweep_motor_id} sweep waypoints: "
        + " -> ".join(f"{value:+.3f}" for value in values),
        flush=True,
    )
    return [{int(sweep_motor_id): value} for value in values]


def _build_segments(
    waypoints: list[dict[int, float]],
    motor_ids: list[int],
    fixed_q: Mapping[int, float],
    initial_duration_s: float,
    segment_duration_s: float,
    zero_crossing_eps_rad: float,
    start_at_first_waypoint: bool,
) -> list[dict[str, object]]:
    current = {motor_id: 0.0 for motor_id in DEFAULT_MOTOR_JOINT_MAP}
    waypoint_iter = list(waypoints)
    if start_at_first_waypoint and waypoint_iter:
        current.update(fixed_q)
        current.update(waypoint_iter[0])
        waypoint_iter = waypoint_iter[1:]
    segments: list[dict[str, object]] = []
    for idx, waypoint in enumerate(waypoint_iter):
        target = dict(current)
        target.update(fixed_q)
        target.update(waypoint)
        segments.append(
            {
                "start": dict(current),
                "end": target,
                "duration_s": initial_duration_s if idx == 0 and not start_at_first_waypoint else segment_duration_s,
                "label": f"waypoint_{idx + 1:02d}",
            }
        )
        current = target
    return _insert_zero_crossing_segments(segments, motor_ids, zero_crossing_eps_rad)


def _floor_min_z_detail(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    excluded_body_names: set[str],
) -> tuple[float, str, str]:
    min_z = float("inf")
    min_geom_name = ""
    min_body_name = ""
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name in excluded_body_names:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        if vert_num <= 0:
            continue
        local_vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=float)
        xmat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        xpos = np.asarray(data.geom_xpos[geom_id], dtype=float)
        world_vertices = local_vertices @ xmat.T + xpos
        geom_min_z = float(np.min(world_vertices[:, 2]))
        if geom_min_z < min_z:
            min_z = geom_min_z
            min_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            min_body_name = body_name
    return min_z, min_geom_name, min_body_name


def _floor_min_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    excluded_body_names: set[str],
) -> float:
    min_z, _geom_name, _body_name = _floor_min_z_detail(model, data, excluded_body_names)
    return min_z


def _segment_floor_clearance_ok(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_idx_by_motor: Mapping[int, int],
    qvel_idx_by_motor: Mapping[int, int],
    start_q: Mapping[int, float],
    end_q: Mapping[int, float],
    min_floor_clearance_m: float,
    sample_count: int,
    excluded_body_names: set[str],
) -> bool:
    zeros = {motor_id: 0.0 for motor_id in DEFAULT_MOTOR_JOINT_MAP}
    for idx in range(sample_count):
        u = idx / max(sample_count - 1, 1)
        alpha, _dalpha_du = _smoothstep5(u)
        q_sample: dict[int, float] = {}
        for motor_id in DEFAULT_MOTOR_JOINT_MAP:
            start = float(start_q.get(motor_id, 0.0))
            end = float(end_q.get(motor_id, start))
            q_sample[motor_id] = start + (end - start) * alpha
        _apply_q(model, data, qpos_idx_by_motor, qvel_idx_by_motor, q_sample, zeros)
        if _floor_min_z(model, data, excluded_body_names) < float(min_floor_clearance_m):
            return False
    return True


def _apply_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_idx_by_motor: Mapping[int, int],
    qvel_idx_by_motor: Mapping[int, int],
    q_by_motor: Mapping[int, float],
    qd_by_motor: Mapping[int, float],
) -> None:
    for motor_id, qpos_idx in qpos_idx_by_motor.items():
        data.qpos[qpos_idx] = float(q_by_motor.get(motor_id, 0.0))
    for motor_id, qvel_idx in qvel_idx_by_motor.items():
        data.qvel[qvel_idx] = float(qd_by_motor.get(motor_id, 0.0))
    mujoco.mj_forward(model, data)


def _sample_segment(segment: Mapping[str, object], elapsed_s: float) -> tuple[dict[int, float], dict[int, float]]:
    start = segment["start"]
    end = segment["end"]
    duration_s = max(float(segment["duration_s"]), 1.0e-6)
    alpha, dalpha_du = _smoothstep5(elapsed_s / duration_s)
    q: dict[int, float] = {}
    qd: dict[int, float] = {}
    for motor_id in DEFAULT_MOTOR_JOINT_MAP:
        start_q = float(start.get(motor_id, 0.0))
        end_q = float(end.get(motor_id, start_q))
        delta = end_q - start_q
        q[motor_id] = start_q + delta * alpha
        qd[motor_id] = delta * dalpha_du / duration_s
    return q, qd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", default="", help="MuJoCo XML path. Empty uses sim/robot.xml")
    parser.add_argument("--motor-ids-json", default="[2, 3]")
    parser.add_argument("--range-json", default='{"2": [-0.78, -0.62], "3": [-1.35, -1.05]}')
    parser.add_argument("--fixed-q-json", default="{}", help="Motor id to fixed joint angle map, e.g. '{\"2\": -0.70}'")
    parser.add_argument("--waypoints-json", default="", help="Explicit waypoint list JSON. Overrides random/sweep generation.")
    parser.add_argument("--waypoints-path", default="", help="Path to explicit waypoint list JSON. Overrides random/sweep generation.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--random-waypoint-count", type=int, default=5)
    parser.add_argument("--random-min-step-norm-rad", type=float, default=0.08)
    parser.add_argument("--random-max-step-norm-rad", type=float, default=0.35)
    parser.add_argument("--random-max-attempts", type=int, default=500)
    parser.add_argument("--max-abs-joint-rad", type=float, default=math.pi)
    parser.add_argument("--initial-duration-s", type=float, default=12.0)
    parser.add_argument("--segment-duration-s", type=float, default=25.0)
    parser.add_argument("--viewer-sync-hz", type=float, default=20.0)
    parser.add_argument("--zero-crossing-eps-rad", type=float, default=1.0e-4)
    parser.add_argument("--sweep-motor-id", type=int, default=0, help="If >0, use deterministic single-joint sweep instead of random waypoints")
    parser.add_argument("--sweep-high-rad", type=float, default=-0.80)
    parser.add_argument("--sweep-low-rad", type=float, default=-1.35)
    parser.add_argument("--sweep-cycles", type=int, default=2)
    parser.add_argument(
        "--start-at-first-waypoint",
        action="store_true",
        help="Start preview from the first generated waypoint instead of animating from all-zero q.",
    )
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--software-gl", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print generated segments and exit before opening viewer")
    parser.add_argument("--disable-floor-filter", action="store_true")
    parser.add_argument("--min-floor-clearance-m", type=float, default=0.02)
    parser.add_argument("--floor-samples-per-candidate", type=int, default=8)
    parser.add_argument("--floor-exclude-body-names-json", default='["world", "joint1"]')
    parser.add_argument("--warn-floor-clearance-m", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.software_gl:
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

    motor_ids = _parse_motor_ids(args.motor_ids_json)
    fixed_q = _parse_fixed_q(args.fixed_q_json)
    floor_excluded_body_names = _parse_body_name_set(args.floor_exclude_body_names_json)
    model_xml = resolve_share_file("sim", "robot.xml", args.model_xml)
    model, _used_workaround = load_model_with_workaround(str(model_xml))
    data = mujoco.MjData(model)
    qpos_idx_by_motor, qvel_idx_by_motor = _qpos_and_qvel_indices(model)

    explicit_waypoints = _load_waypoints_json(args.waypoints_json, args.waypoints_path)
    if explicit_waypoints:
        waypoints = explicit_waypoints
        motor_ids = sorted({motor_id for waypoint in waypoints for motor_id in waypoint})
        print(f"[preview] loaded {len(waypoints)} explicit waypoints motors={motor_ids}", flush=True)
    elif int(args.sweep_motor_id) > 0:
        motor_ids = [int(args.sweep_motor_id)]
        waypoints = _generate_sweep_waypoints(
            int(args.sweep_motor_id),
            float(args.sweep_high_rad),
            float(args.sweep_low_rad),
            int(args.sweep_cycles),
            float(args.max_abs_joint_rad),
        )
    else:
        ranges = _parse_ranges(args.range_json, motor_ids)
        waypoints = _generate_waypoints(
            model=model,
            data=data,
            qpos_idx_by_motor=qpos_idx_by_motor,
            qvel_idx_by_motor=qvel_idx_by_motor,
            motor_ids=motor_ids,
            ranges=ranges,
            count=max(1, int(args.random_waypoint_count)),
            seed=int(args.random_seed),
            min_step_norm_rad=float(args.random_min_step_norm_rad),
            max_step_norm_rad=float(args.random_max_step_norm_rad),
            max_abs_joint_rad=float(args.max_abs_joint_rad),
            max_attempts=max(1, int(args.random_max_attempts)),
            floor_filter_enabled=not bool(args.disable_floor_filter),
            min_floor_clearance_m=float(args.min_floor_clearance_m),
            floor_samples_per_candidate=max(2, int(args.floor_samples_per_candidate)),
            floor_excluded_body_names=floor_excluded_body_names,
            fixed_q=fixed_q,
        )
    segments = _build_segments(
        waypoints,
        motor_ids,
        fixed_q,
        max(0.5, float(args.initial_duration_s)),
        max(0.5, float(args.segment_duration_s)),
        float(args.zero_crossing_eps_rad),
        bool(args.start_at_first_waypoint),
    )
    if not bool(args.disable_floor_filter):
        for segment in segments:
            if not _segment_floor_clearance_ok(
                model,
                data,
                qpos_idx_by_motor,
                qvel_idx_by_motor,
                segment["start"],
                segment["end"],
                float(args.min_floor_clearance_m),
                max(2, int(args.floor_samples_per_candidate)),
                floor_excluded_body_names,
            ):
                print(
                    f"[preview][warn] segment {segment['label']} does not satisfy "
                    f"min_floor_clearance_m={float(args.min_floor_clearance_m):.4f}. "
                    "Use a narrower j2/j3 range or --disable-floor-filter for visual-only replay.",
                    flush=True,
                )
    print("[preview] segments:", flush=True)
    for idx, segment in enumerate(segments, start=1):
        print(
            f"  {idx:02d}. {segment['label']} "
            f"duration={float(segment['duration_s']):.2f}s end={segment['end']}",
            flush=True,
        )
    if not segments:
        print("[preview] no motion segments to replay", flush=True)
        return
    if args.dry_run:
        return

    import mujoco.viewer as mujoco_viewer

    viewer = mujoco_viewer.launch_passive(model, data)
    period_s = 1.0 / max(float(args.viewer_sync_hz), 1.0)
    segment_idx = 0
    segment_start_s = time.monotonic()
    last_warning_s = 0.0
    last_floor_warning_segment_idx = -1
    try:
        while viewer.is_running():
            now_s = time.monotonic()
            segment = segments[segment_idx]
            elapsed_s = now_s - segment_start_s
            if elapsed_s > float(segment["duration_s"]):
                segment_idx += 1
                if segment_idx >= len(segments):
                    if not args.repeat:
                        segment_idx = len(segments) - 1
                        segment_start_s = now_s - float(segments[segment_idx]["duration_s"])
                    else:
                        segment_idx = 0
                        segment_start_s = now_s
                else:
                    segment_start_s += float(segment["duration_s"])
                print(
                    f"[preview] segment {segment_idx + 1}/{len(segments)}: "
                    f"{segments[segment_idx]['label']}",
                    flush=True,
                )
                segment = segments[segment_idx]
                elapsed_s = now_s - segment_start_s

            q, qd = _sample_segment(segment, elapsed_s)
            _apply_q(model, data, qpos_idx_by_motor, qvel_idx_by_motor, q, qd)
            if (now_s - last_warning_s) > 1.0:
                min_z, geom_name, body_name = _floor_min_z_detail(model, data, floor_excluded_body_names)
                if min_z < float(args.warn_floor_clearance_m):
                    if segment_idx != last_floor_warning_segment_idx:
                        print(
                            f"[preview][warn] floor clearance low: min_z={min_z:.4f}m "
                            f"< {float(args.warn_floor_clearance_m):.4f}m "
                            f"body={body_name} geom={geom_name}",
                            flush=True,
                        )
                        last_floor_warning_segment_idx = segment_idx
                last_warning_s = now_s
            viewer.sync()
            time.sleep(period_s)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
