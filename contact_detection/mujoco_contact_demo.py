"""Run a MuJoCo contact-detection demo with injected or mouse-applied disturbance.

논문/시연 흐름에서 이 파일은 실제 로봇으로 가기 전 "학습된 모델이 MuJoCo에서
외란에 반응하는지 눈으로 확인"하는 단계다.

두 가지 모드가 있다:
- scripted mode: 코드가 정해진 시간에 disturbance torque를 넣고 P(contact)를 기록한다.
- --mouse mode: 사용자가 MuJoCo viewer에서 Ctrl+drag로 body force를 직접 걸어본다.

주의:
- mouse force나 tau_ext는 시연/라벨 확인용이지 모델 입력이 아니다.
- 모델 입력은 학습과 동일하게 [q, qdot, e_q, tau_cmd, optional delta]만 사용한다.
"""

from __future__ import annotations

# argparse: demo 실행 옵션(--mouse, --manual, --duration 등)을 받는다.
import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

# mujoco: XML model load, physics step, viewer mouse perturb, HUD overlay에 사용한다.
import mujoco
# NumPy: q/qdot/tau/feature row를 vector 연산으로 다루는 기본 array 라이브러리.
import numpy as np

from models import GRUDetector
from utils import (
    StandardScaler,
    apply_stage_config,
    ensure_output_dirs,
    load_config,
    load_json,
    output_root,
    robot_joint_names,
    select_torch_device,
    sigmoid,
)


DEFAULT_ROS_SIGN_BY_MOTOR = {2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0}
@dataclass(frozen=True)
class Event:
    start: float
    end: float
    tau_ros: np.ndarray


@dataclass(frozen=True)
class BodyForceEvent:
    start: float
    end: float
    body_id: int
    force_world: np.ndarray


def parse_float_map_json(text: str) -> dict[int, float]:
    if not text.strip():
        return {}
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("JSON map must be an object")
    return {int(key): float(value) for key, value in raw.items()}


def resolve_model_xml(config: dict, model_xml_text: str) -> Path:
    if model_xml_text.strip():
        path = Path(model_xml_text).expanduser()
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    config_dir = Path(config["_config_dir"]).resolve()
    candidate = (config_dir.parent / "src" / "sim" / "robot.xml").resolve()
    if candidate.exists():
        return candidate
    return (config_dir.parent / "install" / "sim" / "share" / "sim" / "robot.xml").resolve()


def qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos_idx: list[int] = []
    qvel_idx: list[int] = []
    for joint_name in joint_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        qpos_idx.append(int(model.jnt_qposadr[joint_id]))
        qvel_idx.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_idx, dtype=np.int64), np.asarray(qvel_idx, dtype=np.int64)


def named_joint_ids(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    ids: list[int] = []
    for joint_name in joint_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        ids.append(joint_id)
    return np.asarray(ids, dtype=np.int64)


def actuator_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    indices: list[int] = []
    for joint_name in joint_names:
        actuator_candidates = (f"tau_{joint_name}", f"{joint_name}_motor", joint_name)
        actuator_id = -1
        actuator_name = actuator_candidates[0]
        for candidate in actuator_candidates:
            candidate_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate))
            if candidate_id >= 0:
                actuator_id = candidate_id
                actuator_name = candidate
                break
        if actuator_id < 0:
            raise ValueError(
                f"MuJoCo actuator not found for joint {joint_name}. "
                f"Tried: {', '.join(actuator_candidates)}"
            )
        indices.append(actuator_id)
    return np.asarray(indices, dtype=np.int64)


def default_events(dof: int) -> list[Event]:
    specs = [
        (2.0, 2.7, {4: 1.3}),
        (4.5, 5.2, {2: -1.4, 5: 0.8}),
        (7.0, 7.8, {4: -1.3, 5: 0.8}),
    ]
    events: list[Event] = []
    for start, end, values in specs:
        tau = np.zeros(dof, dtype=np.float64)
        for motor_id, value in values.items():
            if 1 <= motor_id <= dof:
                tau[motor_id - 1] = float(value)
        events.append(Event(float(start), float(end), tau))
    return events


def parse_events_json(text: str, dof: int) -> list[Event]:
    if not text.strip():
        return default_events(dof)
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("events JSON must be a list like [[start, end, {\"4\": 1.3}], ...]")
    events: list[Event] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError("each event must be [start, end, motor_tau_map]")
        start, end, values = item
        if not isinstance(values, dict):
            raise ValueError("event motor_tau_map must be an object")
        tau = np.zeros(dof, dtype=np.float64)
        for motor_id_text, value in values.items():
            motor_id = int(motor_id_text)
            if 1 <= motor_id <= dof:
                tau[motor_id - 1] = float(value)
        events.append(Event(float(start), float(end), tau))
    return events


def tau_ext_at(time_s: float, events: list[Event], sign: np.ndarray) -> tuple[np.ndarray, int]:
    tau_ros = np.zeros(sign.shape[0], dtype=np.float64)
    label = 0
    for event in events:
        if event.start <= time_s < event.end:
            tau_ros += event.tau_ros
            label = 1
    tau_mj = tau_ros * sign
    return tau_mj, label


def make_manual_state(initial_motor: int) -> dict:
    return {
        "motor": int(initial_motor),
        "axis": 0,
        "magnitude_scale": 1.0,
        "duration_scale": 1.0,
        "requests": [],
        "clear": False,
        "lock": threading.Lock(),
    }


def make_key_callback(state: dict, dof: int):
    def on_key(key: int) -> None:
        with state["lock"]:
            if ord("1") <= key <= ord(str(min(dof, 9))):
                state["motor"] = int(chr(key))
                print(f"[manual] selected motor {state['motor']}")
            elif key in (ord("F"), ord("f")):
                state["requests"].append(1.0)
            elif key in (ord("R"), ord("r")):
                state["requests"].append(-1.0)
            elif key in (ord("C"), ord("c")):
                state["clear"] = True
            elif key in (ord("X"), ord("x")):
                state["axis"] = 0
                print("[manual] selected world force axis X")
            elif key in (ord("Y"), ord("y")):
                state["axis"] = 1
                print("[manual] selected world force axis Y")
            elif key in (ord("Z"), ord("z")):
                state["axis"] = 2
                print("[manual] selected world force axis Z")
            elif key in (ord("="), ord("+")):
                state["magnitude_scale"] = min(5.0, float(state["magnitude_scale"]) * 1.25)
                print(f"[manual] magnitude scale x{state['magnitude_scale']:.2f}")
            elif key in (ord("-"), ord("_")):
                state["magnitude_scale"] = max(0.1, float(state["magnitude_scale"]) / 1.25)
                print(f"[manual] magnitude scale x{state['magnitude_scale']:.2f}")
            elif key in (ord("]"), ord("}")):
                state["duration_scale"] = min(5.0, float(state["duration_scale"]) * 1.25)
                print(f"[manual] duration scale x{state['duration_scale']:.2f}")
            elif key in (ord("["), ord("{")):
                state["duration_scale"] = max(0.1, float(state["duration_scale"]) / 1.25)
                print(f"[manual] duration scale x{state['duration_scale']:.2f}")

    return on_key


def consume_manual_joint_events(
    state: dict,
    sim_time: float,
    dof: int,
    magnitude: float,
    duration: float,
) -> tuple[list[Event], bool]:
    with state["lock"]:
        motor = int(state["motor"])
        magnitude_scale = float(state["magnitude_scale"])
        duration_scale = float(state["duration_scale"])
        requests = list(state["requests"])
        state["requests"].clear()
        clear = bool(state["clear"])
        state["clear"] = False

    events: list[Event] = []
    scaled_magnitude = float(magnitude) * magnitude_scale
    scaled_duration = float(duration) * duration_scale
    for direction in requests:
        tau = np.zeros(dof, dtype=np.float64)
        tau[motor - 1] = float(direction) * scaled_magnitude
        events.append(Event(sim_time, sim_time + scaled_duration, tau))
        sign_text = "+" if direction > 0.0 else "-"
        print(f"[manual] motor {motor} {sign_text}{scaled_magnitude:.3f} Nm for {scaled_duration:.3f}s")
    return events, clear


def consume_manual_body_force_events(
    state: dict,
    sim_time: float,
    body_id: int,
    magnitude: float,
    duration: float,
) -> tuple[list[BodyForceEvent], bool]:
    with state["lock"]:
        axis = int(state["axis"])
        magnitude_scale = float(state["magnitude_scale"])
        duration_scale = float(state["duration_scale"])
        requests = list(state["requests"])
        state["requests"].clear()
        clear = bool(state["clear"])
        state["clear"] = False

    events: list[BodyForceEvent] = []
    axis_name = "XYZ"[axis]
    scaled_magnitude = float(magnitude) * magnitude_scale
    scaled_duration = float(duration) * duration_scale
    for direction in requests:
        force = np.zeros(3, dtype=np.float64)
        force[axis] = float(direction) * scaled_magnitude
        events.append(BodyForceEvent(sim_time, sim_time + scaled_duration, body_id, force))
        sign_text = "+" if direction > 0.0 else "-"
        print(f"[manual] body force {sign_text}{scaled_magnitude:.3f} N along world {axis_name} for {scaled_duration:.3f}s")
    return events, clear


def update_disturbance_markers(
    viewer,
    data: mujoco.MjData,
    joint_ids: np.ndarray,
    tau_ros: np.ndarray,
    active_body_ids: list[int],
) -> None:
    """Draw red spheres in the MuJoCo viewer at currently disturbed joints/bodies."""
    if viewer is None or viewer.user_scn is None:
        return
    scene = viewer.user_scn
    scene.ngeom = 0
    active = np.flatnonzero(np.abs(tau_ros) > 1e-9)
    for idx in active:
        if scene.ngeom >= len(scene.geoms):
            break
        joint_id = int(joint_ids[idx])
        pos = np.asarray(data.xanchor[joint_id], dtype=np.float64).copy()
        size = np.asarray([0.055, 0.0, 0.0], dtype=np.float64)
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        rgba = np.asarray([1.0, 0.05, 0.02, 0.85], dtype=np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            pos,
            mat,
            rgba,
        )
        scene.ngeom += 1
    for body_id in active_body_ids:
        if scene.ngeom >= len(scene.geoms):
            break
        pos = np.asarray(data.xpos[int(body_id)], dtype=np.float64).copy()
        size = np.asarray([0.065, 0.0, 0.0], dtype=np.float64)
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        rgba = np.asarray([1.0, 0.0, 0.0, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            pos,
            mat,
            rgba,
        )
        scene.ngeom += 1


def apply_mouse_perturb_force(viewer, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, list[int]]:
    """Apply MuJoCo viewer Ctrl+drag perturbation to data.xfrc_applied."""
    if viewer is None or viewer.perturb is None:
        return False, []
    perturb = viewer.perturb
    selected = int(perturb.select)
    active = int(perturb.active) != 0 or int(perturb.active2) != 0
    if selected <= 0 or not active:
        return False, []
    mujoco.mjv_applyPerturbForce(model, data, perturb)
    return True, [selected]


def selected_perturb_body_id(viewer) -> int:
    if viewer is None or viewer.perturb is None:
        return -1
    selected = int(viewer.perturb.select)
    return selected if selected > 0 else -1


def body_wrench_to_joint_tau_ros(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    force_world: np.ndarray,
    torque_world: np.ndarray,
    qvel_idx: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    """Map a selected-body Cartesian wrench to 6-DOF joint torque order.

    MuJoCo's mouse perturbation writes a world-frame wrench to
    ``data.xfrc_applied``.  For debugging the detector it is easier to inspect
    the equivalent generalized joint torque, so we compute J^T F with
    ``mj_applyFT`` and then convert to the ROS/motor joint order used in plots.
    """

    if body_id < 0:
        return np.zeros_like(sign, dtype=np.float64)
    force = np.asarray(force_world, dtype=np.float64).reshape(3)
    torque = np.asarray(torque_world, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(force) + np.linalg.norm(torque)) <= 0.0:
        return np.zeros_like(sign, dtype=np.float64)
    qfrc = np.zeros(model.nv, dtype=np.float64)
    point = np.asarray(data.xipos[body_id], dtype=np.float64).reshape(3)
    mujoco.mj_applyFT(model, data, force, torque, point, int(body_id), qfrc)
    return qfrc[np.asarray(qvel_idx, dtype=np.int64)] * sign


def object_name(model: mujoco.MjModel, obj_type, obj_id: int) -> str:
    if obj_id < 0:
        return "none"
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return str(name) if name else f"id={obj_id}"


def make_force_hud_figure(
    body_name: str,
    force: np.ndarray,
    torque: np.ndarray,
    probability: float,
    threshold: float,
    e_norm: float,
    baseline_gamma: float | None,
    joint_tau_equiv: np.ndarray | None = None,
) -> mujoco.MjvFigure:
    """Build a MuJoCo MjvFigure overlay showing selected body force and P(contact)."""
    f_norm = float(np.linalg.norm(force))
    t_norm = float(np.linalg.norm(torque))
    tau_eq = np.zeros(6, dtype=np.float64) if joint_tau_equiv is None else np.asarray(joint_tau_equiv, dtype=np.float64)
    tau_eq_norm = float(np.linalg.norm(tau_eq))
    prob_value = 0.0 if not math.isfinite(probability) else float(probability)
    e_norm_value = 0.0 if not math.isfinite(e_norm) else float(e_norm)
    baseline_text = "n/a" if baseline_gamma is None else ("contact" if e_norm_value >= float(baseline_gamma) else "no-contact")
    fig = mujoco.MjvFigure()
    fig.title = (
        "Mouse force HUD\n"
        f"body: {body_name}\n"
        f"F[N]: [{force[0]:+.2f}, {force[1]:+.2f}, {force[2]:+.2f}]  ||F||={f_norm:.2f}\n"
        f"T[Nm]: [{torque[0]:+.2f}, {torque[1]:+.2f}, {torque[2]:+.2f}]  ||T||={t_norm:.2f}\n"
        f"equiv joint ||tau||={tau_eq_norm:.2f} Nm  max|tau|={np.max(np.abs(tau_eq)):.2f}\n"
        f"P(contact)={prob_value:.3f}  threshold={threshold:.3f}\n"
        f"||e_q||={e_norm_value:.3f}  baseline={baseline_text}"
    )
    fig.xlabel = "F/T from xfrc_applied; tau = J^T wrench in joint order"
    fig.flg_barplot = 1
    fig.flg_legend = 1
    fig.gridsize[:] = [3, 5]
    fig.range[0, :] = [0.0, 5.0]
    fig.range[1, :] = [0.0, max(1.0, f_norm, t_norm, tau_eq_norm, 10.0 * max(prob_value, threshold))]
    names_and_values = [
        ("||F|| N", f_norm, [1.0, 0.25, 0.05]),
        ("||T|| Nm", t_norm, [0.95, 0.65, 0.10]),
        ("||tau||", tau_eq_norm, [0.30, 0.75, 0.25]),
        ("P x10", 10.0 * prob_value, [0.10, 0.45, 1.0]),
        ("thr x10", 10.0 * float(threshold), [0.25, 0.25, 0.25]),
    ]
    for idx, (name, value, color) in enumerate(names_and_values):
        fig.linename[idx] = name
        fig.linepnt[idx] = 1
        fig.linedata[idx, 0] = float(value)
        fig.linergb[idx, :] = np.asarray(color, dtype=np.float32)
    return fig


def update_force_hud(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    probability: float,
    threshold: float,
    e_norm: float,
    baseline_gamma: float | None,
    joint_tau_equiv: np.ndarray | None = None,
) -> None:
    """Refresh the viewer HUD during --mouse mode."""
    if viewer is None:
        return
    viewport = viewer.viewport
    if viewport is None:
        return
    force = np.zeros(3, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    body_name = "none"
    if body_id >= 0:
        force = np.asarray(data.xfrc_applied[body_id, :3], dtype=np.float64)
        torque = np.asarray(data.xfrc_applied[body_id, 3:], dtype=np.float64)
        body_name = object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    fig = make_force_hud_figure(
        body_name,
        force,
        torque,
        probability,
        threshold,
        e_norm,
        baseline_gamma,
        joint_tau_equiv,
    )
    width = min(520, max(320, int(viewport.width * 0.38)))
    height = 230
    left = max(0, int(viewport.width) - width - 10)
    bottom = max(0, int(viewport.height) - height - 10)
    viewer.set_figures((mujoco.MjrRect(left, bottom, width, height), fig))


def slow_sine_reference(time_s: float, dof: int) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray([0.0, -0.55, -0.25, 0.10, -0.25, 0.0], dtype=np.float64)[:dof]
    amplitude = np.asarray([0.12, 0.10, 0.08, 0.06, 0.05, 0.03], dtype=np.float64)[:dof]
    phase = np.asarray([0.0, 0.6, 1.2, 1.8, 2.4, 0.3], dtype=np.float64)[:dof]
    frequency = 0.12
    omega = 2.0 * math.pi * frequency
    angle = omega * float(time_s) + phase
    q_des = center + amplitude * np.sin(angle)
    qd_des = amplitude * omega * np.cos(angle)
    return q_des, qd_des


def hold_reference(dof: int) -> tuple[np.ndarray, np.ndarray]:
    q_des = np.asarray([0.0, -0.55, -0.25, 0.10, -0.25, 0.0], dtype=np.float64)[:dof]
    qd_des = np.zeros(dof, dtype=np.float64)
    return q_des, qd_des


def reference_at(time_s: float, dof: int, trajectory: str) -> tuple[np.ndarray, np.ndarray]:
    if trajectory == "hold":
        return hold_reference(dof)
    if trajectory == "slow_sine":
        return slow_sine_reference(time_s, dof)
    raise ValueError(f"Unknown trajectory mode: {trajectory}")


def build_feature_row(
    q_ros: np.ndarray,
    qd_ros: np.ndarray,
    q_des_ros: np.ndarray,
    tau_cmd_ros: np.ndarray,
    use_delta_features: bool,
    prev_e_q: np.ndarray | None,
    prev_qd: np.ndarray | None,
    prev_tau: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one online feature row using the same order as offline training."""
    e_q = q_des_ros - q_ros
    blocks = [q_ros, qd_ros, e_q, tau_cmd_ros]
    if use_delta_features:
        if prev_e_q is None or prev_qd is None or prev_tau is None:
            blocks.extend([np.zeros_like(e_q), np.zeros_like(qd_ros), np.zeros_like(tau_cmd_ros)])
        else:
            blocks.extend([e_q - prev_e_q, qd_ros - prev_qd, tau_cmd_ros - prev_tau])
    return np.concatenate(blocks, axis=0), e_q, qd_ros.copy(), tau_cmd_ros.copy()


def fallback_feature_names(dof: int, use_delta_features: bool) -> list[str]:
    names: list[str] = []
    for prefix in ("q", "qdot", "e_q", "tau_cmd"):
        names.extend([f"{prefix}{idx + 1}" for idx in range(dof)])
    if use_delta_features:
        for prefix in ("delta_e_q", "delta_qdot", "delta_tau_cmd"):
            names.extend([f"{prefix}{idx + 1}" for idx in range(dof)])
    return names


def feature_debug_metrics(
    feature_row: np.ndarray,
    feature_scaled: np.ndarray,
    dof: int,
    use_delta_features: bool,
    feature_names: list[str],
) -> dict[str, float | int | str]:
    """Return compact norms that explain why P(contact) did or did not move."""

    row = np.asarray(feature_row, dtype=np.float64).reshape(-1)
    scaled = np.asarray(feature_scaled, dtype=np.float64).reshape(-1)
    names = feature_names if len(feature_names) == row.size else fallback_feature_names(dof, use_delta_features)
    metrics: dict[str, float | int | str] = {
        "q_norm": float(np.linalg.norm(row[0:dof])),
        "qdot_norm": float(np.linalg.norm(row[dof : 2 * dof])),
        "e_norm": float(np.linalg.norm(row[2 * dof : 3 * dof])),
        "tau_cmd_norm": float(np.linalg.norm(row[3 * dof : 4 * dof])),
        "delta_e_norm": 0.0,
        "delta_qdot_norm": 0.0,
        "delta_tau_cmd_norm": 0.0,
    }
    if use_delta_features and row.size >= 7 * dof:
        metrics["delta_e_norm"] = float(np.linalg.norm(row[4 * dof : 5 * dof]))
        metrics["delta_qdot_norm"] = float(np.linalg.norm(row[5 * dof : 6 * dof]))
        metrics["delta_tau_cmd_norm"] = float(np.linalg.norm(row[6 * dof : 7 * dof]))
    if scaled.size:
        top_idx = int(np.argmax(np.abs(scaled)))
        metrics["scaled_feature_max_abs"] = float(np.abs(scaled[top_idx]))
        metrics["scaled_feature_top_index"] = int(top_idx)
        metrics["scaled_feature_top_name"] = str(names[top_idx]) if top_idx < len(names) else f"feature_{top_idx}"
        metrics["scaled_feature_top_value"] = float(scaled[top_idx])
    else:
        metrics["scaled_feature_max_abs"] = 0.0
        metrics["scaled_feature_top_index"] = -1
        metrics["scaled_feature_top_name"] = "none"
        metrics["scaled_feature_top_value"] = 0.0
    return metrics


def save_demo_plot(path: Path, history: dict[str, list[float]]) -> None:
    """Save the time-series demo plot shown in the paper/debug workflow."""
    # Matplotlib import를 함수 안에서 하는 이유:
    # GUI viewer만 쓰는 경우에도 plotting dependency 초기화를 늦춰 demo startup을 가볍게 한다.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_arr = np.asarray(history["time"], dtype=np.float64)
    prob = np.asarray(history["probability"], dtype=np.float64)
    label = np.asarray(history["label"], dtype=np.float64)
    tau_ext_norm = np.asarray(history["tau_ext_norm"], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].step(time_arr, label, where="post", color="black", linewidth=2.0, label="contact label")
    axes[0].plot(time_arr, tau_ext_norm, color="tab:orange", linewidth=1.5, label="disturbance norm")
    axes[0].set_ylabel("Label / disturbance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time_arr, prob, color="tab:blue", linewidth=2.0, label="GRU P(contact)")
    axes[1].axhline(float(history["threshold"][0]), color="tab:blue", linestyle="--", linewidth=1.2)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("P(contact)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    q = np.asarray(history["q"], dtype=np.float64)
    for idx in range(q.shape[1]):
        axes[2].plot(time_arr, q[:, idx], linewidth=1.0, label=f"q{idx + 1}")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("q_ros [rad]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=6, fontsize=8, loc="upper right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _active_intervals(time_arr: np.ndarray, label_arr: np.ndarray) -> list[tuple[float, float]]:
    """Convert a binary contact trace into shaded intervals for paper plots."""
    intervals: list[tuple[float, float]] = []
    if time_arr.size == 0:
        return intervals
    active = label_arr.astype(bool)
    start_idx: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        if start_idx is not None and ((not is_active) or idx == len(active) - 1):
            end_idx = idx if not is_active else idx + 1
            left = float(time_arr[start_idx])
            right_idx = min(end_idx, len(time_arr) - 1)
            right = float(time_arr[right_idx])
            if right > left:
                intervals.append((left, right))
            start_idx = None
    return intervals


def _paper_crop(time_arr: np.ndarray, label_arr: np.ndarray, crop_active: bool) -> tuple[float, float]:
    """Choose a compact paper-figure time window around user interaction."""
    if time_arr.size == 0:
        return 0.0, 1.0
    if not crop_active or not np.any(label_arr > 0):
        return float(time_arr[0]), float(time_arr[-1])
    active_idx = np.flatnonzero(label_arr > 0)
    return max(float(time_arr[0]), float(time_arr[active_idx[0]]) - 0.5), min(
        float(time_arr[-1]),
        float(time_arr[active_idx[-1]]) + 0.8,
    )


def save_paper_interaction_plot(path: Path, history: dict[str, list], crop_active: bool = True) -> tuple[float, float]:
    """Save a clean 2-panel paper figure from scripted/manual/mouse interaction history.

    For `--mouse`, the top panel uses the selected-body force norm read from
    `data.xfrc_applied[selected_body, :3]` in Newtons. If no mouse force was
    recorded, it falls back to the generic disturbance norm used by scripted
    demos. The plotted signals are all recorded from the live MuJoCo run; no
    synthetic data are created here.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_arr = np.asarray(history["time"], dtype=np.float64)
    prob = np.asarray(history["probability"], dtype=np.float64)
    label = np.asarray(history["label"], dtype=np.float64)
    threshold = float(history["threshold"][0])
    tau_ext_norm = np.asarray(history["tau_ext_norm"], dtype=np.float64)
    mouse_force_norm = np.asarray(history.get("mouse_force_norm", []), dtype=np.float64)
    mouse_torque_norm = np.asarray(history.get("mouse_torque_norm", []), dtype=np.float64)

    if mouse_force_norm.shape == time_arr.shape and np.nanmax(mouse_force_norm) > 1.0e-9:
        top_signal = mouse_force_norm
        top_label = r"$||F_\mathrm{mouse}||$ [N]"
        top_ylabel = "Mouse force [N]"
    elif mouse_torque_norm.shape == time_arr.shape and np.nanmax(mouse_torque_norm) > 1.0e-9:
        top_signal = mouse_torque_norm
        top_label = r"$||T_\mathrm{mouse}||$ [Nm]"
        top_ylabel = "Mouse torque [Nm]"
    else:
        top_signal = tau_ext_norm
        top_label = r"$||\tau_\mathrm{ext}||$"
        top_ylabel = "Disturbance"

    crop_start, crop_end = _paper_crop(time_arr, label, crop_active)
    mask = (time_arr >= crop_start) & (time_arr <= crop_end)

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(6.7, 3.55), sharex=True, constrained_layout=True)
    intervals = _active_intervals(time_arr, label)
    for ax in axes:
        for left, right in intervals:
            if right < crop_start or left > crop_end:
                continue
            ax.axvspan(max(left, crop_start), min(right, crop_end), color="0.88", alpha=0.8, linewidth=0)
        ax.grid(True, axis="y", color="0.90", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].plot(time_arr[mask], top_signal[mask], color="tab:orange", linewidth=1.8, label=top_label)
    axes[0].step(time_arr[mask], label[mask], where="post", color="black", linewidth=1.2, label="contact label")
    axes[0].set_ylabel(top_ylabel)
    axes[0].set_title("MuJoCo interaction example", pad=3)
    axes[0].legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.9, handlelength=1.5)
    axes[0].set_ylim(bottom=-0.05)

    axes[1].plot(time_arr[mask], prob[mask], color="tab:blue", linewidth=1.9, label="GRU P(contact)")
    axes[1].axhline(threshold, color="0.35", linestyle="--", linewidth=1.1, label="threshold")
    axes[1].set_ylabel("P(contact)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_xlim(crop_start, crop_end)
    axes[1].legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.9, handlelength=1.5)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return crop_start, crop_end


def save_history_npz(path: Path, history: dict[str, list]) -> None:
    """Persist the live MuJoCo demo trace for later paper figure regeneration."""
    arrays: dict[str, np.ndarray] = {}
    for key, values in history.items():
        if key in {"selected_body_name", "scaled_feature_top_name"}:
            arrays[key] = np.asarray(values, dtype="<U64")
        else:
            arrays[key] = np.asarray(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _safe_float(value) -> float | None:
    try:
        value_float = float(value)
    except Exception:
        return None
    return value_float if math.isfinite(value_float) else None


def _history_array(history: dict[str, list], key: str) -> np.ndarray:
    return np.asarray(history.get(key, []), dtype=np.float64)


def _row_summary(history: dict[str, list], index: int) -> dict:
    if index < 0 or index >= len(history.get("time", [])):
        return {}
    keys = [
        "time",
        "probability",
        "label",
        "mouse_force_norm",
        "mouse_torque_norm",
        "joint_tau_equiv_norm",
        "q_norm",
        "qdot_norm",
        "e_norm",
        "delta_e_norm",
        "delta_qdot_norm",
        "tau_cmd_norm",
        "delta_tau_cmd_norm",
        "scaled_feature_max_abs",
        "scaled_feature_top_value",
    ]
    row = {key: _safe_float(history[key][index]) for key in keys if key in history and len(history[key]) > index}
    if "selected_body_name" in history and len(history["selected_body_name"]) > index:
        row["selected_body_name"] = str(history["selected_body_name"][index])
    if "scaled_feature_top_name" in history and len(history["scaled_feature_top_name"]) > index:
        row["scaled_feature_top_name"] = str(history["scaled_feature_top_name"][index])
    if "joint_tau_equiv" in history and len(history["joint_tau_equiv"]) > index:
        row["joint_tau_equiv"] = [float(value) for value in np.asarray(history["joint_tau_equiv"][index]).reshape(-1)]
    return row


def build_debug_summary(history: dict[str, list], threshold: float, model_path: Path, scaler_path: Path) -> dict:
    prob = _history_array(history, "probability")
    finite_prob = prob[np.isfinite(prob)]
    tau_eq = _history_array(history, "joint_tau_equiv_norm")
    force = _history_array(history, "mouse_force_norm")
    torque = _history_array(history, "mouse_torque_norm")
    e_norm = _history_array(history, "e_norm")
    delta_e = _history_array(history, "delta_e_norm")
    qdot = _history_array(history, "qdot_norm")
    delta_qdot = _history_array(history, "delta_qdot_norm")
    dtau = _history_array(history, "delta_tau_cmd_norm")
    scaled = _history_array(history, "scaled_feature_max_abs")

    peak_prob_index = int(np.nanargmax(prob)) if np.any(np.isfinite(prob)) else -1
    peak_tau_index = int(np.nanargmax(tau_eq)) if tau_eq.size else -1
    max_probability = None if finite_prob.size == 0 else float(np.max(finite_prob))
    max_tau_eq = None if tau_eq.size == 0 else float(np.max(tau_eq))
    detected = bool(max_probability is not None and max_probability >= float(threshold))

    diagnosis = []
    if max_tau_eq is not None and max_tau_eq > 1.0 and not detected:
        diagnosis.append(
            "Large applied equivalent joint torque was observed, but P(contact) did not cross threshold."
        )
        max_delta_motion = max(
            float(np.max(delta_e)) if delta_e.size else 0.0,
            float(np.max(delta_qdot)) if delta_qdot.size else 0.0,
            float(np.max(qdot)) if qdot.size else 0.0,
        )
        if max_delta_motion < 1.0e-2:
            diagnosis.append(
                "The applied wrench produced little proprioceptive response in q/qdot/e_q; sensorless input cues may be weak."
            )
        else:
            diagnosis.append(
                "Proprioceptive response exists, but this body-side wrench pattern may be outside the training distribution."
            )
    elif detected:
        diagnosis.append("P(contact) crossed the decision threshold during this run.")
    else:
        diagnosis.append("No sufficiently large applied torque/contact-like response was observed.")

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "decision_threshold": float(threshold),
        "model_input_note": (
            "joint_tau_equiv, mouse force/torque, and tau_ext are debug signals only. "
            "The GRU input is [q, qdot, e_q, tau_cmd, delta_e_q, delta_qdot, delta_tau_cmd]."
        ),
        "max_probability": max_probability,
        "detected_by_threshold": detected,
        "fraction_probability_above_threshold": None
        if finite_prob.size == 0
        else float(np.mean(finite_prob >= float(threshold))),
        "max_mouse_force_norm_n": None if force.size == 0 else float(np.max(force)),
        "max_mouse_torque_norm_nm": None if torque.size == 0 else float(np.max(torque)),
        "max_joint_tau_equiv_norm_nm": max_tau_eq,
        "max_e_norm": None if e_norm.size == 0 else float(np.max(e_norm)),
        "max_delta_e_norm": None if delta_e.size == 0 else float(np.max(delta_e)),
        "max_qdot_norm": None if qdot.size == 0 else float(np.max(qdot)),
        "max_delta_qdot_norm": None if delta_qdot.size == 0 else float(np.max(delta_qdot)),
        "max_delta_tau_cmd_norm": None if dtau.size == 0 else float(np.max(dtau)),
        "max_scaled_feature_abs": None if scaled.size == 0 else float(np.max(scaled)),
        "peak_probability_sample": _row_summary(history, peak_prob_index),
        "peak_joint_tau_sample": _row_summary(history, peak_tau_index),
        "diagnosis": diagnosis,
        "recommended_next_checks": [
            "If joint_tau_equiv is high but q/qdot/e_q deltas are small, use stronger or longer body force or a body/contact direction that creates motion.",
            "If q/qdot/e_q deltas are high but P(contact) stays low, retrain with body_force or mixed disturbances on torso/link contact frames.",
            "For body-side contact coverage, regenerate data with disturbance.type='mixed' or 'body_force' and include intermediate link frames/bodies.",
        ],
    }


def save_debug_summary(path: Path, history: dict[str, list], threshold: float, model_path: Path, scaler_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(build_debug_summary(history, threshold, model_path, scaler_path), stream, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to contact_detection/config.yaml")
    parser.add_argument("--stage", default=None, help="Curriculum stage/model directory to use.")
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional GRU checkpoint path. Defaults to outputs/<stage>/models/gru_detector.pt.",
    )
    parser.add_argument(
        "--scaler-path",
        default="",
        help="Optional scaler path. Defaults to outputs/<stage>/models/scaler.pkl.",
    )
    parser.add_argument("--model-xml", default="", help="Optional MuJoCo XML path.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--trajectory", choices=["hold", "slow_sine"], default="hold")
    parser.add_argument("--mouse", action="store_true", help="Use MuJoCo mouse perturbation as the external contact source.")
    parser.add_argument("--manual", action="store_true", help="Use viewer key commands to inject disturbance pulses.")
    parser.add_argument("--manual-mode", choices=["joint_torque", "body_force"], default="joint_torque")
    parser.add_argument("--manual-motor", type=int, default=4, help="Initial motor id for manual pulse injection.")
    parser.add_argument("--manual-tau", type=float, default=0.6, help="Manual disturbance pulse magnitude in Nm.")
    parser.add_argument("--manual-body", default="ee_link", help="MuJoCo body name for manual body-force pulses.")
    parser.add_argument("--manual-force", type=float, default=5.0, help="Manual body-force pulse magnitude in N.")
    parser.add_argument("--manual-duration", type=float, default=0.35, help="Manual disturbance pulse duration in seconds.")
    parser.add_argument(
        "--events-json",
        default="",
        help='Optional events, e.g. \'[[2.0,2.7,{"4":1.3}],[4.5,5.2,{"2":-1.4,"5":0.8}]]\'.',
    )
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer and replay in real time.")
    parser.add_argument(
        "--viewer-sync-hz",
        type=float,
        default=60.0,
        help="MuJoCo viewer refresh rate. Lower values reduce GUI overhead when the demo runs slowly.",
    )
    parser.add_argument(
        "--real-time-factor",
        type=float,
        default=1.0,
        help="Simulation seconds per wall-clock second. Use >1.0 to replay faster when the GUI is slow.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not sleep to match wall-clock time. Useful for headless scripted debugging.",
    )
    parser.add_argument(
        "--print-period-s",
        type=float,
        default=0.5,
        help="Console status print period in simulation seconds.",
    )
    parser.add_argument("--output", default="", help="Optional output figure path.")
    parser.add_argument("--history-output", default="", help="Optional .npz path for the recorded live demo trace.")
    parser.add_argument(
        "--debug-summary-output",
        default="",
        help="Optional JSON path for a compact contact-debug diagnosis summary.",
    )
    parser.add_argument("--paper-output", default="", help="Optional clean 2-panel paper figure path.")
    parser.add_argument(
        "--no-paper-crop-active",
        action="store_true",
        help="Do not crop the paper figure around active interaction intervals.",
    )
    parser.add_argument("--ros-sign-by-motor-json", default=json.dumps(DEFAULT_ROS_SIGN_BY_MOTOR, sort_keys=True))
    args = parser.parse_args()

    # 1) stage별 model/scaler 위치를 결정한다. 보통 --stage randomized_sim을 쓴다.
    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    out_dirs = ensure_output_dirs(output_root(config))
    dof = int(config["robot"]["dof"])
    motor_ids = list(range(1, dof + 1))
    joint_names = robot_joint_names(config)
    sign_map = parse_float_map_json(args.ros_sign_by_motor_json)
    sign = np.asarray([-1.0 if sign_map.get(motor_id, 1.0) < 0.0 else 1.0 for motor_id in motor_ids])

    model_path = Path(args.model_path).expanduser() if args.model_path.strip() else out_dirs["models"] / "gru_detector.pt"
    scaler_path = Path(args.scaler_path).expanduser() if args.scaler_path.strip() else out_dirs["models"] / "scaler.pkl"
    model_path = model_path.resolve()
    scaler_path = scaler_path.resolve()
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Train the detector before running mujoco_contact_demo.py")

    # 2) PyTorch checkpoint와 scaler를 load한다. demo에서도 학습된 scaler를 그대로 써야 한다.
    import torch

    checkpoint = torch.load(model_path, map_location="cpu")
    scaler = StandardScaler.load(scaler_path)
    model_type = str(checkpoint.get("model_type", "binary")).strip().lower()
    if model_type != "binary":
        raise ValueError(f"{model_path} is not a binary checkpoint. Retrain with config.yaml.")
    detector = GRUDetector(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        bidirectional=bool(checkpoint.get("bidirectional", False)),
    )
    detector.load_state_dict(checkpoint["state_dict"])
    device = select_torch_device(torch, config["training"].get("device", "auto"))
    detector.to(device)
    detector.eval()
    window_length = int(checkpoint.get("window_length", config["dataset"]["window_length"]))
    use_delta_features = bool(checkpoint.get("use_delta_features", config["dataset"]["use_delta_features"]))
    feature_names = list(checkpoint.get("feature_names", []))
    configured_threshold = config["training"].get("gru_decision_threshold")
    threshold = float(
        configured_threshold if configured_threshold is not None else checkpoint.get("decision_threshold", 0.5)
    )
    baseline_gamma = None
    threshold_path = model_path.parent / "threshold.json" if args.model_path.strip() else out_dirs["models"] / "threshold.json"
    if threshold_path.exists():
        threshold_payload = load_json(threshold_path)
        if threshold_payload.get("threshold_metric", "error_norm") == "error_norm":
            baseline_gamma = float(threshold_payload["gamma"])

    # 3) MuJoCo XML을 load하고 joint/actuator index를 이름 기반으로 찾는다.
    #    이름 기반 mapping을 쓰면 XML 내부 순서 변화에 더 안전하다.
    model_xml = resolve_model_xml(config, args.model_xml)
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    qpos_idx, qvel_idx = qpos_indices(model, joint_names)
    joint_ids = named_joint_ids(model, joint_names)
    ctrl_idx = actuator_indices(model, joint_names)
    manual_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.manual_body))
    if args.manual and args.manual_mode == "body_force" and manual_body_id < 0:
        raise ValueError(f"MuJoCo body not found for manual body force: {args.manual_body}")
    q0_ros, _ = reference_at(0.0, dof, args.trajectory)
    data.qpos[qpos_idx] = q0_ros * sign
    data.qvel[qvel_idx] = 0.0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    kp = np.asarray(config["simulation"]["kp"], dtype=np.float64)
    kd = np.asarray(config["simulation"]["kd"], dtype=np.float64)
    # Mouse/manual demos should reflect the user's interaction only by default.
    # If a scripted + mouse mixed demo is needed, pass --events-json explicitly.
    if args.manual or (args.mouse and not args.events_json.strip()):
        events = []
    else:
        events = parse_events_json(args.events_json, dof)
    manual_events: list[Event] = []
    manual_body_force_events: list[BodyForceEvent] = []
    manual_state = make_manual_state(max(1, min(dof, int(args.manual_motor))))
    feature_window: deque[np.ndarray] = deque(maxlen=window_length)
    prev_e_q: np.ndarray | None = None
    prev_qd: np.ndarray | None = None
    prev_tau: np.ndarray | None = None

    # 4) viewer가 켜지면 scripted disturbance뿐 아니라 mouse/manual 입력도 받을 수 있다.
    viewer = None
    viewer_enabled = bool(args.viewer or args.manual or args.mouse)
    if viewer_enabled:
        import mujoco.viewer as mujoco_viewer

        key_callback = make_key_callback(manual_state, dof) if args.manual else None
        viewer = mujoco_viewer.launch_passive(model, data, key_callback=key_callback)
        if args.mouse:
            print("[mouse] MuJoCo mouse perturb enabled")
            print("[mouse] double-click a body to select it, then Ctrl + drag in the viewer to pull/rotate it")
        if args.manual:
            print("[manual] viewer controls: F positive pulse, R negative pulse, C clear pulses")
            print("[manual] +/- changes magnitude, [/ ] changes duration")
            if args.manual_mode == "joint_torque":
                print("[manual] joint torque mode: 1-6 select motor/joint")
                print(f"[manual] default pulse: motor {manual_state['motor']}, |tau|={args.manual_tau:.3f} Nm, duration={args.manual_duration:.3f}s")
            else:
                print("[manual] body force mode: X/Y/Z select world force axis")
                print(f"[manual] default pulse: body {args.manual_body}, |force|={args.manual_force:.3f} N, duration={args.manual_duration:.3f}s")

    history: dict[str, list] = {
        "time": [],
        "probability": [],
        "label": [],
        "tau_ext_norm": [],
        "body_force_norm": [],
        "mouse_force": [],
        "mouse_torque": [],
        "mouse_force_norm": [],
        "mouse_torque_norm": [],
        "joint_tau_equiv": [],
        "joint_tau_equiv_norm": [],
        "selected_body_id": [],
        "selected_body_name": [],
        "q_norm": [],
        "qdot_norm": [],
        "e_norm": [],
        "tau_cmd_norm": [],
        "delta_e_norm": [],
        "delta_qdot_norm": [],
        "delta_tau_cmd_norm": [],
        "scaled_feature_max_abs": [],
        "scaled_feature_top_index": [],
        "scaled_feature_top_name": [],
        "scaled_feature_top_value": [],
        "threshold": [threshold],
        "q": [],
    }
    last_print_s = -1.0
    steps = int(round(float(args.duration) / dt))
    wall_start = time.monotonic()
    viewer_period_s = 1.0 / max(float(args.viewer_sync_hz), 1.0e-6)
    next_viewer_sync_s = 0.0
    interrupted = False

    try:
      for step in range(steps):
        # 5) 매 sim step에서 현재 q/qdot, reference, commanded torque를 계산한다.
        sim_time = step * dt
        if args.manual:
            if args.manual_mode == "joint_torque":
                new_events, clear_manual = consume_manual_joint_events(
                    manual_state,
                    sim_time,
                    dof,
                    args.manual_tau,
                    args.manual_duration,
                )
                new_body_force_events: list[BodyForceEvent] = []
            else:
                new_events = []
                new_body_force_events, clear_manual = consume_manual_body_force_events(
                    manual_state,
                    sim_time,
                    manual_body_id,
                    args.manual_force,
                    args.manual_duration,
                )
            if clear_manual:
                manual_events.clear()
                manual_body_force_events.clear()
                print("[manual] cleared active manual pulses")
            manual_events.extend(new_events)
            manual_body_force_events.extend(new_body_force_events)
            manual_events = [event for event in manual_events if event.end > sim_time]
            manual_body_force_events = [event for event in manual_body_force_events if event.end > sim_time]

        q_mj = np.asarray(data.qpos[qpos_idx], dtype=np.float64)
        qd_mj = np.asarray(data.qvel[qvel_idx], dtype=np.float64)
        q_ros = q_mj * sign
        qd_ros = qd_mj * sign
        q_des_ros, qd_des_ros = reference_at(sim_time, dof, args.trajectory)
        q_des_mj = q_des_ros * sign
        qd_des_mj = qd_des_ros * sign

        # qfrc_bias는 MuJoCo가 계산한 bias force(Coriolis/centrifugal/gravity)다.
        # 여기서는 gravity-hold 학습 정의와 맞추기 위해 PD + bias를 tau_cmd로 둔다.
        mujoco.mj_forward(model, data)
        tau_cmd_mj = kp * (q_des_mj - q_mj) + kd * (qd_des_mj - qd_mj) + data.qfrc_bias[qvel_idx]
        # The trained detector's tau_cmd feature follows the command/joint
        # convention used by the offline generator and ROS command stream.  The
        # MuJoCo actuator force is already expressed in that convention for the
        # current XML; applying the position sign map here flips gravity torques
        # and makes a no-contact hold look like contact.
        tau_cmd_ros = tau_cmd_mj.copy()
        data.ctrl[ctrl_idx] = tau_cmd_mj
        # scripted/manual joint torque disturbance는 qfrc_applied로 넣는다.
        # 이 값은 label/demo용이고 feature_row에는 들어가지 않는다.
        tau_ext_mj, label = tau_ext_at(sim_time, events + manual_events, sign)
        tau_ext_ros = tau_ext_mj * sign
        data.qfrc_applied[qvel_idx] = tau_ext_mj
        data.xfrc_applied[:, :] = 0.0
        active_body_ids: list[int] = []
        # body force disturbance와 mouse perturb는 xfrc_applied에 들어간다.
        # 역시 모델 입력이 아니라 MuJoCo physics와 viewer HUD용이다.
        for event in manual_body_force_events:
            if event.start <= sim_time < event.end:
                data.xfrc_applied[event.body_id, :3] += event.force_world
                active_body_ids.append(event.body_id)
                label = 1
        mouse_active = False
        mouse_body_id = -1
        if args.mouse:
            mouse_body_id = selected_perturb_body_id(viewer)
            mouse_active, mouse_body_ids = apply_mouse_perturb_force(viewer, model, data)
            active_body_ids.extend(mouse_body_ids)
            if mouse_active:
                label = 1
        body_force_norm = float(np.linalg.norm(data.xfrc_applied))
        logged_body_id = mouse_body_id if args.mouse else (active_body_ids[0] if active_body_ids else -1)
        logged_force = np.zeros(3, dtype=np.float64)
        logged_torque = np.zeros(3, dtype=np.float64)
        logged_body_name = "none"
        if logged_body_id >= 0:
            logged_force = np.asarray(data.xfrc_applied[logged_body_id, :3], dtype=np.float64).copy()
            logged_torque = np.asarray(data.xfrc_applied[logged_body_id, 3:], dtype=np.float64).copy()
            logged_body_name = object_name(model, mujoco.mjtObj.mjOBJ_BODY, logged_body_id)
        joint_tau_equiv_ros = body_wrench_to_joint_tau_ros(
            model,
            data,
            logged_body_id,
            logged_force,
            logged_torque,
            qvel_idx,
            sign,
        )
        applied_joint_tau_ros = tau_ext_ros + joint_tau_equiv_ros

        # 6) 학습과 같은 feature order로 row를 만들고 train scaler로 normalize한다.
        feature_row, prev_e_q, prev_qd, prev_tau = build_feature_row(
            q_ros,
            qd_ros,
            q_des_ros,
            tau_cmd_ros,
            use_delta_features,
            prev_e_q,
            prev_qd,
            prev_tau,
        )
        e_norm = float(np.linalg.norm(prev_e_q))
        baseline_pred = None if baseline_gamma is None else int(e_norm >= float(baseline_gamma))
        feature_scaled = scaler.transform(feature_row[None, :]).astype(np.float32)[0]
        debug_metrics = feature_debug_metrics(
            feature_row,
            feature_scaled,
            dof,
            use_delta_features,
            feature_names,
        )
        feature_window.append(feature_scaled)

        probability = float("nan")
        # 7) window_length만큼 쌓인 뒤에만 GRU inference를 수행한다.
        if len(feature_window) == window_length:
            window = np.stack(list(feature_window), axis=0)[None, :, :]
            with torch.no_grad():
                logits = detector(torch.as_tensor(window, dtype=torch.float32, device=device))
                probability = float(sigmoid(logits.cpu().numpy())[0])

        # 8) physics step을 진행하고 viewer/HUD/history를 갱신한다.
        mujoco.mj_step(model, data)
        if viewer is not None and sim_time + 1.0e-12 >= next_viewer_sync_s:
            update_disturbance_markers(viewer, data, joint_ids, tau_ext_ros, active_body_ids)
            if args.mouse:
                update_force_hud(
                    viewer,
                    model,
                    data,
                    mouse_body_id,
                    probability,
                    threshold,
                    e_norm,
                    baseline_gamma,
                    applied_joint_tau_ros,
                )
            try:
                viewer.sync()
            except KeyboardInterrupt:
                interrupted = True
                print("\nInterrupted by user; closing MuJoCo viewer.")
                break
            except Exception as exc:
                interrupted = True
                print(f"\nMuJoCo viewer sync failed; stopping demo cleanly: {exc}")
                break
            if not args.no_realtime:
                target_wall = wall_start + sim_time / max(float(args.real_time_factor), 1.0e-6)
                sleep_s = target_wall - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(min(sleep_s, viewer_period_s))
            if not viewer.is_running():
                break
            next_viewer_sync_s = sim_time + viewer_period_s

        history["time"].append(sim_time)
        history["probability"].append(probability)
        history["label"].append(label)
        history["tau_ext_norm"].append(float(np.linalg.norm(tau_ext_ros)) + body_force_norm)
        history["body_force_norm"].append(body_force_norm)
        history["mouse_force"].append(logged_force.copy())
        history["mouse_torque"].append(logged_torque.copy())
        history["mouse_force_norm"].append(float(np.linalg.norm(logged_force)))
        history["mouse_torque_norm"].append(float(np.linalg.norm(logged_torque)))
        history["joint_tau_equiv"].append(applied_joint_tau_ros.copy())
        history["joint_tau_equiv_norm"].append(float(np.linalg.norm(applied_joint_tau_ros)))
        history["selected_body_id"].append(int(logged_body_id))
        history["selected_body_name"].append(logged_body_name)
        history["q_norm"].append(float(debug_metrics["q_norm"]))
        history["qdot_norm"].append(float(debug_metrics["qdot_norm"]))
        history["e_norm"].append(float(debug_metrics["e_norm"]))
        history["tau_cmd_norm"].append(float(debug_metrics["tau_cmd_norm"]))
        history["delta_e_norm"].append(float(debug_metrics["delta_e_norm"]))
        history["delta_qdot_norm"].append(float(debug_metrics["delta_qdot_norm"]))
        history["delta_tau_cmd_norm"].append(float(debug_metrics["delta_tau_cmd_norm"]))
        history["scaled_feature_max_abs"].append(float(debug_metrics["scaled_feature_max_abs"]))
        history["scaled_feature_top_index"].append(int(debug_metrics["scaled_feature_top_index"]))
        history["scaled_feature_top_name"].append(str(debug_metrics["scaled_feature_top_name"]))
        history["scaled_feature_top_value"].append(float(debug_metrics["scaled_feature_top_value"]))
        history["q"].append(q_ros.copy())

        if sim_time - last_print_s >= float(args.print_period_s):
            last_print_s = sim_time
            prob_text = "warming" if not math.isfinite(probability) else f"{probability:.3f}"
            baseline_text = "" if baseline_pred is None else f" baseline={baseline_pred} ||e_q||={e_norm:.3f}"
            print(
                f"t={sim_time:5.2f}s label={label} P(contact)={prob_text}{baseline_text} "
                f"||F||={np.linalg.norm(logged_force):.2f}N "
                f"||T||={np.linalg.norm(logged_torque):.2f}Nm "
                f"||tau_eq||={np.linalg.norm(applied_joint_tau_ros):.2f}Nm "
                f"max|tau_eq|={np.max(np.abs(applied_joint_tau_ros)):.2f}Nm "
                f"||qdot||={debug_metrics['qdot_norm']:.3f} "
                f"||de||={debug_metrics['delta_e_norm']:.4f} "
                f"||dqdot||={debug_metrics['delta_qdot_norm']:.4f} "
                f"||tau_cmd||={debug_metrics['tau_cmd_norm']:.2f} "
                f"||dtau_cmd||={debug_metrics['delta_tau_cmd_norm']:.3f} "
                f"zmax={debug_metrics['scaled_feature_max_abs']:.2f}"
                f"({debug_metrics['scaled_feature_top_name']}={debug_metrics['scaled_feature_top_value']:.2f}) "
                f"body={logged_body_name}"
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user; saving the trace collected so far.")

    if viewer is not None:
        try:
            viewer.close()
        except Exception as exc:
            if not interrupted:
                print(f"MuJoCo viewer close warning: {exc}")

    output_path = Path(args.output).expanduser().resolve() if args.output else out_dirs["figures"] / "mujoco_contact_demo.png"
    save_demo_plot(output_path, history)
    print(f"Saved MuJoCo contact demo plot to {output_path}")
    history_output = Path(args.history_output).expanduser().resolve() if args.history_output else None
    debug_summary_output = (
        Path(args.debug_summary_output).expanduser().resolve() if args.debug_summary_output else None
    )
    paper_output = Path(args.paper_output).expanduser().resolve() if args.paper_output else None
    if paper_output is not None and history_output is None:
        history_output = paper_output.with_suffix(".npz")
    if history_output is not None:
        save_history_npz(history_output, history)
        print(f"Saved MuJoCo contact demo raw trace to {history_output}")
        if debug_summary_output is None:
            debug_summary_output = history_output.with_suffix(".debug_summary.json")
    if debug_summary_output is not None:
        save_debug_summary(debug_summary_output, history, threshold, model_path, scaler_path)
        print(f"Saved MuJoCo debug summary to {debug_summary_output}")
    if paper_output is not None:
        crop = save_paper_interaction_plot(
            paper_output,
            history,
            crop_active=not bool(args.no_paper_crop_active),
        )
        print(f"Saved clean paper interaction figure to {paper_output}")
        print(f"Paper figure interval: {crop[0]:.2f}s to {crop[1]:.2f}s")


if __name__ == "__main__":
    main()
