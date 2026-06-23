"""Generate simulation datasets for sensorless binary contact detection.

논문 흐름에서 이 파일은 "실제 로봇에는 F/T 센서가 없어서 정답 접촉
라벨을 얻기 어렵다"는 문제를 해결하기 위한 시뮬레이션 데이터 생성
단계에 해당한다.

핵심 아이디어:
- 실제 로봇과 같은 6-DOF joint order와 URDF 기반 동역학을 사용한다.
- controller가 내는 commanded torque(tau_cmd)를 기록한다.
- 외부 접촉 효과는 selected joint disturbance torque(tau_ext)로 근사한다.
- label은 ||tau_ext|| > eps인 구간을 contact=1로 둔다.
- tau_ext는 저장만 한다. 모델 입력 feature에는 절대 넣지 않는다.

저장되는 데이터:
- q, qdot, q_des, qdot_des, tau_cmd_raw, tau_cmd, tau_ext, label, episode_id
- train/val/test는 episode 단위로 생성되며 episode_id가 겹치면 에러를 낸다.
"""

from __future__ import annotations

# argparse: terminal command에서 --config, --stage를 받기 위한 표준 라이브러리.
# math/warnings: sine trajectory 계산, fallback/품질 warning 출력에 사용한다.
import argparse
import math
import warnings
from dataclasses import dataclass

# NumPy는 이 파이프라인의 기본 array 엔진이다. 모든 q/qdot/tau/label 시계열은
# [N, dof] 또는 [N] 형태의 numpy array로 만들고 npz로 저장한다.
import numpy as np

from utils import (
    apply_stage_config,
    ensure_output_dirs,
    ensure_vector_length,
    frame_to_region_id,
    load_config,
    output_root,
    resolve_path,
    robot_joint_names,
    save_config_yaml,
    save_json,
    set_global_seed,
    validate_motor_joint_map,
)


try:
    # Pinocchio는 URDF 기반 rigid-body dynamics 계산 라이브러리다.
    # 설치되어 있으면 gravity, mass matrix, nonlinear effects를 실제 URDF로 계산한다.
    import pinocchio as pin
except ImportError:
    pin = None


@dataclass
class EpisodeRandomization:
    # episode마다 달라지는 sim-to-real gap 요소.
    # 마찰/댐핑/토크 스케일/제어 delay가 매 episode에서 랜덤하게 바뀐다.
    damping: np.ndarray
    viscous_friction: np.ndarray
    coulomb_friction: np.ndarray
    torque_scale: np.ndarray
    control_delay_steps: int


@dataclass
class DisturbanceEvent:
    # 한 번의 접촉 이벤트를 표현한다. 기본 논문 실험은 kind="joint_torque"를 쓴다.
    # body_force 관련 필드는 MuJoCo/URDF frame force를 joint torque로 변환하는 optional path다.
    start_step: int
    end_step: int
    kind: str
    tau: np.ndarray | None = None
    region_id: int = 0
    frame_name: str | None = None
    force_world: np.ndarray | None = None
    torque_world: np.ndarray | None = None
    event_id: int = -1
    representative_magnitude: float = 0.0


class PinocchioNominalDynamics:
    """URDF-based nominal dynamics driven by Pinocchio."""

    def __init__(self, urdf_path: str, dof: int, joint_names: list[str]) -> None:
        if pin is None:
            raise ImportError("Pinocchio is not available")
        self.urdf_path = str(urdf_path)
        self.dof = int(dof)
        self.mass_matrix_regularization = 1.0e-4
        self.max_abs_qddot = 200.0
        self.max_abs_qdot = 20.0
        full_model = pin.buildModelFromUrdf(self.urdf_path)
        # Pinocchio model.names[0]은 universe라서 제외한다. 새 URDF에 gripper
        # prismatic joint처럼 학습에 쓰지 않는 movable joint가 붙어 있을 수 있다.
        # contact detection 연구 범위는 6-DOF arm이므로 config의 joint_names에
        # 없는 movable joint는 neutral position에서 lock한 reduced model을 만든다.
        full_joint_names = [str(name) for name in full_model.names[1:]]
        desired_joint_names = list(joint_names)
        missing_joints = [name for name in desired_joint_names if name not in full_joint_names]
        if missing_joints:
            raise ValueError(
                f"URDF is missing configured robot.joint_names {missing_joints}. "
                f"Movable joints found in URDF: {full_joint_names}"
            )
        extra_joints = [name for name in full_joint_names if name not in desired_joint_names]
        if extra_joints:
            locked_joint_ids = [int(full_model.getJointId(name)) for name in extra_joints]
            reference_q = pin.neutral(full_model)
            self.model = pin.buildReducedModel(full_model, locked_joint_ids, reference_q)
            warnings.warn(
                "Locked extra URDF movable joints for 6-DOF contact detection: "
                f"{extra_joints}. If these are gripper joints, this is expected."
            )
        else:
            self.model = full_model
        self.data = self.model.createData()
        lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64)
        self.lower_position_limit = np.where(np.isfinite(lower), lower, -np.inf)
        self.upper_position_limit = np.where(np.isfinite(upper), upper, np.inf)
        if self.model.nq != self.dof or self.model.nv != self.dof:
            raise ValueError(
                f"URDF DoF mismatch: expected nq=nv={self.dof}, got nq={self.model.nq}, nv={self.model.nv}"
            )
        # 여기서 joint order가 config의 j1..j6와 다르면 학습/실제 적용이 바로 깨질 수 있다.
        actual_joint_names = [str(name) for name in self.model.names[1:]]
        if actual_joint_names != list(joint_names):
            raise ValueError(f"URDF joint order mismatch: {actual_joint_names} vs config {joint_names}")

    def gravity(self, q: np.ndarray) -> np.ndarray:
        # computeGeneralizedGravity는 현재 자세 q에서 정지 상태를 유지하는 데 필요한
        # generalized gravity torque를 반환한다. 학습 tau_cmd_raw에 +g(q)로 들어간다.
        return np.asarray(pin.computeGeneralizedGravity(self.model, self.data, q), dtype=np.float64)

    def has_frame(self, frame_name: str) -> bool:
        return bool(self.model.existFrame(frame_name))

    def contact_force_to_joint_torque(
        self,
        q: np.ndarray,
        frame_name: str,
        force_world: np.ndarray,
        torque_world: np.ndarray,
    ) -> np.ndarray:
        """Project a world-frame contact wrench at a URDF frame to generalized joint torque."""

        if not self.has_frame(frame_name):
            raise ValueError(f"URDF frame not found for body-force disturbance: {frame_name}")
        frame_id = int(self.model.getFrameId(frame_name))
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian = np.asarray(
            pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED),
            dtype=np.float64,
        )
        wrench = np.concatenate(
            [np.asarray(force_world, dtype=np.float64), np.asarray(torque_world, dtype=np.float64)],
            axis=0,
        )
        return np.asarray(jacobian.T @ wrench, dtype=np.float64)

    def step(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        tau_cmd_applied: np.ndarray,
        tau_ext: np.ndarray,
        randomization: EpisodeRandomization,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 시뮬레이션 방정식:
        #   M(q) qddot + h(q,qdot) = tau_cmd + tau_ext - friction
        # 여기서 h는 Coriolis/centrifugal/gravity를 포함한 nonLinearEffects다.
        tau_total = np.asarray(tau_cmd_applied + tau_ext, dtype=np.float64)
        tau_total -= randomization.damping * qdot
        tau_total -= randomization.viscous_friction * qdot
        tau_total -= randomization.coulomb_friction * np.sign(qdot)
        # crba: Composite Rigid Body Algorithm. M(q)를 계산한다.
        mass_matrix = np.asarray(pin.crba(self.model, self.data, q), dtype=np.float64)
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        # nonLinearEffects: C(q,qdot)qdot + g(q)에 해당하는 항.
        nonlinear_effects = np.asarray(pin.nonLinearEffects(self.model, self.data, q, qdot), dtype=np.float64)
        rhs = tau_total - nonlinear_effects
        regularized_mass = mass_matrix + self.mass_matrix_regularization * np.eye(self.dof)
        qddot = np.linalg.solve(regularized_mass, rhs)
        qddot = np.clip(qddot, -self.max_abs_qddot, self.max_abs_qddot)
        qdot_next = np.clip(qdot + float(dt) * qddot, -self.max_abs_qdot, self.max_abs_qdot)
        # revolute joint angle integration은 단순 q += dq보다 Pinocchio integrate를 쓰는 것이 안전하다.
        q_next = np.asarray(pin.integrate(self.model, q, qdot_next * float(dt)), dtype=np.float64)
        q_next = np.clip(q_next, self.lower_position_limit, self.upper_position_limit)
        return q_next, qdot_next


class FallbackJointSpaceDynamics:
    """Fallback model: six independent second-order joints with nominal gravity."""

    def __init__(self, dof: int) -> None:
        # Pinocchio/URDF가 없을 때만 쓰는 단순 모델이다.
        # 논문 결과는 Pinocchio backend 기준으로 쓰는 것이 맞다.
        self.dof = int(dof)
        self.inertia = np.linspace(0.8, 1.3, self.dof, dtype=np.float64)
        self.gravity_gain = np.asarray([0.0, 2.6, 1.3, 0.45, 0.18, 0.0], dtype=np.float64)[: self.dof]

    def gravity(self, q: np.ndarray) -> np.ndarray:
        return self.gravity_gain * np.sin(q)

    def step(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        tau_cmd_applied: np.ndarray,
        tau_ext: np.ndarray,
        randomization: EpisodeRandomization,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        tau_total = np.asarray(tau_cmd_applied + tau_ext, dtype=np.float64)
        tau_total -= self.gravity(q)
        tau_total -= randomization.damping * qdot
        tau_total -= randomization.viscous_friction * qdot
        tau_total -= randomization.coulomb_friction * np.sign(qdot)
        qddot = tau_total / self.inertia
        qdot_next = qdot + float(dt) * qddot
        q_next = q + float(dt) * qdot_next
        return q_next, qdot_next


def sample_uniform_array(rng: np.random.Generator, bounds: list[float], dof: int) -> np.ndarray:
    low, high = float(bounds[0]), float(bounds[1])
    return rng.uniform(low, high, size=dof)


def sample_randomization(config: dict, rng: np.random.Generator, dof: int) -> EpisodeRandomization:
    # Domain randomization은 sim-to-real gap을 줄이기 위한 장치다.
    # randomized_sim stage에서는 noise/friction/delay/torque scale을 약하게 흔든다.
    domain_cfg = config.get("domain_randomization", {})
    enabled = bool(domain_cfg.get("enabled", True))
    if not enabled:
        return EpisodeRandomization(
            damping=np.zeros(dof, dtype=np.float64),
            viscous_friction=np.zeros(dof, dtype=np.float64),
            coulomb_friction=np.zeros(dof, dtype=np.float64),
            torque_scale=np.ones(dof, dtype=np.float64),
            control_delay_steps=0,
        )

    delay_low, delay_high = domain_cfg.get("control_delay_steps_range", [0, 0])
    return EpisodeRandomization(
        damping=sample_uniform_array(rng, domain_cfg.get("damping_range", [0.0, 0.0]), dof),
        viscous_friction=sample_uniform_array(rng, domain_cfg.get("viscous_friction_range", [0.0, 0.0]), dof),
        coulomb_friction=sample_uniform_array(rng, domain_cfg.get("coulomb_friction_range", [0.0, 0.0]), dof),
        torque_scale=sample_uniform_array(rng, domain_cfg.get("torque_scale_range", [1.0, 1.0]), dof),
        control_delay_steps=int(rng.integers(int(delay_low), int(delay_high) + 1)),
    )


def sample_hold_target(config: dict, rng: np.random.Generator, dof: int) -> np.ndarray:
    # safe_joint_range 안에서 q0/hold target을 뽑아 joint limit이나 특이 자세 근처를 피한다.
    if "safe_joint_range" in config["simulation"]:
        ranges = np.asarray(config["simulation"]["safe_joint_range"], dtype=np.float64)
        if ranges.shape != (dof, 2):
            raise ValueError(f"simulation.safe_joint_range must be [{dof}, 2], got {ranges.shape}")
        return rng.uniform(ranges[:, 0], ranges[:, 1])
    low, high = config["simulation"].get("hold_position_range", [-0.8, 0.8])
    return rng.uniform(float(low), float(high), size=dof)


def sample_sine_profile(config: dict, rng: np.random.Generator, dof: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    # slow_sine trajectory는 center, amplitude, frequency, phase를 episode마다 다르게 뽑는다.
    # 같은 자세만 외우는 overfitting을 줄이기 위한 단계다.
    center = sample_hold_target(config, rng, dof)
    amp_low, amp_high = config["simulation"].get("sine_amplitude_range", [0.05, 0.25])
    freq_low, freq_high = config["simulation"].get("sine_frequency_range", [0.05, 0.30])
    amplitude = rng.uniform(float(amp_low), float(amp_high), size=dof)
    frequency = float(rng.uniform(float(freq_low), float(freq_high)))
    phase = rng.uniform(-math.pi, math.pi, size=dof)
    if "safe_joint_range" in config["simulation"]:
        ranges = np.asarray(config["simulation"]["safe_joint_range"], dtype=np.float64)
        margin = np.minimum(center - ranges[:, 0], ranges[:, 1] - center)
        amplitude = np.minimum(amplitude, np.maximum(0.01, 0.8 * margin))
    return center, amplitude, frequency, phase


def make_trajectory_function(config: dict, rng: np.random.Generator, dof: int, mode: str):
    if mode == "hold":
        q_target = sample_hold_target(config, rng, dof)

        def hold_trajectory(time_s: float) -> tuple[np.ndarray, np.ndarray]:
            del time_s
            return q_target.copy(), np.zeros(dof, dtype=np.float64)

        return hold_trajectory

    if mode == "slow_sine":
        center, amplitude, frequency, phase = sample_sine_profile(config, rng, dof)
        omega = 2.0 * math.pi * frequency

        def slow_sine_trajectory(time_s: float) -> tuple[np.ndarray, np.ndarray]:
            angles = omega * float(time_s) + phase
            q_des = center + amplitude * np.sin(angles)
            qdot_des = amplitude * omega * np.cos(angles)
            return q_des, qdot_des

        return slow_sine_trajectory

    raise ValueError(f"Unsupported simulation mode: {mode}")


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(0.0, 1.0, size=3)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        vector[0] = 1.0
        return vector
    return vector / norm


def sample_joint_torque_event(
    disturbance_cfg: dict,
    rng: np.random.Generator,
    start_step: int,
    end_step: int,
    dof: int,
) -> DisturbanceEvent:
    # 기본 contact approximation: 실제 접촉이 관절에 만들어내는 효과를
    # selected joint disturbance torque로 근사한다. single/multi-joint 모두 랜덤 선택된다.
    torque_low, torque_high = disturbance_cfg.get(
        "joint_torque_range",
        disturbance_cfg.get("magnitude_range", [0.2, 1.5]),
    )
    affected = list(disturbance_cfg.get("affected_joints", list(range(dof))))
    if not affected:
        raise ValueError("disturbance.affected_joints must contain at least one joint index")
    num_joints = int(rng.integers(1, len(affected) + 1))
    joint_subset = rng.choice(affected, size=num_joints, replace=False)
    magnitudes = rng.uniform(float(torque_low), float(torque_high), size=num_joints)
    signs = rng.choice([-1.0, 1.0], size=num_joints)
    tau = np.zeros(dof, dtype=np.float64)
    tau[np.asarray(joint_subset, dtype=np.int64)] = magnitudes * signs
    return DisturbanceEvent(
        start_step=start_step,
        end_step=end_step,
        kind="joint_torque",
        tau=tau,
        region_id=0,
        representative_magnitude=float(np.linalg.norm(tau)),
    )


def sample_body_force_event(
    disturbance_cfg: dict,
    rng: np.random.Generator,
    start_step: int,
    end_step: int,
    dynamics: object,
    frame_region_ids: dict[str, int],
) -> DisturbanceEvent:
    contact_frames = list(disturbance_cfg.get("contact_frames", []))
    if not contact_frames:
        raise ValueError("disturbance.contact_frames is required when using body_force disturbances")
    valid_frames = [str(frame) for frame in contact_frames if getattr(dynamics, "has_frame", lambda _: False)(str(frame))]
    if not valid_frames:
        raise ValueError("None of disturbance.contact_frames exist in the URDF model")
    force_low, force_high = disturbance_cfg.get("force_range", [0.5, 8.0])
    torque_low, torque_high = disturbance_cfg.get("body_torque_range", [0.0, 0.0])
    force_world = random_unit_vector(rng) * float(rng.uniform(float(force_low), float(force_high)))
    torque_world = random_unit_vector(rng) * float(rng.uniform(float(torque_low), float(torque_high)))
    if bool(disturbance_cfg.get("region_balanced_sampling", False)):
        by_region: dict[int, list[str]] = {}
        for frame in valid_frames:
            by_region.setdefault(int(frame_region_ids[frame]), []).append(frame)
        sampled_region = int(rng.choice(sorted(by_region)))
        frame_name = str(rng.choice(by_region[sampled_region]))
    else:
        frame_name = str(rng.choice(valid_frames))
    return DisturbanceEvent(
        start_step=start_step,
        end_step=end_step,
        kind="body_force",
        region_id=int(frame_region_ids.get(frame_name, 0)),
        frame_name=frame_name,
        force_world=force_world,
        torque_world=torque_world,
        representative_magnitude=float(np.linalg.norm(force_world)),
    )


def sample_disturbance_plan(
    config: dict,
    rng: np.random.Generator,
    num_steps: int,
    dt: float,
    dof: int,
    dynamics: object,
) -> list[DisturbanceEvent]:
    # episode 안에 몇 개의 contact event를 넣을지, 언제 시작하고 얼마나 지속될지 정한다.
    # label은 이 plan에서 active인 구간을 기준으로 생성된다.
    disturbance_cfg = config["disturbance"]
    event_low, event_high = disturbance_cfg["num_events_per_episode"]
    duration_low, duration_high = disturbance_cfg["duration_range"]
    num_events = int(rng.integers(int(event_low), int(event_high) + 1))
    disturbance_type = str(disturbance_cfg.get("type", "joint_torque"))
    body_force_probability = float(disturbance_cfg.get("body_force_probability", 0.5))
    can_use_body_force = hasattr(dynamics, "contact_force_to_joint_torque")
    frame_region_ids = frame_to_region_id(config) if can_use_body_force and config.get("contact_regions") else {}
    events: list[DisturbanceEvent] = []

    for _ in range(num_events):
        duration_s = float(rng.uniform(float(duration_low), float(duration_high)))
        duration_steps = max(1, int(round(duration_s / float(dt))))
        start_step = int(rng.integers(0, max(1, num_steps - duration_steps + 1)))
        end_step = min(num_steps, start_step + duration_steps)

        use_body_force = False
        if disturbance_type == "body_force":
            use_body_force = can_use_body_force
        elif disturbance_type == "mixed":
            use_body_force = can_use_body_force and rng.random() < body_force_probability
        elif disturbance_type != "joint_torque":
            raise ValueError("disturbance.type must be one of: joint_torque, body_force, mixed")

        if use_body_force:
            events.append(sample_body_force_event(disturbance_cfg, rng, start_step, end_step, dynamics, frame_region_ids))
        else:
            events.append(sample_joint_torque_event(disturbance_cfg, rng, start_step, end_step, dof))

    return events


def disturbance_tau_at_step(
    events: list[DisturbanceEvent],
    step_idx: int,
    q: np.ndarray,
    dynamics: object,
    dof: int,
) -> tuple[np.ndarray, bool, int, int, float]:
    # 현재 step에 활성화된 모든 disturbance event를 joint torque tau_ext로 합산한다.
    # 반환값 active는 label 생성에 쓰이고, tau는 dynamics.step에 들어간다.
    tau = np.zeros(dof, dtype=np.float64)
    active = False
    region_id = 0
    strongest_norm = 0.0
    strongest_event_id = -1
    strongest_magnitude = 0.0
    for event in events:
        if not (event.start_step <= step_idx < event.end_step):
            continue
        active = True
        if event.kind == "joint_torque":
            if event.tau is not None:
                contribution = np.asarray(event.tau, dtype=np.float64)
                tau += contribution
                contribution_norm = float(np.linalg.norm(contribution))
                if contribution_norm > strongest_norm:
                    strongest_norm = contribution_norm
                    region_id = int(event.region_id)
                    strongest_event_id = int(event.event_id)
                    strongest_magnitude = float(event.representative_magnitude or contribution_norm)
        elif event.kind == "body_force":
            contribution = dynamics.contact_force_to_joint_torque(
                q,
                str(event.frame_name),
                np.asarray(event.force_world, dtype=np.float64),
                np.asarray(event.torque_world, dtype=np.float64),
            )
            tau += contribution
            contribution_norm = float(np.linalg.norm(contribution))
            if contribution_norm > strongest_norm:
                strongest_norm = contribution_norm
                region_id = int(event.region_id)
                strongest_event_id = int(event.event_id)
                strongest_magnitude = float(event.representative_magnitude or contribution_norm)
        else:
            raise ValueError(f"Unsupported disturbance event kind: {event.kind}")
    return tau, active, region_id, strongest_event_id, strongest_magnitude


def sample_disturbance_profile(config: dict, rng: np.random.Generator, num_steps: int, dt: float, dof: int) -> np.ndarray:
    disturbance_cfg = config["disturbance"]
    tau_ext = np.zeros((num_steps, dof), dtype=np.float64)

    event_low, event_high = disturbance_cfg["num_events_per_episode"]
    duration_low, duration_high = disturbance_cfg["duration_range"]
    mag_low, mag_high = disturbance_cfg["magnitude_range"]
    affected = list(disturbance_cfg["affected_joints"])
    num_events = int(rng.integers(int(event_low), int(event_high) + 1))

    for _ in range(num_events):
        num_joints = int(rng.integers(1, len(affected) + 1))
        joint_subset = rng.choice(affected, size=num_joints, replace=False)
        duration_s = float(rng.uniform(float(duration_low), float(duration_high)))
        duration_steps = max(1, int(round(duration_s / float(dt))))
        start_step = int(rng.integers(0, max(1, num_steps - duration_steps + 1)))
        end_step = min(num_steps, start_step + duration_steps)

        disturbance = np.zeros(dof, dtype=np.float64)
        magnitudes = rng.uniform(float(mag_low), float(mag_high), size=num_joints)
        signs = rng.choice([-1.0, 1.0], size=num_joints)
        disturbance[np.asarray(joint_subset, dtype=np.int64)] = magnitudes * signs
        tau_ext[start_step:end_step] += disturbance

    return tau_ext


def choose_backend(config: dict) -> tuple[object, str]:
    # 우선순위: Pinocchio + URDF가 가능하면 실제 로봇 명목 동역학을 사용한다.
    # 불가능할 때만 fallback dynamics를 사용하고 warning을 낸다.
    dof = int(config["robot"]["dof"])
    use_pinocchio = bool(config["robot"].get("use_pinocchio", True))
    urdf_path = resolve_path(config["robot"]["urdf_path"], base_dir=config["_config_dir"])
    joint_names = robot_joint_names(config)

    if use_pinocchio and pin is not None and urdf_path.exists():
        return PinocchioNominalDynamics(str(urdf_path), dof, joint_names), "pinocchio"

    if use_pinocchio and pin is None:
        warnings.warn("Pinocchio is not installed. Falling back to simplified joint-space dynamics.")
    elif use_pinocchio and not urdf_path.exists():
        warnings.warn(f"URDF not found at {urdf_path}. Falling back to simplified joint-space dynamics.")

    return FallbackJointSpaceDynamics(dof), "fallback"


def generate_split(
    split_name: str,
    num_episodes: int,
    config: dict,
    dynamics: object,
    rng: np.random.Generator,
    episode_id_offset: int = 0,
) -> dict[str, np.ndarray]:
    # 하나의 split(train/val/test)을 episode 단위로 생성한다.
    # 이 함수 안에서는 sample 단위 random split을 하지 않는다.
    sim_cfg = config["simulation"]
    domain_cfg = config.get("domain_randomization", {})
    dof = int(config["robot"]["dof"])
    dt = float(sim_cfg["dt"])
    episode_duration = float(sim_cfg["episode_duration"])
    num_steps = int(round(episode_duration / dt))
    if num_steps <= 0:
        raise ValueError(f"episode_duration/dt produced invalid step count: {episode_duration}/{dt}")

    kp = ensure_vector_length("simulation.kp", sim_cfg["kp"], dof)
    kd = ensure_vector_length("simulation.kd", sim_cfg["kd"], dof)
    torque_limit = ensure_vector_length("simulation.torque_limit", sim_cfg.get("torque_limit", [np.inf] * dof), dof)
    q_noise_std = float(domain_cfg.get("q_noise_std", 0.0))
    qdot_noise_std = float(domain_cfg.get("qdot_noise_std", 0.0))
    label_eps = float(sim_cfg.get("disturbance_label_eps", 1.0e-6))
    modes = list(sim_cfg["modes"])

    time_chunks: list[np.ndarray] = []
    q_chunks: list[np.ndarray] = []
    qdot_chunks: list[np.ndarray] = []
    q_des_chunks: list[np.ndarray] = []
    qdot_des_chunks: list[np.ndarray] = []
    tau_cmd_chunks: list[np.ndarray] = []
    tau_cmd_raw_chunks: list[np.ndarray] = []
    tau_ext_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    episode_chunks: list[np.ndarray] = []
    saturated_chunks: list[np.ndarray] = []
    trajectory_mode_chunks: list[np.ndarray] = []
    active_event_id_chunks: list[np.ndarray] = []
    active_event_magnitude_chunks: list[np.ndarray] = []
    event_table_ids: list[int] = []
    event_table_episode_ids: list[int] = []
    event_table_modes: list[str] = []
    event_table_start_steps: list[int] = []
    event_table_end_steps: list[int] = []
    event_table_start_indices: list[int] = []
    event_table_end_indices: list[int] = []
    event_table_magnitudes: list[float] = []
    event_table_kinds: list[str] = []
    next_event_id = 0

    for episode_idx in range(num_episodes):
        # 1) episode마다 trajectory, randomization, disturbance plan을 새로 샘플링한다.
        #    이렇게 해야 특정 자세/특정 외란만 외우는 모델이 되는 것을 줄일 수 있다.
        mode = str(rng.choice(modes))
        trajectory = make_trajectory_function(config, rng, dof, mode)
        randomization = sample_randomization(config, rng, dof)
        disturbance_plan = sample_disturbance_plan(config, rng, num_steps, dt, dof, dynamics)
        split_sample_offset = episode_idx * num_steps
        episode_id_value = episode_id_offset + episode_idx
        for event in disturbance_plan:
            event.event_id = int(next_event_id)
            next_event_id += 1
            event_table_ids.append(int(event.event_id))
            event_table_episode_ids.append(int(episode_id_value))
            event_table_modes.append(str(mode))
            event_table_start_steps.append(int(event.start_step))
            event_table_end_steps.append(int(event.end_step))
            event_table_start_indices.append(int(split_sample_offset + event.start_step))
            event_table_end_indices.append(int(split_sample_offset + event.end_step - 1))
            event_table_magnitudes.append(float(event.representative_magnitude))
            event_table_kinds.append(str(event.kind))

        q = sample_hold_target(config, rng, dof)
        qdot = np.zeros(dof, dtype=np.float64)
        tau_buffer = [np.zeros(dof, dtype=np.float64) for _ in range(randomization.control_delay_steps)]

        time_arr = np.zeros(num_steps, dtype=np.float64)
        q_arr = np.zeros((num_steps, dof), dtype=np.float64)
        qdot_arr = np.zeros((num_steps, dof), dtype=np.float64)
        q_des_arr = np.zeros((num_steps, dof), dtype=np.float64)
        qdot_des_arr = np.zeros((num_steps, dof), dtype=np.float64)
        tau_cmd_arr = np.zeros((num_steps, dof), dtype=np.float64)
        tau_cmd_raw_arr = np.zeros((num_steps, dof), dtype=np.float64)
        tau_ext_arr = np.zeros((num_steps, dof), dtype=np.float64)
        label_arr = np.zeros(num_steps, dtype=np.int64)
        saturated_arr = np.zeros(num_steps, dtype=np.int64)
        trajectory_mode_arr = np.full(num_steps, fill_value=str(mode), dtype="<U16")
        active_event_id_arr = np.full(num_steps, fill_value=-1, dtype=np.int64)
        active_event_magnitude_arr = np.zeros(num_steps, dtype=np.float64)

        for step_idx in range(num_steps):
            # 2) controller가 바라보는 sensor 값을 만든다. q/qdot noise는 여기서 추가된다.
            time_s = step_idx * dt
            q_des, qdot_des = trajectory(time_s)
            q_meas = q + rng.normal(0.0, q_noise_std, size=dof)
            qdot_meas = qdot + rng.normal(0.0, qdot_noise_std, size=dof)
            # 3) commanded torque 정의. 실제 ROS2 online detector도 같은 식으로 tau_cmd를 재구성한다.
            #    tau_cmd_raw는 clipping 전, tau_cmd는 실제 모터 limit을 반영한 command다.
            tau_nominal = dynamics.gravity(q_meas)
            tau_cmd_raw = kp * (q_des - q_meas) + kd * (qdot_des - qdot_meas) + tau_nominal
            tau_cmd = np.clip(tau_cmd_raw, -torque_limit, torque_limit)

            # 4) control delay가 있으면 이전 command를 dynamics에 적용한다.
            tau_buffer.append(tau_cmd.copy())
            delayed_tau_cmd = tau_buffer.pop(0)
            tau_applied = randomization.torque_scale * delayed_tau_cmd
            # 5) 현재 step의 외란 tau_ext를 계산한다. 이 값은 label/dynamics용이지 feature용이 아니다.
            tau_ext_step, disturbance_active, _region_id, strongest_event_id, strongest_event_magnitude = disturbance_tau_at_step(
                disturbance_plan,
                step_idx,
                q,
                dynamics,
                dof,
            )
            q, qdot = dynamics.step(q, qdot, tau_applied, tau_ext_step, randomization, dt)

            # 6) 학습/진단에 필요한 모든 raw signal을 저장한다.
            time_arr[step_idx] = time_s
            q_arr[step_idx] = q_meas
            qdot_arr[step_idx] = qdot_meas
            q_des_arr[step_idx] = q_des
            qdot_des_arr[step_idx] = qdot_des
            tau_cmd_raw_arr[step_idx] = tau_cmd_raw
            tau_cmd_arr[step_idx] = tau_cmd
            tau_ext_arr[step_idx] = tau_ext_step
            saturated_arr[step_idx] = int(np.any(np.abs(tau_cmd_raw - tau_cmd) > 1.0e-9))
            # 7) binary label. 외란이 들어간 전체 구간을 contact state로 둔다.
            label_arr[step_idx] = int(disturbance_active and np.linalg.norm(tau_ext_step) > label_eps)
            active_event_id_arr[step_idx] = int(strongest_event_id)
            active_event_magnitude_arr[step_idx] = float(strongest_event_magnitude)

        episode_ids = np.full(num_steps, fill_value=episode_id_value, dtype=np.int64)

        time_chunks.append(time_arr)
        q_chunks.append(q_arr)
        qdot_chunks.append(qdot_arr)
        q_des_chunks.append(q_des_arr)
        qdot_des_chunks.append(qdot_des_arr)
        tau_cmd_raw_chunks.append(tau_cmd_raw_arr)
        tau_cmd_chunks.append(tau_cmd_arr)
        tau_ext_chunks.append(tau_ext_arr)
        label_chunks.append(label_arr)
        episode_chunks.append(episode_ids)
        saturated_chunks.append(saturated_arr)
        trajectory_mode_chunks.append(trajectory_mode_arr)
        active_event_id_chunks.append(active_event_id_arr)
        active_event_magnitude_chunks.append(active_event_magnitude_arr)

    split = {
        "time": np.concatenate(time_chunks, axis=0),
        "q": np.concatenate(q_chunks, axis=0),
        "qdot": np.concatenate(qdot_chunks, axis=0),
        "q_des": np.concatenate(q_des_chunks, axis=0),
        "qdot_des": np.concatenate(qdot_des_chunks, axis=0),
        "tau_cmd_raw": np.concatenate(tau_cmd_raw_chunks, axis=0),
        "tau_cmd": np.concatenate(tau_cmd_chunks, axis=0),
        "tau_ext": np.concatenate(tau_ext_chunks, axis=0),
        "label": np.concatenate(label_chunks, axis=0),
        "episode_id": np.concatenate(episode_chunks, axis=0),
        "is_saturated": np.concatenate(saturated_chunks, axis=0),
        "trajectory_mode": np.concatenate(trajectory_mode_chunks, axis=0),
        "active_event_id": np.concatenate(active_event_id_chunks, axis=0),
        "active_event_magnitude": np.concatenate(active_event_magnitude_chunks, axis=0),
        "event_table_id": np.asarray(event_table_ids, dtype=np.int64),
        "event_table_episode_id": np.asarray(event_table_episode_ids, dtype=np.int64),
        "event_table_mode": np.asarray(event_table_modes, dtype="<U16"),
        "event_table_start_step": np.asarray(event_table_start_steps, dtype=np.int64),
        "event_table_end_step": np.asarray(event_table_end_steps, dtype=np.int64),
        "event_table_start_index": np.asarray(event_table_start_indices, dtype=np.int64),
        "event_table_end_index": np.asarray(event_table_end_indices, dtype=np.int64),
        "event_table_magnitude": np.asarray(event_table_magnitudes, dtype=np.float64),
        "event_table_kind": np.asarray(event_table_kinds, dtype="<U16"),
        "joint_names": np.asarray(robot_joint_names(config), dtype="<U32"),
    }
    print(
        f"[{split_name}] generated {num_episodes} episodes, {split['time'].shape[0]} samples, "
        f"positive ratio={split['label'].mean():.4f}"
    )
    return split


def save_split(path: str, payload: dict[str, np.ndarray]) -> None:
    # npz 저장 직전에 NaN/Inf를 막는다. 비정상 dynamics가 저장되면 학습 전체가 망가질 수 있다.
    for key, values in payload.items():
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
            raise ValueError(f"Refusing to save split with non-finite values in '{key}'")
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to contact_detection/config.yaml")
    parser.add_argument("--stage", default=None, help="Curriculum stage override.")
    args = parser.parse_args()

    # 실행 순서:
    #   load config -> stage override -> dynamics backend 선택 -> train/val/test split 생성.
    config = load_config(args.config)
    apply_stage_config(config, args.stage)
    validate_motor_joint_map(config)
    set_global_seed(int(config.get("seed", 42)))
    out_dirs = ensure_output_dirs(output_root(config))
    save_config_yaml(out_dirs["root"] / "experiment_config_used.yaml", config)
    dynamics, backend_name = choose_backend(config)

    rng = np.random.default_rng(int(config.get("seed", 42)))
    splits = {
        "sim_train": int(config["simulation"]["num_train_episodes"]),
        "sim_val": int(config["simulation"]["num_val_episodes"]),
        "sim_test": int(config["simulation"]["num_test_episodes"]),
    }

    summary = {
        "backend": backend_name,
        "stage": config["experiment_stage"],
        "joint_names": robot_joint_names(config),
        "dof": int(config["robot"]["dof"]),
        "disturbance_type": str(config.get("disturbance", {}).get("type", "joint_torque")),
        "fallback_note": (
            "Simplified independent joint-space dynamics used because Pinocchio or the URDF was unavailable."
            if backend_name == "fallback"
            else None
        ),
    }

    episode_offset = 0
    episode_sets: dict[str, set[int]] = {}
    for split_name, num_episodes in splits.items():
        # episode_id_offset을 누적해서 train/val/test episode id가 절대 겹치지 않게 한다.
        split_payload = generate_split(split_name, num_episodes, config, dynamics, rng, episode_id_offset=episode_offset)
        episode_ids = set(int(value) for value in np.unique(split_payload["episode_id"]))
        if any(episode_ids & seen for seen in episode_sets.values()):
            raise RuntimeError(f"Episode id leakage detected while generating {split_name}")
        episode_sets[split_name] = episode_ids
        episode_offset += int(num_episodes)
        save_split(out_dirs["datasets"] / f"{split_name}.npz", split_payload)
        saturation_ratio = float(np.mean(split_payload["is_saturated"]))
        if saturation_ratio > 0.05:
            warnings.warn(f"[{split_name}] high torque saturation ratio={saturation_ratio:.3f}")
        summary[split_name] = {
            "num_samples": int(split_payload["time"].shape[0]),
            "positive_ratio": float(split_payload["label"].mean()),
            "saturation_ratio": saturation_ratio,
            "episode_id_min": int(np.min(split_payload["episode_id"])),
            "episode_id_max": int(np.max(split_payload["episode_id"])),
            "trajectory_mode_counts": {
                str(mode_name): int(np.sum(split_payload["trajectory_mode"] == str(mode_name)))
                for mode_name in np.unique(split_payload["trajectory_mode"])
            },
            "num_events": int(split_payload["event_table_id"].shape[0]),
        }

    save_json(out_dirs["metrics"] / "dataset_summary.json", summary)
    print(f"Dataset generation complete using backend='{backend_name}'. Files written to {out_dirs['datasets']}")


if __name__ == "__main__":
    main()
