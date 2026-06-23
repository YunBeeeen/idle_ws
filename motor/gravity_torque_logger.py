"""Type24(active report) 기반 중력보상 토크 로거 (제어 송신 없음).

- 입력: Type24 피드백(일부 펌웨어의 Type02 유사 payload)
- 출력: 모터별 q / tau_g / measured_tor 텍스트 로그
- 송신: 시작 시 Type24 active report enable 1회(옵션으로 비활성화 가능)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lib.common import unpack_ext_id
from lib.config import DEFAULT_CH, HOST_ID
from lib.frames import TYPE24, frame_type24_active_report
from lib.parse import parse_feedback_like_type2
from lib.runtime import run_with_bus


DEFAULT_URDF_REL = "src/sim/urdf/robot.urdf"


def _dedupe(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def _arb_id_with_comm_type(arb_id: int, comm_type: int) -> int:
    return ((comm_type & 0x1F) << 24) | (arb_id & 0x00FFFFFF)


def _looks_type24_active_report_cmd(data: bytes) -> bool:
    # Host -> motor 설정 프레임: 01 02 03 04 05 06 {0|1} 00
    return (
        len(data) == 8
        and data[0:6] == bytes((1, 2, 3, 4, 5, 6))
        and data[6] in (0, 1)
        and data[7] == 0
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_urdf(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return p
    return (_repo_root() / p).resolve()


def _parse_motor_joint_map_json(text: str) -> dict[int, str]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"mapping parse error: invalid --motor_joint_map_json JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise SystemExit("mapping parse error: --motor_joint_map_json must be a JSON object")

    parsed: dict[int, str] = {}
    for k, v in raw.items():
        try:
            motor_id = int(k)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"mapping parse error: invalid motor key '{k}'") from exc
        if not isinstance(v, str) or not v.strip():
            raise SystemExit(f"mapping parse error: motor {motor_id} joint name must be non-empty string")
        parsed[motor_id] = v.strip()
    return parsed


def _parse_sign_by_motor_json(text: str) -> dict[int, float]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"mapping parse error: invalid --sign_by_motor_json JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise SystemExit("mapping parse error: --sign_by_motor_json must be a JSON object")

    parsed: dict[int, float] = {}
    for k, v in raw.items():
        try:
            motor_id = int(k)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"mapping parse error: invalid sign key '{k}'") from exc
        try:
            sign = float(v)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"mapping parse error: motor {motor_id} sign must be numeric") from exc
        parsed[motor_id] = sign
    return parsed


def _load_pinocchio_or_exit():
    try:
        import pinocchio as pin  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"pinocchio import failed: {exc}") from exc
    return pin


def _build_model_or_exit(pin, urdf_path: Path):
    if not urdf_path.exists():
        raise SystemExit(f"URDF load failed: file not found: {urdf_path}")
    try:
        model = pin.buildModelFromUrdf(str(urdf_path))
    except Exception as exc:
        raise SystemExit(f"URDF load failed: {exc}") from exc
    return model


def _resolve_joint_indices_or_exit(model, motor_joint_map: dict[int, str], can_ids: list[int]) -> dict[int, tuple[str, int]]:
    result: dict[int, tuple[str, int]] = {}
    for motor_id in can_ids:
        joint_name = motor_joint_map[motor_id]
        if not model.existJointName(joint_name):
            raise SystemExit(f"joint mapping error: joint '{joint_name}' not found in URDF")
        jid = model.getJointId(joint_name)
        joint = model.joints[jid]
        if joint.nq != 1 or joint.nv != 1:
            raise SystemExit(
                f"joint mapping error: joint '{joint_name}' must be 1-DoF (nq={joint.nq}, nv={joint.nv})"
            )
        result[motor_id] = (joint_name, joint.idx_q)
    return result


def _format_map_text(mapping: dict[int, object], keys: list[int]) -> str:
    return "{" + ", ".join(f"{k}:{mapping[k]}" for k in keys) + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can_id", type=lambda x: int(x, 0), nargs="+", default=[1, 2])
    ap.add_argument("--ch", default=DEFAULT_CH)
    ap.add_argument("--host_id", type=lambda x: int(x, 0), default=HOST_ID)
    ap.add_argument("--urdf", default=DEFAULT_URDF_REL)
    ap.add_argument("--motor_joint_map_json", default='{"1":"j1","2":"j2"}')
    ap.add_argument("--sign_by_motor_json", default='{"1":1,"2":1}')
    ap.add_argument("--log_hz", type=float, default=5.0)
    ap.add_argument("--print_header", dest="print_header", action="store_true", default=True)
    ap.add_argument("--no_print_header", dest="print_header", action="store_false")
    ap.add_argument("--no_enable_active_report", action="store_true")
    args = ap.parse_args()

    can_ids = _dedupe(args.can_id)
    if not can_ids:
        raise SystemExit("need at least one --can_id")
    if args.log_hz <= 0:
        raise SystemExit("--log_hz must be > 0")

    motor_joint_map = _parse_motor_joint_map_json(args.motor_joint_map_json)
    sign_by_motor_raw = _parse_sign_by_motor_json(args.sign_by_motor_json)

    missing_joint_map = [cid for cid in can_ids if cid not in motor_joint_map]
    if missing_joint_map:
        missing_text = ",".join(str(x) for x in missing_joint_map)
        raise SystemExit(f"joint mapping error: missing motor ids in --motor_joint_map_json: {missing_text}")

    sign_by_motor: dict[int, float] = {cid: sign_by_motor_raw.get(cid, 1.0) for cid in can_ids}

    pin = _load_pinocchio_or_exit()
    urdf_path = _resolve_urdf(args.urdf)
    model = _build_model_or_exit(pin, urdf_path)
    data = model.createData()
    joint_idx_by_motor = _resolve_joint_indices_or_exit(model, motor_joint_map, can_ids)

    # URDF joint j1~j4를 기본 가정으로 0 초기화. (모델 정의가 바뀌어도 관절 인덱스로 채움)
    q_latest = np.zeros(model.nq)
    measured_tor_by_motor: dict[int, float] = {}
    got_feedback: dict[int, bool] = {cid: False for cid in can_ids}

    can_filter = set(can_ids)
    period = 1.0 / args.log_hz
    now = time.perf_counter()
    next_log_tp_by_motor = {cid: now for cid in can_ids}

    if args.print_header:
        print(
            f"gravity_torque_logger start ch={args.ch} host_id=0x{args.host_id:02X} "
            f"can_id={can_ids} urdf={urdf_path} pinocchio={pin.__version__} "
            f"log_hz={args.log_hz:.3f} enable_active_report={not args.no_enable_active_report}"
        )
        print(
            "gravity_torque_logger map "
            f"motor_joint={_format_map_text({k: joint_idx_by_motor[k][0] for k in can_ids}, can_ids)} "
            f"sign={_format_map_text(sign_by_motor, can_ids)}"
        )

    def _emit_logs_if_due() -> None:
        t_now = time.perf_counter()
        due_motors = [cid for cid in can_ids if got_feedback[cid] and t_now >= next_log_tp_by_motor[cid]]
        if not due_motors:
            return

        tau_g = pin.computeGeneralizedGravity(model, data, q_latest)
        wall_ts = time.time()
        for motor_id in due_motors:
            joint_name, joint_q_idx = joint_idx_by_motor[motor_id]
            tau_joint = float(tau_g[joint_q_idx])
            tau_out = sign_by_motor[motor_id] * tau_joint
            q = float(q_latest[joint_q_idx])
            measured_tor = float(measured_tor_by_motor.get(motor_id, 0.0))
            print(
                f"ts={wall_ts:.3f} can_id={motor_id} joint={joint_name} "
                f"q={q:+.6f} tau_g={tau_out:+.6f} measured_tor={measured_tor:+.6f}"
            )
            # 밀린 경우에도 주기를 고정하기 위해 누적 증가
            while next_log_tp_by_motor[motor_id] <= t_now:
                next_log_tp_by_motor[motor_id] += period

    def _run(bus):
        if not args.no_enable_active_report:
            for can_id in can_ids:
                arb_id, payload = frame_type24_active_report(args.host_id, can_id, enable=True)
                bus.send_ext(arb_id, payload)
                print(
                    f"sent ACTIVE_REPORT enable=1 can_id={can_id} "
                    f"arb_id=0x{arb_id:08X} data={payload.hex()}"
                )

        while True:
            active_due_times = [next_log_tp_by_motor[cid] for cid in can_ids if got_feedback[cid]]
            if active_due_times:
                soonest_due = min(active_due_times)
                timeout = max(0.0, min(0.2, soonest_due - time.perf_counter()))
            else:
                timeout = 0.2
            msg = bus.recv(timeout=timeout)

            if msg is not None:
                comm_type, _, _ = unpack_ext_id(msg.arbitration_id)
                if comm_type == TYPE24:
                    raw = bytes(msg.data)
                    if _looks_type24_active_report_cmd(raw):
                        _emit_logs_if_due()
                        continue

                    # Type24 payload를 Type02 피드백 포맷으로 재해석
                    fb = parse_feedback_like_type2(_arb_id_with_comm_type(msg.arbitration_id, 0x02), raw)
                    if fb is not None and fb.motor_id in can_filter:
                        _, joint_q_idx = joint_idx_by_motor[fb.motor_id]
                        q_latest[joint_q_idx] = fb.pos
                        measured_tor_by_motor[fb.motor_id] = fb.tor
                        got_feedback[fb.motor_id] = True

            _emit_logs_if_due()

    try:
        run_with_bus(args.ch, _run)
    except KeyboardInterrupt:
        print("\ninterrupt: gravity_torque_logger stopped")


if __name__ == "__main__":
    main()


# 실행 예시 (기본 채널: can0)
# python3 motor/gravity_torque_logger.py
# python3 motor/gravity_torque_logger.py --can_id 1 2 --log_hz 5
# python3 motor/gravity_torque_logger.py --motor_joint_map_json '{"1":"j1","2":"j2"}' --sign_by_motor_json '{"1":1,"2":-1}'
# python3 motor/gravity_torque_logger.py --no_enable_active_report
