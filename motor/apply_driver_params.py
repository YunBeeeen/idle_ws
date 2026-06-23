"""driver_params.yaml(Type18) 내용을 모터에 적용하는 유지보수 스크립트."""

import argparse
import time

from lib.common import decode_raw4, encode_value_to_u32, wait_type17_param_reply
from lib.config import DEFAULT_CH, HOST_ID
from lib.control_tuning import assert_control_inactive
from lib.frames import frame_type17_read, frame_type18_write_u32, frame_type22_save
from lib.param_store import load_driver_original
from lib.params import resolve_param
from lib.runtime import run_with_bus


def _dedupe(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def _motor_ids_from_doc(doc: dict) -> list[int]:
    motors = doc.get("motors", {})
    if not isinstance(motors, dict):
        return []
    out: list[int] = []
    for key in motors.keys():
        try:
            out.append(int(str(key), 10))
        except Exception:
            continue
    return sorted(set(out))


def apply_driver_params(
    bus,
    *,
    doc: dict,
    can_ids: list[int],
    host_id: int,
    verify: bool,
    timeout: float,
    save: bool,
    sleep_ms: float,
) -> int:
    motors = doc.get("motors", {})
    if not isinstance(motors, dict):
        raise ValueError("driver params must contain dict 'motors'")

    total = 0
    for can_id in can_ids:
        raw_cfg = motors.get(str(can_id))
        if not isinstance(raw_cfg, dict) or not raw_cfg:
            print(f"[skip] can_id={can_id}: no driver params")
            continue

        wrote = 0
        for name, value in raw_cfg.items():
            pdef = resolve_param(str(name))
            if pdef.access == "R":
                print(f"[skip] can_id={can_id} param={name} index=0x{pdef.index:04X} read-only")
                continue

            value_u32 = encode_value_to_u32(str(value), pdef.dtype)
            arb_id, data = frame_type18_write_u32(host_id, can_id, pdef.index, value_u32)
            bus.send_ext(arb_id, data)
            wrote += 1
            total += 1
            print(
                f"[write] can_id={can_id} param={name} index=0x{pdef.index:04X} "
                f"type={pdef.dtype} value={value} arb_id=0x{arb_id:08X}"
            )
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

            if verify:
                arb_r, data_r = frame_type17_read(host_id, can_id, pdef.index)
                bus.send_ext(arb_r, data_r)
                try:
                    status, raw4 = wait_type17_param_reply(
                        bus,
                        index=pdef.index,
                        timeout=timeout,
                        can_id=can_id,
                        host_id=host_id,
                    )
                    decoded = decode_raw4(raw4, pdef.dtype)
                    print(
                        f"  [verify] can_id={can_id} index=0x{pdef.index:04X} "
                        f"status=0x{status:02X} value={decoded} raw={raw4.hex()}"
                    )
                except TimeoutError:
                    print(f"  [verify] can_id={can_id} index=0x{pdef.index:04X} timeout")

        if wrote > 0 and save:
            arb_s, data_s = frame_type22_save(host_id, can_id)
            bus.send_ext(arb_s, data_s)
            print(f"[save] can_id={can_id} arb_id=0x{arb_s:08X} data={data_s.hex()}")
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    return total


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Apply driver params(Type18) from param/original/driver_params.yaml. "
            "This is a maintenance tool and only runs on explicit request."
        )
    )
    ap.add_argument("--can_id", type=lambda x: int(x, 0), nargs="+", default=None)
    ap.add_argument("--ch", default=DEFAULT_CH)
    ap.add_argument("--host_id", type=lambda x: int(x, 0), default=HOST_ID)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--timeout", type=float, default=0.6)
    ap.add_argument("--save", action="store_true", help="send Type22 save per motor after write")
    ap.add_argument("--sleep_ms", type=float, default=2.0, help="delay between write frames")
    args = ap.parse_args()

    assert_control_inactive()
    doc = load_driver_original()

    can_ids = _dedupe(args.can_id) if args.can_id else _motor_ids_from_doc(doc)
    if not can_ids:
        raise SystemExit("no target motor id: use --can_id or define motors in driver_params.yaml")

    def _run(bus):
        total = apply_driver_params(
            bus,
            doc=doc,
            can_ids=can_ids,
            host_id=args.host_id,
            verify=args.verify,
            timeout=args.timeout,
            save=args.save,
            sleep_ms=args.sleep_ms,
        )
        print(f"done: total_writes={total} target_can_id={can_ids}")

    run_with_bus(args.ch, _run)


if __name__ == "__main__":
    main()
