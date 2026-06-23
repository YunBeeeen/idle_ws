#!/usr/bin/env python3
"""단일 파라미터 Type18 write(+옵션 Type22 save) 유지보수 스크립트."""

import argparse

from lib.config import HOST_ID, DEFAULT_CH
from lib.control_tuning import assert_control_inactive
from lib.runtime import run_with_bus
from lib.frames import frame_type18_write_u32, frame_type22_save, frame_type17_read
from lib.common import (
    encode_value_to_u32,
    wait_type17_param_reply,
    decode_raw4,
    epscan_from_ms,
)
from lib.params import resolve_param, iter_params


def _dedupe(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can_id", type=lambda x: int(x, 0), nargs="+", required=True)
    ap.add_argument("--param", required=True, help="e.g. loc_kp, damper, EPScan_time, or 0x701E")
    ap.add_argument("--ch", default=DEFAULT_CH)
    ap.add_argument("--host_id", type=lambda x: int(x, 0), default=HOST_ID)
    ap.add_argument("--timeout", type=float, default=0.8)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--list", action="store_true")

    # value 입력: 일반은 --value, EPScan_time는 편하게 --ms도 허용
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--value", help="float like 60 or int like 0x10")
    g.add_argument("--ms", type=int, help="EPScan_time용: 10,15,20,... (ms)")

    args = ap.parse_args()
    assert_control_inactive()
    can_ids = _dedupe(args.can_id)

    if args.list:
        for k, v in iter_params():
            print(f"{k:12s} index=0x{v.index:04X} type={v.dtype:6s} access={v.access}")
        return

    pdef = resolve_param(args.param)
    if pdef.access == "R":
        raise SystemExit(f"param '{args.param}' (0x{pdef.index:04X}) is read-only")

    # value_u32 생성
    if args.ms is not None:
        if args.param != "EPScan_time" and pdef.index != 0x7026:
            raise SystemExit("--ms는 EPScan_time에서만 사용 가능")
        # 매뉴얼 규칙(1->10ms, 2->15ms, ...)이면 one-based가 자연스러워서 zero_based=False 사용
        value_u32 = epscan_from_ms(args.ms, zero_based=False)
        value_repr = f"{args.ms}ms -> {value_u32}"
    else:
        value_u32 = encode_value_to_u32(args.value, pdef.dtype)
        value_repr = args.value

    def _run(bus):
        for can_id in can_ids:
            # 1) write
            arb_id, data = frame_type18_write_u32(args.host_id, can_id, pdef.index, value_u32)
            bus.send_ext(arb_id, data)
            print(
                f"sent WRITE(Type18): can_id={can_id} host_id=0x{args.host_id:02X} "
                f"param={args.param} index=0x{pdef.index:04X} type={pdef.dtype} value={value_repr} "
                f"arb_id=0x{arb_id:08X} data={data.hex()}"
            )

            # 2) verify (Type17 readback)
            if args.verify:
                arb_id_r, data_r = frame_type17_read(args.host_id, can_id, pdef.index)
                bus.send_ext(arb_id_r, data_r)
                try:
                    status, raw4 = wait_type17_param_reply(
                        bus, index=pdef.index, timeout=args.timeout, can_id=can_id, host_id=args.host_id
                    )
                    st = "OK" if status == 0 else f"STATUS={status}"
                    val = decode_raw4(raw4, pdef.dtype)
                    print(f"verify READ(Type17): can_id={can_id} [{st}] value={val} raw={raw4.hex()}")
                except TimeoutError:
                    print(f"verify READ(Type17): can_id={can_id} [TIMEOUT]")

            # 3) save
            if not args.no_save:
                arb_id_s, data_s = frame_type22_save(args.host_id, can_id)
                bus.send_ext(arb_id_s, data_s)
                print(f"sent SAVE(Type22): can_id={can_id} arb_id=0x{arb_id_s:08X} data={data_s.hex()}")

    run_with_bus(args.ch, _run)

if __name__ == "__main__":
    main()

# 실행 예시 (기본 채널: can0)
# python3 param_write_save.py --can_id 1 --param loc_kp --value 10 --verify
# python3 param_write_save.py --can_id 1 --param EPScan_time --ms 15
# python3 param_write_save.py --can_id 1 2 --param EPScan_time --ms 15 --verify
