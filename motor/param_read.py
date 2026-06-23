#!/usr/bin/env python3
"""Type17로 드라이버 파라미터를 읽어오는 조회 스크립트."""

import argparse

from lib.config import HOST_ID, DEFAULT_CH
from lib.runtime import run_with_bus
from lib.frames import frame_type17_read
from lib.common import wait_type17_param_reply, decode_raw4
from lib.params import resolve_param, iter_params


def _dedupe(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can_id", type=lambda x: int(x, 0), nargs="+", required=True)
    ap.add_argument("--param", help="e.g. q_vel, loc_kp, EPScan_time, or 0x701E")
    ap.add_argument("--ch", default=DEFAULT_CH)
    ap.add_argument("--host_id", type=lambda x: int(x, 0), default=HOST_ID)
    ap.add_argument("--timeout", type=float, default=0.6)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    can_ids = _dedupe(args.can_id)

    if args.list:
        for k, v in iter_params():
            print(f"{k:12s} index=0x{v.index:04X} type={v.dtype:6s} access={v.access}")
        return
    if not args.param:
        raise SystemExit("need --param (or use --list)")

    pdef = resolve_param(args.param)

    def _run(bus):
        for can_id in can_ids:
            arb_id, data = frame_type17_read(args.host_id, can_id, pdef.index)
            bus.send_ext(arb_id, data)

            try:
                status, raw4 = wait_type17_param_reply(
                    bus, index=pdef.index, timeout=args.timeout, can_id=can_id, host_id=args.host_id
                )
            except TimeoutError:
                print(
                    f"[TIMEOUT] can_id={can_id} host_id=0x{args.host_id:02X} "
                    f"param={args.param} index=0x{pdef.index:04X}"
                )
                continue

            val = decode_raw4(raw4, pdef.dtype)
            st = "OK" if status == 0 else f"STATUS={status}"
            print(
                f"[{st}] can_id={can_id} host_id=0x{args.host_id:02X} "
                f"param={args.param} index=0x{pdef.index:04X} type={pdef.dtype} value={val} raw={raw4.hex()}"
            )

    run_with_bus(args.ch, _run)

if __name__ == "__main__":
    main()

# 실행 예시 (기본 채널: can0)
# python3 param_read.py --can_id 1 --param q_vel
# python3 param_read.py --can_id 1 2 --param q_vel
# python3 param_read.py --can_id 1 --list
