"""EPScan_time 파라미터를 간편하게 설정하는 유틸."""

import argparse
from lib.config import HOST_ID, DEFAULT_CH
from lib.control_tuning import assert_control_inactive
from lib.frames import frame_type18_write_u32
from lib.common import epscan_from_ms
from lib.runtime import run_with_bus

EPSCAN_INDEX = 0x7026  # EPScan_time


def _dedupe(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can_id", type=lambda x:int(x,0), nargs="+", required=True)
    ap.add_argument("--epscan", type=int, default=None, help="직접 값(매뉴얼 기준 1=10ms, 2=15ms, ...)")
    ap.add_argument("--period_ms", type=int, default=None, help="10,15,20... (자동으로 epscan 계산)")
    ap.add_argument("--ch", default=DEFAULT_CH)
    ap.add_argument("--host_id", type=lambda x:int(x,0), default=HOST_ID)
    args = ap.parse_args()
    assert_control_inactive()
    can_ids = _dedupe(args.can_id)

    if (args.epscan is None) == (args.period_ms is None):
        raise SystemExit("둘 중 하나만 넣어줘: --epscan N  또는  --period_ms 10/15/20...")

    eps = args.epscan if args.epscan is not None else epscan_from_ms(args.period_ms)

    def _run(bus):
        for can_id in can_ids:
            arb_id, data = frame_type18_write_u32(args.host_id, can_id, EPSCAN_INDEX, eps)
            bus.send_ext(arb_id, data)
            print(f"sent EPSCAN: can_id={can_id} epscan={eps} arb_id=0x{arb_id:08X} data={data.hex()}")

    run_with_bus(args.ch, _run)

if __name__ == "__main__":
    main()

# 실행 예시 (기본 채널: can0)
# python3 set_epscan.py --can_id 1 --period_ms 15
# python3 set_epscan.py --can_id 1 --epscan 2
# python3 set_epscan.py --can_id 1 2 --period_ms 15
