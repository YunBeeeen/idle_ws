"""호환용 stop 스크립트(enable.py --enable 0 대체)."""

import argparse
from lib.config import HOST_ID, DEFAULT_CH
from lib.power import send_stop
from lib.runtime import run_with_bus


def _dedupe(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can_id", type=lambda x:int(x,0), nargs="+", required=True)
    ap.add_argument("--clear_fault", type=int, choices=[0,1], default=0)
    ap.add_argument("--ch", default=DEFAULT_CH)
    ap.add_argument("--host_id", type=lambda x:int(x,0), default=HOST_ID)
    args = ap.parse_args()
    can_ids = _dedupe(args.can_id)

    print("deprecated: use enable.py --enable 0")

    def _run(bus):
        for can_id in can_ids:
            arb_id, data = send_stop(
                bus,
                host_id=args.host_id,
                can_id=can_id,
                clear_fault=bool(args.clear_fault),
            )
            print(
                f"sent STOP: can_id={can_id} clear_fault={args.clear_fault} "
                f"arb_id=0x{arb_id:08X} data={data.hex()}"
            )

    run_with_bus(args.ch, _run)

if __name__ == "__main__":
    main()

# 실행 예시 (기본 채널: can0)
# python3 stop.py --can_id 1 --clear_fault 1
# python3 stop.py --can_id 1 2 --clear_fault 1
