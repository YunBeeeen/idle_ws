"""고정 MIT(Type01) 명령 스트림을 보내는 테스트용 실행 스크립트."""

import argparse
import time

from lib.control_tuning import control_session
from lib.config import DEFAULT_CH, HOST_ID
from lib.frames import frame_type01_mit
from lib.power import send_stop
from lib.runtime import run_with_bus
from lib.stream_state import clear_mit_stream, publish_mit_stream


def _dedupe(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can_id", type=lambda x: int(x, 0), nargs="+", required=True)
    ap.add_argument("--hz", type=float, default=500.0)
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--v", type=float, default=0.0)
    ap.add_argument("--kp", type=float, default=0.0)
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--ch", default=DEFAULT_CH)
    ap.add_argument("--host_id", type=lambda x: int(x, 0), default=HOST_ID)
    args = ap.parse_args()

    can_ids = _dedupe(args.can_id)
    if not can_ids:
        raise SystemExit("need at least one --can_id")
    if args.hz <= 0:
        raise SystemExit("--hz must be > 0")

    def _run(bus):
        with control_session():
            period = 1.0 / args.hz
            next_t = time.perf_counter()
            hb_due = next_t
            can_id_text = ",".join(str(x) for x in can_ids)
            print(f"MIT stream start: can_id=[{can_id_text}] hz={args.hz}")
            publish_mit_stream(can_ids, hz=args.hz, channel=args.ch, host_id=args.host_id)

            try:
                while True:
                    for can_id in can_ids:
                        arb_id, data = frame_type01_mit(can_id, args.p, args.v, args.kp, args.kd, args.t)
                        bus.send_ext(arb_id, data)

                    now = time.perf_counter()
                    if now >= hb_due:
                        publish_mit_stream(can_ids, hz=args.hz, channel=args.ch, host_id=args.host_id)
                        hb_due = now + 1.0

                    next_t += period
                    dt = next_t - time.perf_counter()
                    if dt > 0:
                        time.sleep(dt)
                    else:
                        next_t = time.perf_counter()
            except KeyboardInterrupt:
                try:
                    for can_id in can_ids:
                        arb_id, data = send_stop(bus, host_id=args.host_id, can_id=can_id, clear_fault=False)
                        print(f"\ninterrupt: sent STOP can_id={can_id} arb_id=0x{arb_id:08X} data={data.hex()}")
                except Exception as exc:
                    print(f"\ninterrupt: failed to send STOP: {exc}")
                raise SystemExit(130)
            finally:
                clear_mit_stream(can_ids)

    run_with_bus(args.ch, _run)


if __name__ == "__main__":
    main()


# 실행 예시 (기본 채널: can0)
# python3 mit_stream.py --can_id 1 --hz 200 --p 0 --v 0 --kp 0 --kd 0 --t 0
# python3 mit_stream.py --can_id 1 2 --hz 200 --p 0 --v 0 --kp 0 --kd 0 --t 0
