"""apply_driver_params 유틸 동작 검증 테스트."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apply_driver_params import apply_driver_params
from lib.common import unpack_ext_id


class _FakeBus:
    def __init__(self):
        self.sent: list[tuple[int, bytes]] = []

    def send_ext(self, arb_id: int, data: bytes):
        self.sent.append((arb_id, bytes(data)))


class ApplyDriverParamsTest(unittest.TestCase):
    def test_apply_driver_params_sends_type18_and_save(self):
        bus = _FakeBus()
        doc = {
            "version": 1,
            "motors": {
                "1": {
                    "run_mode": 0,
                    "EPScan_time": 2,
                }
            },
        }

        total = apply_driver_params(
            bus,
            doc=doc,
            can_ids=[1],
            host_id=0xFD,
            verify=False,
            timeout=0.1,
            save=True,
            sleep_ms=0.0,
        )
        self.assertEqual(total, 2)
        self.assertEqual(len(bus.sent), 3)

        types = [unpack_ext_id(arb_id)[0] for arb_id, _ in bus.sent]
        self.assertEqual(types[:2], [0x12, 0x12])
        self.assertEqual(types[2], 0x16)


if __name__ == "__main__":
    unittest.main()
