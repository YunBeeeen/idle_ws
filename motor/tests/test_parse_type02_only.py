"""Type02 전용 파싱 정책 검증 테스트."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.common import pack_ext_id
from lib.parse import parse_feedback_like_type2


class ParseType02OnlyTest(unittest.TestCase):
    def test_parse_accepts_type02_only(self):
        motor_id = 1
        host_id = 0xFD
        flags = (1 << 6) | 0x03
        data2 = ((flags & 0xFF) << 8) | motor_id
        data = bytes.fromhex("000100020003012c")

        arb_type02 = pack_ext_id(0x02, data2=data2, data1=host_id)
        fb = parse_feedback_like_type2(arb_type02, data)
        self.assertIsNotNone(fb)
        self.assertEqual(fb.motor_id, motor_id)

        arb_type24 = pack_ext_id(0x18, data2=data2, data1=host_id)
        self.assertIsNone(parse_feedback_like_type2(arb_type24, data))


if __name__ == "__main__":
    unittest.main()
