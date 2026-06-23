"""모터 모델별 Type01/Type02 스케일 매핑 검증."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.common import epscan_from_ms, pack_ext_id
from lib.frames import (
    MIT_RS00,
    MIT_RS02,
    MIT_RS03,
    MIT_RS05,
    frame_type01_mit,
    mit_ranges_for,
    motor_model_for,
)
from lib.parse import parse_feedback_like_type2


class MotorModelRangesTest(unittest.TestCase):
    def test_motor_id_to_model_and_range(self):
        self.assertEqual(motor_model_for(1), "RS02")
        self.assertEqual(motor_model_for(2), "RS03")
        self.assertEqual(motor_model_for(5), "RS00")
        self.assertEqual(motor_model_for(7), "RS05")
        self.assertIs(mit_ranges_for(1), MIT_RS02)
        self.assertIs(mit_ranges_for(2), MIT_RS03)
        self.assertIs(mit_ranges_for(5), MIT_RS00)
        self.assertIs(mit_ranges_for(7), MIT_RS05)

    def test_type01_uses_rs05_torque_range(self):
        arb_id, _ = frame_type01_mit(7, p=0.0, v=0.0, kp=0.0, kd=0.0, t=5.5)
        self.assertEqual((arb_id >> 8) & 0xFFFF, 0xFFFF)

    def test_type02_parse_uses_rs00_and_rs05_ranges(self):
        data = bytes.fromhex("80008000ffff012c")

        fb_rs00 = parse_feedback_like_type2(pack_ext_id(0x02, data2=5, data1=0xFD), data)
        self.assertIsNotNone(fb_rs00)
        self.assertAlmostEqual(fb_rs00.tor, 14.0)
        self.assertAlmostEqual(fb_rs00.vel, 0.0, delta=0.001)
        self.assertAlmostEqual(fb_rs00.pos, 0.0, delta=0.001)
        self.assertAlmostEqual(fb_rs00.temp_c, 30.0)

        fb_rs05 = parse_feedback_like_type2(pack_ext_id(0x02, data2=7, data1=0xFD), data)
        self.assertIsNotNone(fb_rs05)
        self.assertAlmostEqual(fb_rs05.tor, 5.5)

    def test_epscan_period_ms_uses_manual_one_based_value(self):
        self.assertEqual(epscan_from_ms(10), 1)
        self.assertEqual(epscan_from_ms(15), 2)
        self.assertEqual(epscan_from_ms(20), 3)


if __name__ == "__main__":
    unittest.main()
