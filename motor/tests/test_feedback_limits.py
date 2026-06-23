import math
import unittest

from lib.limits import feedback_position_violation, position_limits_for_motor
from lib.parse import Feedback


class TestFeedbackLimits(unittest.TestCase):
    def test_motor4_uses_robot_pi_limit(self):
        q_min, q_max = position_limits_for_motor(4)
        self.assertAlmostEqual(q_min, -math.pi)
        self.assertAlmostEqual(q_max, math.pi)

    def test_motor4_warns_when_feedback_exceeds_pi(self):
        fb = Feedback(
            motor_id=4,
            host_id=0xFD,
            mode_status=2,
            fault_bits=0,
            pos=4.10,
            vel=0.0,
            tor=0.0,
            temp_c=25.0,
        )

        violation = feedback_position_violation(fb)

        self.assertIsNotNone(violation)
        q_limited, q_min, q_max = violation
        self.assertAlmostEqual(q_limited, math.pi)
        self.assertAlmostEqual(q_min, -math.pi)
        self.assertAlmostEqual(q_max, math.pi)

    def test_motor4_does_not_warn_inside_pi(self):
        fb = Feedback(
            motor_id=4,
            host_id=0xFD,
            mode_status=2,
            fault_bits=0,
            pos=1.57,
            vel=0.0,
            tor=0.0,
            temp_c=25.0,
        )

        self.assertIsNone(feedback_position_violation(fb))


if __name__ == "__main__":
    unittest.main()
