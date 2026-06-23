"""fault transition 추적 로직 검증 테스트."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.error_events import EVENT_CHANGED, EVENT_CLEARED, EVENT_RAISED, FaultTransitionTracker
from lib.parse import Feedback


def _fb(fault_bits: int) -> Feedback:
    return Feedback(
        motor_id=1,
        host_id=0xFD,
        mode_status=1,
        fault_bits=fault_bits,
        pos=0.0,
        vel=0.0,
        tor=0.0,
        temp_c=25.0,
    )


class FaultTransitionTrackerTest(unittest.TestCase):
    def test_fault_events_emit_only_on_change(self):
        tracker = FaultTransitionTracker()

        self.assertIsNone(tracker.update_from_feedback(_fb(0)))

        ev1 = tracker.update_from_feedback(_fb(3))
        self.assertIsNotNone(ev1)
        self.assertEqual(ev1.event, EVENT_RAISED)
        self.assertEqual(ev1.prev_fault_bits, 0)
        self.assertEqual(ev1.fault_bits, 3)

        self.assertIsNone(tracker.update_from_feedback(_fb(3)))

        ev2 = tracker.update_from_feedback(_fb(1))
        self.assertIsNotNone(ev2)
        self.assertEqual(ev2.event, EVENT_CHANGED)
        self.assertEqual(ev2.prev_fault_bits, 3)
        self.assertEqual(ev2.fault_bits, 1)

        ev3 = tracker.update_from_feedback(_fb(0))
        self.assertIsNotNone(ev3)
        self.assertEqual(ev3.event, EVENT_CLEARED)
        self.assertEqual(ev3.prev_fault_bits, 1)
        self.assertEqual(ev3.fault_bits, 0)


if __name__ == "__main__":
    unittest.main()
