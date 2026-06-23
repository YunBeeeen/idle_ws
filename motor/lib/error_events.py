"""Type02 fault transition tracker."""

from __future__ import annotations

from dataclasses import dataclass

from .parse import Feedback

EVENT_RAISED = 1
EVENT_CLEARED = 2
EVENT_CHANGED = 3


@dataclass(frozen=True)
class MotorErrorEvent:
    motor_id: int
    mode_status: int
    prev_fault_bits: int
    fault_bits: int
    event: int


class FaultTransitionTracker:
    def __init__(self):
        self._fault_by_motor: dict[int, int] = {}

    def update_from_feedback(self, fb: Feedback) -> MotorErrorEvent | None:
        motor_id = int(fb.motor_id)
        prev = int(self._fault_by_motor.get(motor_id, 0))
        cur = int(fb.fault_bits) & 0x3F
        self._fault_by_motor[motor_id] = cur

        if prev == cur:
            return None

        if prev == 0 and cur != 0:
            event = EVENT_RAISED
        elif prev != 0 and cur == 0:
            event = EVENT_CLEARED
        else:
            event = EVENT_CHANGED
        return MotorErrorEvent(
            motor_id=motor_id,
            mode_status=int(fb.mode_status),
            prev_fault_bits=prev,
            fault_bits=cur,
            event=event,
        )
