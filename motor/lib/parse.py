"""Type02 피드백 프레임 파서."""

# lib/parse.py (핵심만)

from dataclasses import dataclass
from .common import unpack_ext_id, u16_to_float
from .frames import mit_ranges_for

@dataclass
class Feedback:
    motor_id: int
    host_id: int
    mode_status: int
    fault_bits: int
    pos: float
    vel: float
    tor: float
    temp_c: float

def parse_feedback_like_type2(arb_id: int, data: bytes) -> Feedback | None:
    comm_type, data2, data1 = unpack_ext_id(arb_id)

    if comm_type != 0x02 or len(data) != 8:
        return None

    motor_id = data2 & 0xFF
    spec = mit_ranges_for(motor_id)

    flags = (data2 >> 8) & 0xFF
    mode_status = (flags >> 6) & 0x03
    fault_bits  = flags & 0x3F

    p_u16 = int.from_bytes(data[0:2], "big")
    v_u16 = int.from_bytes(data[2:4], "big")
    t_u16 = int.from_bytes(data[4:6], "big")
    temp_u16 = int.from_bytes(data[6:8], "big")

    pos = u16_to_float(p_u16, spec.P_MIN, spec.P_MAX)
    vel = u16_to_float(v_u16, spec.V_MIN, spec.V_MAX)
    tor = u16_to_float(t_u16, spec.T_MIN, spec.T_MAX)
    temp_c = temp_u16 / 10.0

    return Feedback(motor_id, data1, mode_status, fault_bits, pos, vel, tor, temp_c)
