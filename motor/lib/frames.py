"""각종 CAN 명령/응답 프레임 생성 함수 모음."""

# lib/frames.py
import math
import struct
from dataclasses import dataclass

from .common import pack_ext_id, float_to_u16

# Communication types (RobStride private protocol manuals section 4.1)
TYPE00_GET_DEVICE_ID      = 0x00
TYPE01_OPERATION_CONTROL  = 0x01  # MIT(Operation control)
TYPE02_FEEDBACK           = 0x02
TYPE03_ENABLE             = 0x03
TYPE04_STOP               = 0x04
TYPE06_SET_ZERO           = 0x06
TYPE07_SET_CAN_ID         = 0x07
TYPE17_PARAM_READ         = 0x11
TYPE18_PARAM_WRITE        = 0x12
TYPE21_FAULT_FEEDBACK     = 0x15
TYPE22_DATA_SAVE          = 0x16
TYPE23_BAUD_SET           = 0x17
TYPE24_ACTIVE_REPORT      = 0x18
TYPE25_PROTOCOL_SET       = 0x19
TYPE26_VERSION_READ       = 0x1A

# Backward-compatible aliases
TYPE01 = TYPE01_OPERATION_CONTROL
TYPE02 = TYPE02_FEEDBACK
TYPE03 = TYPE03_ENABLE
TYPE04 = TYPE04_STOP
TYPE06 = TYPE06_SET_ZERO
TYPE17 = TYPE17_PARAM_READ
TYPE18 = TYPE18_PARAM_WRITE
TYPE22 = TYPE22_DATA_SAVE
TYPE24 = TYPE24_ACTIVE_REPORT


# -----------------------------
# Type01/Type02 operation-control ranges by motor model
# -----------------------------

@dataclass(frozen=True)
class MitRanges:
    P_MIN: float
    P_MAX: float
    V_MIN: float
    V_MAX: float
    T_MIN: float
    T_MAX: float
    KP_MIN: float
    KP_MAX: float
    KD_MIN: float
    KD_MAX: float

MIT_RS02 = MitRanges(
    P_MIN=-4.0 * math.pi, P_MAX= 4.0 * math.pi,
    V_MIN=-44.0,          V_MAX= 44.0,
    T_MIN=-17.0,          T_MAX= 17.0,
    KP_MIN=0.0,           KP_MAX=500.0,
    KD_MIN=0.0,           KD_MAX=5.0,
)

MIT_RS00 = MitRanges(
    P_MIN=-4.0 * math.pi, P_MAX= 4.0 * math.pi,
    V_MIN=-33.0,          V_MAX= 33.0,
    T_MIN=-14.0,          T_MAX= 14.0,
    KP_MIN=0.0,           KP_MAX=500.0,
    KD_MIN=0.0,           KD_MAX=5.0,
)

MIT_RS03 = MitRanges(
    P_MIN=-4.0 * math.pi, P_MAX= 4.0 * math.pi,
    V_MIN=-20.0,          V_MAX= 20.0,
    T_MIN=-60.0,          T_MAX= 60.0,
    KP_MIN=0.0,           KP_MAX=5000.0,
    KD_MIN=0.0,           KD_MAX=100.0,
)

MIT_RS05 = MitRanges(
    P_MIN=-4.0 * math.pi, P_MAX= 4.0 * math.pi,
    V_MIN=-50.0,          V_MAX= 50.0,
    T_MIN=-5.5,           T_MAX= 5.5,
    KP_MIN=0.0,           KP_MAX=500.0,
    KD_MIN=0.0,           KD_MAX=5.0,
)

RS03_ID = 2
MOTOR_MODEL_BY_ID = {
    1: "RS02",
    2: "RS03",
    3: "RS02",
    4: "RS02",
    5: "RS00",
    6: "RS00",
    7: "RS05",
}

MIT_RANGES_BY_MODEL = {
    "RS00": MIT_RS00,
    "RS02": MIT_RS02,
    "RS03": MIT_RS03,
    "RS05": MIT_RS05,
}


def motor_model_for(motor_id: int) -> str:
    return MOTOR_MODEL_BY_ID.get(int(motor_id), "RS02")


def mit_ranges_for(motor_id: int) -> MitRanges:
    return MIT_RANGES_BY_MODEL[motor_model_for(motor_id)]


# -----------------------------
# Power / state frames
# -----------------------------

def frame_type03_enable(host_id: int, motor_id: int) -> tuple[int, bytes]:
    arb_id = pack_ext_id(TYPE03, data2=host_id, data1=motor_id)
    return arb_id, bytes(8)

def frame_type04_stop(host_id: int, motor_id: int, clear_fault: bool = False) -> tuple[int, bytes]:
    arb_id = pack_ext_id(TYPE04, data2=host_id, data1=motor_id)
    data = bytearray(8)
    data[0] = 1 if clear_fault else 0
    return arb_id, bytes(data)

def frame_type06_zero(host_id: int, motor_id: int) -> tuple[int, bytes]:
    arb_id = pack_ext_id(TYPE06, data2=host_id, data1=motor_id)
    data = bytearray(8)
    data[0] = 1
    return arb_id, bytes(data)

def frame_type24_active_report(host_id: int, motor_id: int, enable: bool) -> tuple[int, bytes]:
    arb_id = pack_ext_id(TYPE24, data2=host_id, data1=motor_id)
    # 01 02 03 04 05 06 F_CMD 00
    data = bytes([1, 2, 3, 4, 5, 6, 1 if enable else 0, 0])
    return arb_id, data


# -----------------------------
# Save (Type22)
# -----------------------------

def frame_type22_save(host_id: int, motor_id: int) -> tuple[int, bytes]:
    arb_id = pack_ext_id(TYPE22, data2=host_id, data1=motor_id)
    data = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    return arb_id, data


# -----------------------------
# Param read/write (Type17/18)
# -----------------------------

def frame_type17_read(host_id: int, motor_id: int, index: int) -> tuple[int, bytes]:
    """
    Type17 request:
      data[0:2] = index (little-endian)
      data[2:8] = 0
    """
    arb_id = pack_ext_id(TYPE17, data2=host_id, data1=motor_id)
    data = bytearray(8)
    data[0:2] = int(index).to_bytes(2, "little", signed=False)
    # rest are zeros
    return arb_id, bytes(data)

def frame_type18_write_u32(host_id: int, motor_id: int, index: int, value_u32: int) -> tuple[int, bytes]:
    """
    Type18 write:
      data[0:2] = index (little-endian)
      data[2:4] = 0
      data[4:8] = value_u32 (little-endian)
    """
    arb_id = pack_ext_id(TYPE18, data2=host_id, data1=motor_id)
    data = bytearray(8)
    data[0:2] = int(index).to_bytes(2, "little", signed=False)
    data[2:4] = (0).to_bytes(2, "little", signed=False)
    data[4:8] = int(value_u32 & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
    return arb_id, bytes(data)

def frame_type18_write_f32(host_id: int, motor_id: int, index: int, value_f32: float) -> tuple[int, bytes]:
    raw4 = struct.pack("<f", float(value_f32))
    u32 = int.from_bytes(raw4, "little", signed=False)
    return frame_type18_write_u32(host_id, motor_id, index, u32)

def frame_type18_write_u16(host_id: int, motor_id: int, index: int, value_u16: int) -> tuple[int, bytes]:
    return frame_type18_write_u32(host_id, motor_id, index, int(value_u16) & 0xFFFF)

def frame_type18_write_u8(host_id: int, motor_id: int, index: int, value_u8: int) -> tuple[int, bytes]:
    return frame_type18_write_u32(host_id, motor_id, index, int(value_u8) & 0xFF)


# -----------------------------
# MIT operation control (Type01)
# -----------------------------

def frame_type01_mit(motor_id: int, p: float, v: float, kp: float, kd: float, t: float) -> tuple[int, bytes]:
    """
    Type01 operation control:
      - ID.data2(16bit) = torque u16 mapped by motor model
      - data[0:2]=pos, [2:4]=vel, [4:6]=kp, [6:8]=kd (big-endian)
    """
    r = mit_ranges_for(motor_id)

    tq_u16 = float_to_u16(t, r.T_MIN, r.T_MAX)
    arb_id = pack_ext_id(TYPE01, data2=tq_u16, data1=motor_id)

    p_u16  = float_to_u16(p,  r.P_MIN,  r.P_MAX)
    v_u16  = float_to_u16(v,  r.V_MIN,  r.V_MAX)
    kp_u16 = float_to_u16(kp, r.KP_MIN, r.KP_MAX)
    kd_u16 = float_to_u16(kd, r.KD_MIN, r.KD_MAX)

    data = (
        p_u16.to_bytes(2, "big") +
        v_u16.to_bytes(2, "big") +
        kp_u16.to_bytes(2, "big") +
        kd_u16.to_bytes(2, "big")
    )
    return arb_id, data
