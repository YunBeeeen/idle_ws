"""CAN 프레임 보조 유틸(패킹/언패킹, 스케일 변환, 파라미터 응답 대기)."""

# lib/common.py
import math
import struct
import time

TYPE_MASK = 0x1F  # comm type: 5 bits (bit28~24)

def pack_ext_id(comm_type: int, data2: int, data1: int) -> int:
    # bit28~24: type, bit23~8: data2(16), bit7~0: data1(8)
    return ((comm_type & TYPE_MASK) << 24) | ((data2 & 0xFFFF) << 8) | (data1 & 0xFF)

def unpack_ext_id(arb_id: int) -> tuple[int, int, int]:
    comm_type = (arb_id >> 24) & TYPE_MASK
    data2 = (arb_id >> 8) & 0xFFFF
    data1 = arb_id & 0xFF
    return comm_type, data2, data1

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def float_to_u16(x: float, xmin: float, xmax: float) -> int:
    x = clamp(x, xmin, xmax)
    return int(round((x - xmin) * 65535.0 / (xmax - xmin)))

def u16_to_float(u: int, xmin: float, xmax: float) -> float:
    return xmin + (u / 65535.0) * (xmax - xmin)

def epscan_from_ms(period_ms: int, *, zero_based: bool = False) -> int:
    if period_ms < 10 or (period_ms - 10) % 5 != 0:
        raise ValueError("period_ms는 10ms부터 5ms 단위(10,15,20,...)만 지원")
    n = (period_ms - 10) // 5  # manual: 1->10ms, 2->15ms, ...
    return n if zero_based else n + 1


def f32_to_u32_le(x: float) -> int:
    return int.from_bytes(struct.pack("<f", float(x)), "little", signed=False)

def u32_to_f32_le(u: int) -> float:
    return struct.unpack("<f", int(u & 0xFFFFFFFF).to_bytes(4, "little", signed=False))[0]

def encode_value_to_u32(value_str: str, dtype: str) -> int:
    """
    dtype: float|uint32|uint16|uint8
    value_str는 '60', '0x10' 같은 형태 허용
    """
    if dtype == "float":
        return f32_to_u32_le(float(value_str))
    if dtype == "uint32":
        return int(value_str, 0) & 0xFFFFFFFF
    if dtype == "uint16":
        return int(value_str, 0) & 0xFFFF
    if dtype == "uint8":
        return int(value_str, 0) & 0xFF
    raise ValueError(f"unsupported dtype: {dtype}")

def decode_raw4(raw4: bytes, dtype: str):
    """
    Type17 reply의 data[4:8] (4바이트 little-endian)을 dtype으로 해석
    """
    if len(raw4) != 4:
        raise ValueError("raw4 must be 4 bytes")
    if dtype == "float":
        return struct.unpack("<f", raw4)[0]
    if dtype == "uint32":
        return struct.unpack("<I", raw4)[0]
    if dtype == "uint16":
        return struct.unpack("<H", raw4[:2])[0]
    if dtype == "uint8":
        return raw4[0]
    raise ValueError(f"unsupported dtype: {dtype}")

def wait_type17_param_reply(
    bus,
    *,
    index: int,
    timeout: float = 0.6,
    can_id: int | None = None,
    host_id: int | None = None,
) -> tuple[int, bytes]:
    """
    Type17(0x11) 응답을 기다려서 (status, raw4) 반환.
    - status는 ID의 data2 상위바이트(추정). 펌웨어/응답 ID 형식이 달라도 0으로 나오는 경우가 많음.
    - raw4는 data[4:8]
    """
    TYPE17 = 0x11
    want = int(index).to_bytes(2, "little", signed=False)
    deadline = time.time() + timeout

    while time.time() < deadline:
        msg = bus.recv(timeout=max(0.0, deadline - time.time()))
        if msg is None:
            continue
        if len(msg.data) != 8:
            continue

        comm_type, data2, data1 = unpack_ext_id(msg.arbitration_id)
        if comm_type != TYPE17:
            continue
        if bytes(msg.data[0:2]) != want:
            continue

        # 응답 ID가 request와 동일/반대로 오는 케이스가 있어 넉넉히 허용
        if can_id is not None and host_id is not None:
            motor_in = data2 & 0xFF
            id1 = data1 & 0xFF
            if (motor_in not in (can_id, host_id)) and (id1 not in (can_id, host_id)):
                continue

        status = (data2 >> 8) & 0xFF
        raw4 = bytes(msg.data[4:8])
        return status, raw4

    raise TimeoutError(f"Type17 reply timeout (index=0x{index:04X})")
