"""Enable/Stop 전원 제어 프레임 전송 헬퍼."""

from .bus import Bus
from .frames import frame_type03_enable, frame_type04_stop


def send_power(bus: Bus, host_id: int, can_id: int, enable: bool, clear_fault: bool = False) -> tuple[int, bytes]:
    if enable:
        arb_id, data = frame_type03_enable(host_id, can_id)
    else:
        arb_id, data = frame_type04_stop(host_id, can_id, clear_fault=clear_fault)
    bus.send_ext(arb_id, data)
    return arb_id, data


def send_enable(bus: Bus, host_id: int, can_id: int) -> tuple[int, bytes]:
    return send_power(bus, host_id, can_id, enable=True)


def send_stop(bus: Bus, host_id: int, can_id: int, clear_fault: bool = False) -> tuple[int, bytes]:
    return send_power(bus, host_id, can_id, enable=False, clear_fault=clear_fault)
