"""socketcan 기반 CAN 버스 래퍼 (send/recv/close 제공)."""

import can

class Bus:
    def __init__(self, channel: str = "can0"):
        self.bus = can.interface.Bus(channel=channel, interface="socketcan")
        self._closed = False

    def send_ext(self, arb_id: int, data: bytes):
        msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)
        self.bus.send(msg)

    def recv(self, timeout: float = 0.5):
        return self.bus.recv(timeout=timeout)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.bus.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
