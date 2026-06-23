"""Bus 생성/종료를 감싸는 실행 헬퍼."""

from typing import Callable, TypeVar

from .bus import Bus

T = TypeVar("T")


def run_with_bus(channel: str, fn: Callable[[Bus], T]) -> T:
    bus = Bus(channel)
    try:
        return fn(bus)
    finally:
        bus.close()
