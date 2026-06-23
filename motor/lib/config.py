"""기본 CAN 채널/호스트 ID 설정."""

HOST_ID = 0xFD      # PC 단독 제어면 나중에 제어하는 곳 여러게 생기면 분리 (ex. 라즈베리파이)
DEFAULT_CH = "can0"

SCAN_MIN_ID = 1
SCAN_MAX_ID = 10

PARAM_ROOT_ENV = "IDLE_PARAM_ROOT"
