# can_interface

CAN 버스 ↔ ROS 토픽 브리지. MIT 확장 ID 프로토콜을 사용하는 모터 드라이버와 통신.

## 노드

| 노드 | 역할 |
|------|------|
| `can_bridge_node` (C++) | CAN ↔ `/motor_cmd_array`, `/motor_state_array`, `/motor_error_array` 변환 |

## 빠른 사용법

```bash
# CAN 인터페이스 활성화 (한 번만)
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 빌드
colcon build --packages-select msgs can_interface
source install/setup.bash

# 실행
ros2 run can_interface can_bridge_node
```

## 시작 시퀀스 (요약)

1. `Type00` 스캔으로 모터 ID 1-10 검색
2. 발견된 모터에 `Type03` enable 1회 전송
3. `Type02` 피드백 관측되면 `ready` 상태로 전환
4. ready된 모터부터 home 위치로 천천히 이동 (5초)
5. 외부 `/motor_cmd_array` 도착하면 home 명령을 대체

상세 동작 문서는 [src/README.md](src/README.md) 참조.

## 정적 설정 변경

이 노드의 설정은 **ROS parameter가 아닌 C++ 코드 상수**입니다. 변경하려면:

1. [src/can_bridge_node.cpp](src/can_bridge_node.cpp) 상단의 상수 수정
2. `colcon build --packages-select can_interface`
3. 노드 재시작

대표적 상수:
- `kMotorHomeByMotor` — 모터별 home 위치, kp, kd, 관절 한계
- `kTxHzDefault` — 송신 주파수 [Hz]
- `kScanMinId`/`kScanMaxId` — 스캔 ID 범위

## 토픽

| 토픽 | 메시지 | 방향 |
|------|--------|------|
| `/motor_cmd_array` | `msgs/MotorCMDArray` | 구독 |
| `/motor_state_array` | `msgs/MotorStateArray` | 발행 |
| `/motor_error_array` | `msgs/MotorErrorArray` | 발행 |

## 의존성

- `rclcpp`
- `msgs` (커스텀 메시지)
- Linux SocketCAN
- C++17

## 트러블슈팅

### 모터가 안 잡혀요

```bash
# CAN 인터페이스 상태 확인
ip -details link show can0

# CAN 트래픽 모니터링
candump can0
```

`Type00 scan found no motor in id range`라면:
- CAN 케이블 연결 / 종단저항 확인
- 모터 전원 확인
- ID 범위 (`kScanMinId..kScanMaxId`) 확인

### 명령이 전송 안 됨

`/motor_cmd_array`를 발행해도 모터가 안 움직이면:
- 해당 모터가 `ready` 상태인지 로그 확인 (`motor_id=X ready: TYPE02 run-mode observed`)
- `ready` 전에는 외부 명령이 hold됨

### 갑자기 home으로 복귀

`/motor_cmd_array`가 `kCmdTimeoutMs` (기본 100ms) 동안 안 오면 자동으로 home으로 천천히 복귀합니다 — 안전 기능.
