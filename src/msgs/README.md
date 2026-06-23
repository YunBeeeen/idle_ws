# msgs

워크스페이스 전용 ROS2 메시지 정의. CAN 브리지와 컨트롤 노드 사이에서 흐르는 데이터 타입.

## 메시지 목록

| 메시지 | 용도 |
|--------|------|
| [`MotorCMD`](msg/MotorCMD.msg) | 단일 모터에 보낼 MIT 명령 (q_des, qd_des, kp, kd, tau_ff) |
| [`MotorCMDArray`](msg/MotorCMDArray.msg) | tick당 모든 모터 명령을 묶은 배열 |
| [`MotorState`](msg/MotorState.msg) | 모터 한 개의 피드백 상태 (q, qd, tau, 온도) |
| [`MotorStateArray`](msg/MotorStateArray.msg) | tick당 모든 모터 피드백 배열 |
| [`MotorError`](msg/MotorError.msg) | 모터 에러 이벤트 (fault bits + mode) |
| [`MotorErrorArray`](msg/MotorErrorArray.msg) | 에러 이벤트 배열 |

## MotorCMD 필드

```
builtin_interfaces/Time stamp
uint8 motor_id
float32 q_des     # 목표 각도 [rad]
float32 qd_des    # 목표 각속도 [rad/s]
float32 kp        # 위치 게인 [Nm/rad]
float32 kd        # 속도 게인 [Nm·s/rad]
float32 tau_ff    # 피드포워드 토크 [Nm]
```

MIT 모드 토크 계산: `tau = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff`

## MotorState 필드

```
builtin_interfaces/Time stamp
uint8 motor_id
float32 q         # 현재 각도 [rad]
float32 qd        # 현재 각속도 [rad/s]
float32 tau       # 측정 토크 [Nm]
float32 temp_c    # 모터 온도 [°C]
```

## 배열 메시지 규칙

- 한 tick에 들어온 모터 ID별로 **마지막 값 하나씩만** 담음
- 배열 내 원소는 `motor_id` 오름차순 정렬
- 해당 tick에 데이터가 없으면 메시지 자체를 발행하지 않음

## 사용 예시

### Python에서 명령 발행

```python
from msgs.msg import MotorCMD, MotorCMDArray

cmd = MotorCMD()
cmd.motor_id = 1
cmd.q_des = 0.0
cmd.kp = 5.0
cmd.kd = 0.5
cmd.tau_ff = 1.2

msg = MotorCMDArray()
msg.stamp = self.get_clock().now().to_msg()
msg.commands = [cmd]
publisher.publish(msg)
```

### CLI로 토픽 확인

```bash
ros2 topic echo /motor_state_array --qos-reliability best_effort
ros2 topic info /motor_cmd_array -v
```

## 빌드

```bash
colcon build --packages-select msgs
```

## 의존성

- `builtin_interfaces`
- `rosidl_default_generators`
- `rosidl_default_runtime`
