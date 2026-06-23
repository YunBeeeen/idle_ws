# idle_ws — 6-DoF 협동로봇 컨트롤 워크스페이스

ROS2 (Humble) 기반 6-DoF 협동로봇 (cobot) 제어용 워크스페이스입니다. CAN 버스로 모터와 통신하고, Pinocchio 모델로 중력보상을 계산하며, MuJoCo로 시뮬레이션 시각화를 지원합니다.

## 시스템 구성

```
                    +-------------------+
                    |  사용자 입력      |  (터미널, 토픽)
                    +---------+---------+
                              |
                  /ee_target_xyz | /motor_cmd_array
                              v
+-------------+      +--------+--------+      +---------------+
|  hold_node  |      | ee_xyz_traj...  |      |  viewer_node  |
|  (중력유지) |      |  (IK + 궤적)    |      |  (MuJoCo GUI) |
+------+------+      +--------+--------+      +-------+-------+
       |                      |                       ^
       |   /motor_cmd_array   |                       |
       +----------+-----------+      /motor_state_array
                  v                                   |
            +-----+------+                            |
            | can_bridge |                            |
            |   (C++)    +----------------------------+
            +-----+------+   /motor_state_array
                  |
              CAN 0  (1 Mbps)
                  |
          +-------+--------+
          |   모터 1-6     |
          |   + 그리퍼     |
          +----------------+
```

## 패키지 구조

| 패키지 | 역할 | 언어 |
|--------|------|------|
| [idle_common](src/idle_common/) | 공통 유틸 (튜닝 YAML, 모터 매핑, 파라미터, 경로 헬퍼) | Python |
| [idle_launch](src/idle_launch/) | ROS2 launch 파일 | Python |
| [msgs](src/msgs/) | 커스텀 메시지 (`MotorCMD`, `MotorState` 등) | IDL |
| [phy](src/phy/) | 컨트롤 노드 (중력유지, EE 궤적, IK) | Python |
| [sim](src/sim/) | MuJoCo viewer + URDF + 메쉬 | Python |
| [can_interface](src/can_interface/) | CAN ↔ ROS 브리지 노드 | C++ |
| [param/](param/) | 튜닝 YAML 저장소 (`original/`, `tuned/`) | YAML |
| [motor/](motor/) | CLI 도구 (모터 enable/stop/튜닝/진단) | Python |

## 빌드

```bash
cd /home/su/idle_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## 빠른 시작

### 1. CAN 인터페이스 활성화

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### 2. 노드 실행

각 노드를 별도 터미널에서:

```bash
# 터미널 1: CAN 브리지
ros2 run can_interface can_bridge_node

# 터미널 2: 중력유지 컨트롤러 + MuJoCo viewer
ros2 launch idle_launch gravity_hold.launch.py

# (선택) 터미널 3: EE 좌표 궤적 컨트롤러
ros2 run phy ee_xyz_trajectory_node
```

### 3. 모터 튜닝

```bash
# 컨트롤 노드 실행 중에도 가능 — 다음 tick에 자동 반영
cd /home/su/idle_ws/motor
python3 control_param_set.py --can_id 1 --kp 5.0 --kd 0.5
python3 control_param_save.py   # tuned/control_params.yaml 저장
```

## 주요 토픽

| 토픽 | 메시지 타입 | 방향 | 비고 |
|------|-------------|------|------|
| `/motor_state_array` | `msgs/MotorStateArray` | can_bridge → 컨트롤 노드 | 모터 피드백 (q, qd, tau) |
| `/motor_cmd_array` | `msgs/MotorCMDArray` | 컨트롤 노드 → can_bridge | MIT 모드 명령 (q_des, kp, kd, tau_ff) |
| `/motor_error_array` | `msgs/MotorErrorArray` | can_bridge → 진단 | 에러 비트 변화 이벤트 |
| `/ee_target_xyz` | `std_msgs/Float64MultiArray` | 사용자 → ee_xyz_node | EE 목표 좌표 (x, y, z) |

## 의존성

- ROS2 Humble (Ubuntu 22.04)
- Python 3.10
- pinocchio (`apt: ros-humble-pinocchio`)
- mujoco (Python wheel)
- C++17 컴파일러
- CAN 어댑터 (1 Mbps, SocketCAN)

## 협동로봇 설계 의도

이 로봇은 **외력에 자연스럽게 밀리는 협동로봇 (cobot)** 으로 설계되었습니다. 그래서:

- 250 Hz 컨트롤 루프가 중력 보상 토크를 끊김 없이 보냄
- 사람이 손으로 밀면 위치(q)가 바뀌고, 바뀐 q에 맞는 중력 토크가 자동 계산되어 부드럽게 따라감
- kp/kd는 작게 잡아 강성을 낮춤 (안전성 > 정밀도)

자세한 설계 노트는 각 패키지 README와 [param/README.md](param/README.md) 참조.

## 라이선스

Apache-2.0
