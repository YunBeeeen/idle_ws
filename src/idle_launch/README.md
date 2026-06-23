# idle_launch

ROS2 launch 파일 모음. 여러 노드를 한 번에 띄우는 용도.

## launch 파일

| 파일 | 띄우는 노드 | 용도 |
|------|------------|------|
| [`gravity_hold.launch.py`](launch/gravity_hold.launch.py) | `hold_node` + `viewer_node` | 중력 유지 모드 + MuJoCo 시각화 |

## 사용법

```bash
ros2 launch idle_launch gravity_hold.launch.py
```

### launch 인자

```bash
# viewer 끄기 (headless)
ros2 launch idle_launch gravity_hold.launch.py viewer:=false

# UI 패널 숨기기
ros2 launch idle_launch gravity_hold.launch.py viewer_left_ui:=false viewer_right_ui:=false
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `viewer` | true | MuJoCo viewer 창 표시 |
| `viewer_left_ui` | true | viewer 좌측 UI 패널 |
| `viewer_right_ui` | true | viewer 우측 UI 패널 |

## 실행 전 준비

```bash
# 1. CAN 인터페이스 활성화
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 2. 별도 터미널에서 can_bridge 실행
ros2 run can_interface can_bridge_node

# 3. launch 실행
ros2 launch idle_launch gravity_hold.launch.py
```

`can_bridge_node`는 launch에 포함되어 있지 않습니다 (sudo 필요한 CAN 설정과 분리하기 위함).

## 빌드

```bash
colcon build --packages-select idle_launch
```

## 의존성

- `phy` (hold_node 실행)
- `sim` (viewer_node 실행)
- `launch_ros`
