# sim

로봇 모델 + MuJoCo 시뮬레이션 viewer.

## 구성

| 위치 | 내용 |
|------|------|
| [`sim/viewer_node.py`](sim/viewer_node.py) | MuJoCo viewer 노드 — 실제 모터 상태를 시각화 |
| [`urdf/robot.urdf`](urdf/robot.urdf) | 6-DoF arm + 그리퍼 URDF |
| [`meshes/*.stl`](meshes/) | 링크 시각화/충돌 메쉬 |
| `robot.xml` | MuJoCo 모델 (URDF에서 변환) |

## viewer_node

### 동작

1. MuJoCo 모델 로드 (`robot.xml`)
2. `/motor_state_array` 구독 → 모터 ID → URDF joint 매핑 → MuJoCo `qpos`/`qvel`에 mirror
3. `mj_forward` (kinematics 계산) — 새 상태 들어왔을 때만
4. viewer window 갱신 (60 Hz)

**컨트롤은 하지 않음** — 순수 시각화 노드. 실제 모터의 현재 자세를 화면에 그대로 비추기만 함.

### 실행

```bash
ros2 run sim viewer_node
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `viewer` | true | viewer 창 표시 (false → headless) |
| `viewer_hz` | 60.0 | viewer 갱신 주파수 |
| `viewer_left_ui` / `viewer_right_ui` | true | MuJoCo UI 패널 표시 |
| `model_xml` | (sim share/robot.xml) | MuJoCo XML override |
| `motor_joint_map_json` | 6-motor 기본값 | 모터 ID → joint 이름 매핑 |

### 단독 실행 vs launch 사용

`gravity_hold.launch.py`가 viewer_node를 자동 띄우므로 보통 단독 실행할 필요 없음:

```bash
ros2 launch idle_launch gravity_hold.launch.py
```

단독 실행은 디버그 / 다른 컨트롤 노드와 조합 시:

```bash
ros2 run sim viewer_node --ros-args -p viewer:=true
```

## URDF / 메쉬

### 주요 링크 / 관절

- `base` → `j1` → `j2` → ... → `j6` → `rs05` (손목) → `gripper`
- `finger_r` (prismatic, 그리퍼 master)
- `finger_l` (prismatic, finger_r mimic — 1:1 비율)

### 모터 ↔ joint 매핑

기본값 (6-DoF arm):
```
1 → j1
2 → j2
3 → j3
4 → j4
5 → j5
6 → j6
```

그리퍼 (7번 모터)는 패럴 4-bar 기구로 prismatic finger 구동 — URDF의 mimic은 시각화 근사이고 실제 motor 각도 ↔ 그리퍼 폭 매핑은 별도 함수 필요 (미구현).

### URDF 수정 후

```bash
colcon build --packages-select sim
source install/setup.bash
# viewer_node 재시작
```

URDF 변경은 `phy.gravity` / `phy.ik`의 Pinocchio 모델에도 영향 → 관련 노드 재시작 필수.

## MuJoCo XML 호환성

`load_model_with_workaround()`가 `fullinertia` + `quat` 충돌을 자동으로 처리합니다 (mujoco 최신 버전 호환).

## 의존성

- `idle_common` (motor_map, paths)
- `msgs` (MotorStateArray)
- `mujoco` (Python wheel)
- `ament_index_python`
- `rclpy`

## 빌드

```bash
colcon build --packages-select sim
```
