# phy

물리 모델 기반 컨트롤 노드 모음. Pinocchio로 중력 보상, IK, collision-aware
trajectory planning을 계산하고 MIT 모드 토크 명령을 발행한다.

현재 EE 목표점으로 움직이는 본류는
`plan_compute_node + plan_node + send_target` 구조다.
`ee_xyz_trajectory_node`는 이전 단일 노드 방식이며, 간단한 IK 테스트나 비교용으로 남겨둔다.

## 노드

| 노드 | 역할 |
|------|------|
| [`hold_node`](phy/hold_node.py) | 중력 보상만 적용 — 외력에 자연스럽게 밀리는 cobot 기본 모드 |
| [`plan_compute_node`](phy/plan_compute_node.py) | EE 목표 pose 입력 → 현재 q 기준 IK 후보 생성 → SRDF 기반 self-collision check → 안전한 PTP plan 계산 |
| [`plan_node`](phy/plan_node.py) | `/computed_plan`을 받아 250 Hz로 quintic trajectory 실행 → `/motor_cmd_array` 발행 |
| [`send_target`](phy/send_target.py) | 터미널에서 `x y z yaw_deg`를 입력해 `/ee_target_pose` publish |
| [`joint_sweep_node`](phy/joint_sweep_node.py) | 접촉 없는 slow joint motion / sine sweep 로그 수집용 명령 발행 |
| [`ee_xyz_trajectory_node`](phy/ee_xyz_trajectory_node.py) | legacy 단일 노드: EE 좌표 입력 → IK → 5차 다항 궤적 → 모터 명령 |
| [`contact_detector_node`](phy/contact_detector_node.py) | 학습된 contact detector를 ROS topic에서 online inference |

## 라이브러리 모듈

| 파일 | 역할 |
|------|------|
| [`robot_model.py`](phy/robot_model.py) | URDF + Pinocchio 기반 FK / Jacobian / dynamics 공통 모델 |
| [`gravity.py`](phy/gravity.py) | `GravityCompensator` — pinocchio로 중력 토크 계산 |
| [`ik.py`](phy/ik.py) | Pinocchio 기반 DLS IK, 6D pose IK, periodic joint branch 정렬 |
| [`collision.py`](phy/collision.py) | URDF collision geometry + SRDF disable-collision pair 기반 self-collision checker |
| [`plan.py`](phy/plan.py) | IK 후보 ranking, collision-checked quintic plan, fold-and-rotate / wrist-retract fallback |
| [`traj.py`](phy/traj.py) | 5차 다항 (quintic) 궤적 생성 + 샘플링 |

## PTP IK planning 본류

### 전체 흐름

```text
send_target 또는 /ee_target_pose
  → plan_compute_node
      1. /motor_state_array에서 현재 q 확인
      2. 목표 xyz + yaw를 top-down gripper pose로 변환
      3. 여러 IK seed에서 후보 q_goal 계산
      4. manipulability / joint distance / elbow posture / j1 travel cost로 후보 ranking
      5. URDF collision geometry + SRDF를 이용해 self-collision이 없는 trajectory 선택
      6. quintic trajectory 계수를 /computed_plan으로 publish
  → plan_node
      1. /computed_plan 수신
      2. 250 Hz로 trajectory sample
      3. gravity compensation + PD + optional inertia feedforward 계산
      4. /motor_cmd_array publish
  → can_bridge_node
      1. /motor_cmd_array를 CAN MIT command로 변환
      2. 실제 적용 명령을 /motor_cmd_array_applied로 publish
```

`plan_compute_node`는 미리 저장된 IK를 꺼내는 노드가 아니다.
목표 pose가 들어오는 순간의 실제 현재 관절값을 기준으로 IK와 trajectory를 새로 계산하는
planning-only node다.

`plan_node`는 IK 후보를 고르지 않는다. 이미 `plan_compute_node`가 고른 collision-safe
trajectory를 받아서 실행만 담당한다. 따라서 제어 루프는 가볍게 유지되고, IK/collision 계산이
250 Hz command loop를 막지 않는다.

### IK 후보 선택 기준

IK 후보는 다음 기준으로 정렬된다.

- 현재 q와 가까운 후보
- manipulability가 너무 낮지 않은 후보
- j1 회전량이 작은 후보
- elbow-up 쪽에 가까운 후보
- j3/j4 부호 조합이 자연스러운 후보
- trajectory sample에서 self-collision이 없는 후보

SRDF는 여기서 중요하다. URDF collision geometry에서 모든 collision pair를 만들고,
SRDF에 정의된 disable-collision pair를 제거한 뒤 남은 pair만 검사한다.
즉 link3_3과 gripper처럼 구조적으로 붙어 있어서 collision이 아니어야 하는 pair는
SRDF에서 제외되어 있어야 한다.

### 주의

- `/motor_cmd_array` publisher는 한 번에 하나만 켠다.
- `plan_node`, `hold_node`, `joint_sweep_node`, `ee_xyz_trajectory_node`를 동시에 command publisher로 켜지 않는다.
- real logging / contact detector는 실제 적용된 명령을 보기 위해 `/motor_cmd_array_applied`를 사용하는 것이 좋다.
- `can_bridge_node`는 외부 command가 끊기면 home/timeout-home policy를 수행할 수 있다.

## hold_node

### 동작

1. `/motor_state_array` 구독 → 현재 q 측정
2. URDF + Pinocchio로 중력 토크 `τ_g(q)` 계산
3. 튜닝값 (kp, kd, gravity_scale, gravity_bias) 적용
4. `/motor_cmd_array` 발행 (250 Hz)

명령은 `tau_ff = gravity_scale * τ_g + gravity_bias` 그리고 kp, kd는 튜닝 YAML에서 옴.

### 실행

```bash
ros2 run phy hold_node
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `control_hz` | 250.0 | 컨트롤 루프 주파수 [Hz] |
| `state_timeout_s` | 0.2 | 모터 상태 stale 판정 시간 |
| `motor_joint_map_json` | `{"1":"j1",...,"6":"j6"}` | 모터 ID → URDF joint 이름 매핑 |
| `tau_limit_by_motor_json` | `{"1":6.0,"2":20.0,...}` | 모터별 토크 안전 한계 [Nm] |
| `urdf_path` | (sim 패키지 share/urdf/robot.urdf) | URDF override |
| `csv_log_path` | (없음) | 디버그 CSV 로그 경로 |

## ee_xyz_trajectory_node (legacy)

이 노드는 IK와 trajectory 실행을 한 프로세스 안에서 모두 수행하는 이전 구조다.
현재 PTP 작업에는 `plan_compute_node + plan_node` 구조를 우선 사용한다.

### 동작

1. EE 좌표 입력 (터미널 또는 `/ee_target_xyz` 토픽)
2. IK로 목표 관절각 계산 (여러 시드 + residual/jump 정책)
3. 5차 다항 궤적 생성 (현재 → 목표, 시간은 v_max/a_max 기반 자동 산정)
4. 매 tick 궤적 샘플링 + 중력 보상 추가 → `/motor_cmd_array` 발행

### 실행

```bash
# 터미널 입력 모드 (기본)
ros2 run phy ee_xyz_trajectory_node
# target xyz> 0.3 0.0 0.4

# 토픽으로 입력
ros2 topic pub /ee_target_xyz std_msgs/Float64MultiArray "{data: [0.3, 0.0, 0.4]}"
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `kp` / `kd` | 1.0 / 0.05 | 위치/속도 게인 (튜닝 YAML이 override) |
| `v_max` / `a_max` | 0.8 / 1.5 | 관절 속도/가속도 한계 [rad/s, rad/s²] |
| `min_traj_duration` | 0.2 | 최소 궤적 시간 [s] |
| `target_frame` | `ee_link` | IK 목표 프레임 이름 (URDF) |
| `controlled_motor_ids_json` | `[1,2,3]` | IK가 제어할 모터 ID 목록 |
| `max_ik_residual_accept_m` | 0.005 | IK 수렴 허용 오차 [m] |
| `ik_random_restarts` | 24 | IK 랜덤 시드 개수 (로컬 최소 회피) |
| `max_joint_jump_rad` | 0 (비활성) | 관절 점프 거부 임계값 [rad] |

## 튜닝

kp / kd / gravity_scale / gravity_bias 등은 컨트롤 노드 코드가 아니라 **튜닝 YAML**에서 옵니다 (`param/tuned/control_params.yaml`).

실행 중 변경:
```bash
cd /home/su/idle_ws/motor
python3 control_param_set.py --can_id 1 --kp 5.0 --kd 0.5
python3 control_param_save.py
# 다음 tick에 자동 반영 (mtime 캐싱)
```

자세한 내용은 [idle_common/README.md](../idle_common/README.md) 참조.

## 빌드 + 실행 전체 흐름

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select phy msgs can_interface
source install/setup.bash
```

### PTP IK 실행

터미널 1, CAN bridge:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run can_interface can_bridge_node
```

터미널 2, planning-only node:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy plan_compute_node
```

터미널 3, control-only node:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy plan_node
```

터미널 4, 목표 pose 전송:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy send_target -- 0.30 0.00 0.60 0
```

`send_target` 인자는 `x y z yaw_deg` 순서다.

### 중력 보상 hold만 실행

`hold_node`를 사용할 때는 `plan_node`를 끈다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy hold_node
```

### legacy EE 궤적 노드

```bash
ros2 run phy ee_xyz_trajectory_node
```

실물 no-contact hard negative 데이터 수집 절차는
[contact_detection/REAL_NO_CONTACT_COLLECTION.md](../../contact_detection/REAL_NO_CONTACT_COLLECTION.md)에 정리되어 있다.

## 의존성

- `idle_common`
- `msgs`
- `rclpy`
- `numpy`
- `pinocchio` (apt: `ros-humble-pinocchio`)
- `sim` (URDF/메쉬 파일 제공)
