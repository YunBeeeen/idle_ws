# idle_common

워크스페이스 전체에서 공유하는 **Python 유틸리티 라이브러리**. ROS2 노드는 아니고, 다른 패키지 (`phy`, `sim`, `motor/` CLI 도구)가 import해서 사용합니다.

## 개요

각 컨트롤 노드가 동일한 코드를 재구현하지 않도록 공통 로직을 모아둔 곳입니다. 주요 책임:

- 튜닝 YAML 읽기/쓰기 + 캐싱
- 모터 ↔ URDF joint 매핑 상수와 JSON 파서
- ROS 패키지 share 디렉토리 경로 해석
- ROS 파라미터 선언 헬퍼

## 모듈

| 파일 | 역할 |
|------|------|
| [`control_tuning.py`](idle_common/control_tuning.py) | 튜닝 YAML 게이트 + `control_params_for_motor()` (mtime 캐시) |
| [`param_store.py`](idle_common/param_store.py) | YAML 파일 I/O, 원본/튜닝 병합, 락 |
| [`motor_map.py`](idle_common/motor_map.py) | `DEFAULT_MOTOR_JOINT_MAP`, `DEFAULT_TAU_LIMIT_BY_MOTOR`, JSON 파서 |
| [`paths.py`](idle_common/paths.py) | `resolve_share_file(pkg, rel, override_text)` |
| [`ros_params.py`](idle_common/ros_params.py) | `declare_typed(node, name, default, cast=None)` |

## 사용법

### 튜닝값 읽기 (컨트롤 루프에서)

```python
from idle_common.control_tuning import control_params_for_motor

params = control_params_for_motor(motor_id=1)
# {'kp': 3.0, 'kd': 0.12, 'gravity_scale': 1.0, ...}
```

내부적으로 두 YAML 파일 (`param/original/control_params.yaml`, `param/tuned/control_params.yaml`)을 mtime 기반으로 캐싱합니다. CLI로 튜닝값을 저장하면 다음 호출에서 자동 반영됩니다.

### 모터 매핑 파싱

```python
from idle_common.motor_map import DEFAULT_MOTOR_JOINT_MAP, parse_motor_joint_map_json
# DEFAULT_MOTOR_JOINT_MAP = {1:"j1", 2:"j2", ..., 6:"j6"}

motor_map = parse_motor_joint_map_json('{"1":"j1","2":"j2"}')
# {1: 'j1', 2: 'j2'}
```

### URDF / 메쉬 경로 해석

```python
from idle_common.paths import resolve_share_file

# ROS parameter로 override 없으면 sim/urdf/robot.urdf 사용
urdf_path = resolve_share_file("sim", "urdf/robot.urdf", override_text="")

# 절대 경로 override
urdf_path = resolve_share_file("sim", "urdf/robot.urdf", "/tmp/my_robot.urdf")
```

### ROS 파라미터 타입 안전 선언

```python
from idle_common.ros_params import declare_typed

# type(default)로 자동 cast
self.control_hz = declare_typed(self, "control_hz", 250.0)      # float
self.viewer_enabled = declare_typed(self, "viewer", True)        # bool

# 명시적 cast (예: strip)
strip_str = lambda v: str(v).strip()
self.urdf_path = declare_typed(self, "urdf_path", "", cast=strip_str)
```

## 튜닝 워크플로우

게이트 상태 파일 (`/tmp/idle_control_gate_state.json`)으로 안전성 보장. **컨트롤 노드 (`hold_node`, `ee_xyz_trajectory_node`)는 게이트에 참여하지 않으므로** 실행 중에도 튜닝 변경 가능.

```bash
# 1. 현재 값 확인
cd /home/su/idle_ws/motor
python3 control_param_show.py

# 2. 값 수정 (컨트롤 실행 중 OK)
python3 control_param_set.py --can_id 1 --kp 5.0 --kd 0.5

# 3. 저장 (param/tuned/control_params.yaml에 영구 기록)
python3 control_param_save.py
```

저장하면 mtime이 바뀌어 다음 컨트롤 tick (~4ms 안)에 자동 반영됩니다.

## 빌드

```bash
colcon build --packages-select idle_common
```

## 의존성

- `python3-yaml`
- `ament_index_python` (paths.py)
