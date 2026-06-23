# 실제 로봇 Static Hold 데이터 수집 빠른 실행

이 문서는 실제 로봇에서 no-contact static hold 데이터를 빠르게 쌓기 위한 현장용 명령어 모음이다.

목표는 사람이 로봇을 건드리지 않은 상태에서 여러 자세의 정상 중력보상/잔류 오차 데이터를 저장하는 것이다.

```text
label = nc
label value = 0
용도 = 실물 정상 상태를 contact로 오검출하지 않게 만드는 hard negative 데이터
```

## 지금 켜야 하는 것

터미널은 보통 3개를 쓴다.

```text
Terminal 1: CAN bridge
Terminal 2: hold_node 중력보상
Terminal 3: record_real_log.py recorder
```

`CAN bridge`와 `hold_node`는 계속 켜두고, `recorder`만 자세마다 한 번씩 실행한다.

## Terminal 1: CAN bridge

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run can_interface can_bridge_node
```

## Terminal 2: 중력보상 hold

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy hold_node
```

이 터미널은 계속 켜둔다.

## Terminal 3: 공통 준비

아래 명령은 Terminal 3에서 한 번만 실행한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/no_contact/${TODAY}
mkdir -p "${LOG_DIR}"
[ -f "${LOG_DIR}/manifest.csv" ] || printf "episode_id,type,description,csv_path,label,notes\n" > "${LOG_DIR}/manifest.csv"
```

## 자세마다 반복하는 순서

각 자세마다 아래 순서로 진행한다.

```text
1. 로봇을 원하는 static hold 자세로 둔다.
2. 손을 뗀다.
3. 3~5초 기다려서 진동/잔류 움직임이 줄어들게 한다.
4. Terminal 3에서 recorder 명령을 실행한다.
5. 기록 중에는 10초 동안 절대 건드리지 않는다.
6. duration reached 로그가 뜨면 다음 자세로 넘어간다.
```

정상 종료 로그는 다음처럼 보인다.

```text
all topics ready; recording timer starts now
duration reached (...s >= 10.00s); wrote 1000 rows
```

## Episode 1: 접은 자세

이미 수집했다면 다시 실행하지 않아도 된다.

```bash
EP=static_hold_folded_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 10 \
  --residual-offset-duration 2.0

echo "${EP},static_hold,folded arm hold,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

## Episode 2: 앞으로 뻗은 자세

이미 수집했다면 다시 실행하지 않아도 된다.

```bash
EP=static_hold_forward_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 10 \
  --residual-offset-duration 2.0

echo "${EP},static_hold,forward extended hold,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

## Episode 3: 옆으로 뻗은 자세

```bash
EP=static_hold_side_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 10 \
  --residual-offset-duration 2.0

echo "${EP},static_hold,side extended hold,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

## Episode 4: 손목/팔꿈치가 꺾인 자세

```bash
EP=static_hold_bent_wrist_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 10 \
  --residual-offset-duration 2.0

echo "${EP},static_hold,bent wrist hold,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

## Episode 5: 중력 부하가 큰 자세

무리한 자세면 10초보다 짧게 해도 된다. 모터가 부담스러워 보이면 즉시 중단한다.

```bash
EP=static_hold_loaded_posture_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 10 \
  --residual-offset-duration 2.0

echo "${EP},static_hold,high gravity load posture,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

## 주의

같은 `EP` 이름으로 다시 실행하면 기존 CSV가 덮어써진다.

```text
static_hold_forward_001.csv
```

을 다시 기록하면 이전 `static_hold_forward_001.csv`는 사라진다.

새로 저장하려면 번호를 올린다.

```bash
EP=static_hold_forward_002
```

`manifest.csv`는 `echo ... >>` 방식이라 줄이 계속 추가된다. 실수로 같은 episode를 여러 번 추가하면 나중에 manifest만 정리하면 된다.

## 저장 확인

수집 후 아래 명령으로 현재 저장 상태를 확인한다.

```bash
find "${LOG_DIR}" -maxdepth 1 -type f | sort
sed -n '1,40p' "${LOG_DIR}/manifest.csv"
```

CSV 하나는 보통 다음 정도면 정상이다.

```text
duration 10초, sample_hz 100이면 rows 약 1000개
NaN/Inf 없어야 함
tau_residual_corrected column 있어야 함
```

## 다음 단계: Slow Sine 데이터

static hold를 5개 안팎으로 모았다면, 다음은 slow sine no-contact 데이터를 모은다.

중요:

```text
hold_node를 끄고 joint_sweep_node만 켠다.
CAN bridge는 계속 켜둔다.
record_real_log.py는 별도 터미널에서 실행한다.
```

빌드 후 설치된 파라미터 파일을 사용한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select phy --allow-overriding phy
source install/setup.bash
```

### Terminal 2: j2 slow sine

```bash
ros2 run phy joint_sweep_node --ros-args \
  --params-file install/phy/share/phy/config/slow_sine_j2.yaml
```

### Terminal 3: j2 recorder

```bash
TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/no_contact/${TODAY}
EP=slow_sine_j2_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 28 \
  --residual-offset-duration 2.0

echo "${EP},slow_joint,j2 slow sine,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

### j3 slow sine

Terminal 2:

```bash
ros2 run phy joint_sweep_node --ros-args \
  --params-file install/phy/share/phy/config/slow_sine_j3.yaml
```

Terminal 3:

```bash
EP=slow_sine_j3_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 28 \
  --residual-offset-duration 2.0

echo "${EP},slow_joint,j3 slow sine,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

### j2+j3 slow sine

Terminal 2:

```bash
ros2 run phy joint_sweep_node --ros-args \
  --params-file install/phy/share/phy/config/slow_sine_j23.yaml
```

Terminal 3:

```bash
EP=slow_sine_j23_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 32 \
  --residual-offset-duration 2.0

echo "${EP},slow_joint,j2+j3 slow sine,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```
