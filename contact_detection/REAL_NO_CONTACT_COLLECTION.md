# Real No-Contact 데이터 수집 절차

이 문서는 실물 로봇에서 `label=nc` hard negative 데이터를 모으기 위한 실행 절차다.
목표는 steady-state tracking error, 느린 움직임의 마찰, 방향 전환, PTP 이동 후 잔류 오차를 모델이 contact로 오해하지 않도록 만드는 것이다.

이번 단계는 **no-contact 데이터 수집 전용**이다. intentional contact label은 아직 수집하지 않는다.

## 저장 규칙

CSV는 episode 단위로 저장한다.

```bash
contact_detection/real_logs/no_contact/<YYYYMMDD>/<episode_id>.csv
```

같은 날짜 폴더에 `manifest.csv`를 두고 각 episode를 기록한다.

```csv
episode_id,type,description,csv_path,label,notes
static_hold_folded_001,static_hold,folded arm hold,contact_detection/real_logs/no_contact/20260607/static_hold_folded_001.csv,nc,no human contact
```

초기화 예시:

```bash
cd ~/idle_ws
TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/no_contact/${TODAY}
mkdir -p "${LOG_DIR}"
printf "episode_id,type,description,csv_path,label,notes\n" > "${LOG_DIR}/manifest.csv"
```

모든 episode의 label은 `nc`, 즉 `0`이다. 기본 42D 모델에서는 `tau_meas`를 모델 입력으로 쓰지 않고 진단용으로만 둔다.

## 실제 로봇 residual column

`record_real_log.py`는 기존 CSV column을 유지하면서 뒤쪽에 다음 column을 추가로 저장한다.

```text
qdot_des1..qdot_des6
tau_residual1..tau_residual6
tau_residual_corrected1..tau_residual_corrected6
```

현재 실물 로봇 command 구조에서는 `tau_cmd`가 이미 다음 전체 명령 토크다.

```text
tau_cmd = kp * (q_des - q) + kd * (qdot_des - qdot) + tau_ff
```

여기서 `tau_ff`는 중력보상을 포함한다. 그래서 residual 기본 정의는 다음으로 고정한다.

```text
tau_residual = tau_meas - tau_cmd
tau_residual_corrected = tau_residual - episode 초기 no-contact 평균 offset
```

중요한 점은 `tau_ext`, F/T sensor 값, 외부 force ground truth는 여전히 모델 입력으로 쓰지 않는다는 것이다. residual feature mode를 쓰더라도 입력은 실제 로봇에서 읽을 수 있는 `tau_meas`, `tau_cmd`, `q`, `qdot`, `q_des`로부터 만든 값만 사용한다.

초기 residual offset 보정 시간은 recorder에서 바꿀 수 있다.

```bash
python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 15 \
  --residual-offset-duration 2.0
```

`2.0s` 동안 사람이 건드리지 않은 상태여야 offset 보정이 의미 있다.

## 공통 준비

터미널마다 아래 setup을 먼저 실행한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Terminal 1: CAN bridge

```bash
ros2 run can_interface can_bridge_node
```

한 번에 `/motor_cmd_array` publisher는 하나만 켠다. `plan_node`, `hold_node`, `joint_sweep_node`, 다른 command publisher를 동시에 켜지 않는다.

## 1. Static Hold

목표: 목표 자세에 도착한 뒤에도 남는 steady-state tracking error를 no-contact로 기록한다.

추천 자세:

- 팔을 접은 자세
- 팔을 앞으로 뻗은 자세
- 팔을 옆으로 뻗은 자세
- 손목이 꺾인 자세
- 그리퍼 장착 상태에서 무게가 크게 걸리는 자세
- 오차가 많이 나는 관절이 포함된 자세

Terminal 2: planner

```bash
ros2 run phy plan_compute_node
```

Terminal 3: controller

```bash
ros2 run phy plan_node
```

Terminal 4: 목표 pose 전송

```bash
ros2 run phy send_target -- x y z yaw_deg
```

예시:

```bash
ros2 run phy send_target -- 0.30 0.00 0.60 0
```

목표 도착 후 바로 기록하지 말고 3~5초 settle을 기다린다. 그 다음 별도 터미널에서 recorder를 실행한다.

Terminal 5: recorder

```bash
TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/no_contact/${TODAY}
EP=static_hold_folded_001
python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 15
```

기록 후 manifest에 한 줄을 추가한다.

```bash
echo "${EP},static_hold,folded arm hold,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

## 2. Slow Joint Motion

목표: 접촉 없이 천천히 움직일 때 생기는 마찰, 방향 전환, `e_q`, `delta_e_q`, `tau_cmd` 변화를 no-contact로 기록한다.

`plan_node`는 끄고 `joint_sweep_node`만 켠다.

Terminal 2: slow sine command

```bash
ros2 run phy joint_sweep_node --ros-args \
  -p sine_motion_enabled:=true \
  -p sine_duration_s:=20.0 \
  -p segment_duration_s:=10.0 \
  -p sine_envelope_enabled:=false \
  -p sweep_kp:=3.0 \
  -p sweep_kd:=0.25 \
  -p hold_kp:=1.5 \
  -p hold_kd:=0.10 \
  -p sine_motor_ids_json:='[2]' \
  -p sine_center_by_motor_json:='{"2": -0.70}' \
  -p sine_amplitude_by_motor_json:='{"2": 0.08}' \
  -p sine_frequency_by_motor_json:='{"2": 0.08}' \
  -p sine_phase_by_motor_json:='{"2": 0.0}'
```

Recorder:

```bash
TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/no_contact/${TODAY}
EP=slow_sine_j2_001
python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 22
echo "${EP},slow_joint,j2 slow sine,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

다른 추천 episode:

- `sine_motor_ids_json:='[1]'`, j1 단독 왕복
- `sine_motor_ids_json:='[3]'`, j3 단독 왕복
- `sine_motor_ids_json:='[4,5]'`, 손목 느린 왕복
- `sine_motor_ids_json:='[2,3]'`, j2+j3 동시 slow sine

period는 8~15초 정도가 좋다. `sine_frequency_by_motor_json`에서 `0.08 Hz`는 period 12.5초다. 처음에는 amplitude를 작게 시작하고, 방향 전환 구간이 최소 1~2번 들어가도록 duration을 잡는다.

## 3. Point-to-Point Trajectory

목표: 실제 사용할 동작과 비슷한 PTP 이동 중/이동 후 no-contact 데이터를 모은다.

Terminal 2:

```bash
ros2 run phy plan_compute_node
```

Terminal 3:

```bash
ros2 run phy plan_node
```

PTP는 recorder를 먼저 켠 뒤 목표를 보낸다. 그래야 move 구간과 hold 구간이 같은 CSV에 들어간다.

Terminal 4: recorder

```bash
TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/no_contact/${TODAY}
EP=ptp_a_to_b_hold_001
python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 15
```

Terminal 5: target

```bash
ros2 run phy send_target -- x y z yaw_deg
```

기록 후 manifest:

```bash
echo "${EP},ptp,pose A to B plus hold,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

권장 구성은 `move 3~5s -> hold 5~10s`다. 같은 방식으로 `A -> B`, `B -> C`, `C -> A`를 각각 별도 CSV로 저장한다.

## 수집 후 확인

CSV가 생성되면 기존 추론/리뷰 스크립트로 false alarm을 본다.

```bash
python3 contact_detection/infer_real_log.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --csv "${LOG_DIR}/${EP}.csv" \
  --model-path contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__gru_014/models/gru_detector.pt \
  --scaler-path contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__gru_014/models/scaler.pkl
```

그 다음 `review_real_contact_log.py`로 no-contact false alarm과 feature 분포를 확인한다.

추론/리뷰를 실행하면 다음 파일도 같이 생성된다.

```text
metrics/real_no_contact_sanity.json
figures/residual_timeseries_example.png
```

`real_no_contact_sanity.json`에는 no-contact 구간에서의 `P(contact)` 평균, 최대값, p95, p99, false alarm fraction이 저장된다. `residual_timeseries_example.png`에는 `||tau_residual||`, `||tau_residual_corrected||`, `P(contact)`가 함께 그려져서 실제 로봇의 정상 추종 오차와 residual이 모델 probability를 얼마나 밀어 올리는지 확인할 수 있다.

## Hard negative로 학습에 추가하기

실제 no-contact CSV는 contact positive label이 아니므로 단독으로 binary detector를 학습할 수 없다. 먼저 기존 sim pretrain을 유지하고, 실제 no-contact CSV는 hard negative로 train/val에 추가한다.

```bash
python3 contact_detection/train_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --ablation-tag real_nc_hard_negative_001 \
  --real-no-contact-csv contact_detection/real_logs/no_contact/<YYYYMMDD>/slow_sine_j2_001.csv \
  --real-no-contact-csv contact_detection/real_logs/no_contact/<YYYYMMDD>/static_hold_folded_001.csv
```

validation용 no-contact CSV를 따로 빼고 싶으면 다음 옵션을 추가한다.

```bash
--real-no-contact-val-csv contact_detection/real_logs/no_contact/<YYYYMMDD>/ptp_a_to_b_hold_001.csv
```

이 방식은 “실물 정상 동작을 contact로 오검출하지 않게 하는” fine-tuning/diagnosis 단계다. 실제 contact positive label을 얻기 전까지는 recall 개선보다 false positive 감소를 확인하는 용도로 해석한다.

## Residual feature mode 실험

기본값은 기존 42D feature다.

```yaml
dataset:
  feature_mode: original_42
```

실제 robot residual 기반 모델을 따로 실험할 때는 config를 복사해서 다음 중 하나로 바꾼다.

```yaml
dataset:
  feature_mode: residual_v1
```

지원되는 mode는 다음이다.

```text
original_42:
  [q, qdot, e_q, tau_cmd, delta_e_q, delta_qdot, delta_tau_cmd]

residual_v1:
  [qdot, tau_cmd, tau_residual_corrected, delta_tau_residual, delta_e_q, delta_qdot]

residual_v2:
  [tau_residual_corrected, delta_tau_residual, qdot, delta_qdot]

residual_v3:
  [q, qdot, tau_cmd, tau_residual_corrected, delta_tau_residual, delta_e_q, delta_qdot]
```

주의할 점은 residual feature mode에는 `tau_residual`이 필요하다는 것이다. 기존 sim `.npz`에 residual column이 없으면 residual mode 학습은 바로 되지 않는다. 현재 실물 로그에서 기존 `original_42` 모델의 false alarm이 크게 관측되었기 때문에, 단순히 `original_42`를 더 튜닝하기보다 residual 분포를 먼저 확인하고 실물 positive contact CSV를 추가로 모은 뒤 `residual_v2`를 1차 후보로 비교하는 흐름이 더 적절하다.

현재까지 모은 no-contact 로그의 residual 분포는 다음 명령으로 요약한다.

```bash
python3 contact_detection/analyze_real_residuals.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --manifest contact_detection/real_logs/no_contact/20260618/manifest.csv \
  --manifest contact_detection/real_logs/no_contact/20260619/manifest.csv \
  --probability-summary-dir contact_detection/outputs_legacy_gru_mlp/randomized_sim/real_inference/no_contact_check
```

주요 출력은 다음이다.

```text
contact_detection/outputs_legacy_gru_mlp/randomized_sim/real_inference/no_contact_residual_analysis/real_no_contact_residual_summary.json
contact_detection/outputs_legacy_gru_mlp/randomized_sim/real_inference/no_contact_residual_analysis/real_no_contact_residual_summary.csv
contact_detection/outputs_legacy_gru_mlp/randomized_sim/real_inference/no_contact_residual_analysis/real_no_contact_residual_overview.png
```

## 주의사항

- 사람이 로봇을 건드린 episode는 `nc` 데이터로 쓰지 않는다.
- 목표 도착 직후 바로 끊지 말고, 잔류 오차가 남은 hold 구간을 포함한다.
- `plan_node`와 `joint_sweep_node`를 동시에 켜지 않는다.
- CSV 파일명과 manifest description에는 자세/동작 조건을 남긴다.
- Fine-tuning은 수집 CSV를 리뷰한 뒤 별도 단계에서 진행한다.

## Contact positive 수동 interval 기록

실제 contact positive 데이터를 모을 때는 `record_real_log.py`의 터미널 타이머를 보고 사람이 정해진 시간에 접촉한다. 예를 들어 15초 기록에서 5초부터 8초까지 EE를 가볍게 밀면 다음처럼 실행한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/contact/${TODAY}
mkdir -p "${LOG_DIR}"
[ -f "${LOG_DIR}/manifest.csv" ] || printf "episode_id,type,description,csv_path,label,contact_intervals,notes\n" > "${LOG_DIR}/manifest.csv"

EP=contact_ee_static_hold_001

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 15 \
  --residual-offset-duration 2.0 \
  --contact-intervals-json '[[5.0, 8.0]]' \
  --progress-period 1.0

echo "${EP},contact,ee gentle push during static hold,${LOG_DIR}/${EP}.csv,pc,\"[[5.0,8.0]]\",manual interval from recorder timer" >> "${LOG_DIR}/manifest.csv"
```

실행 중 터미널에는 다음처럼 현재 시간이 표시된다.

```text
[timer] t=  4.0s / 15.0s | NO CONTACT; contact starts at 5.0s | rows=400
[timer] t=  5.0s / 15.0s | CONTACT until 8.0s | rows=500
[timer] t=  8.0s / 15.0s | NO CONTACT; contact finished | rows=800
```

이 옵션을 쓰면 CSV 마지막 column인 `contact_label`이 contact interval 안에서는 `1`, 나머지는 `0`으로 저장된다. 즉 나중에 별도로 시간을 다시 맞추지 않아도 된다.
