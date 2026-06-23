# 실제 로봇 contact/no-contact 데이터 수집 Runbook

이 문서는 실제 로봇 구현용 contact detector를 학습하기 위한 CSV 수집 명령만 따로 정리한 파일이다.
시뮬레이션 논문용 학습과 섞지 않고, 실제 로봇 `outputs_real/` 실험을 위한 데이터를 모은다.

## 현재 수집 상태

파일 기준 현재 상태는 다음과 같다.

```text
No-contact
static_hold:              6개
home_bridge_hold:         001~003
slow_sine_j2:             001
slow_sine_j3:             001, near_current_001
slow_sine_j23:            001~002

Contact
static_hold:              001~010
home_bridge_hold:         001~010
slow_sine_j2:             001~002
slow_sine_j3:             001~002
slow_sine_j23:            001~006
```

80 episode를 목표로 할 때 이어서 저장할 번호는 다음이다.

```text
No-contact 추가 권장:
no_contact_home_bridge_hold_004~008
slow_sine_j2_002~004
slow_sine_j3_002~004
slow_sine_j23_003~006

Contact 추가 권장:
contact_ee_slow_sine_j2_003~006
contact_ee_slow_sine_j3_003~006
contact_ee_slow_sine_j23_007~008  # optional, 001~006은 이미 있음
```

## 공통 원칙

터미널마다 먼저 실행한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Terminal 1에는 CAN bridge를 켠다.

```bash
ros2 run can_interface can_bridge_node
```

real log 기록은 실제 CAN으로 나간 명령 기준 residual을 계산해야 하므로 항상 다음 옵션을 사용한다.

```bash
--cmd-topic /motor_cmd_array_applied
```

`/motor_cmd_array`는 command publisher가 요청한 값이고, `/motor_cmd_array_applied`는 `can_bridge`가 실제로 CAN에 내보낸 값을 다시 publish한 것이다.
토크 clamp, home fallback, safe default가 있으면 두 값이 달라질 수 있으므로 학습 CSV에는 applied command가 더 맞다.

## No-Contact: Home Bridge Hold

명령 publisher는 따로 켜지 않는다. CAN bridge의 home hold 상태에서 사람 손을 대지 않고 기록한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/no_contact/${TODAY}
mkdir -p "${LOG_DIR}"
[ -f "${LOG_DIR}/manifest.csv" ] || printf "episode_id,type,description,csv_path,label,notes\n" > "${LOG_DIR}/manifest.csv"

EP=no_contact_home_bridge_hold_004

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 15 \
  --residual-offset-duration 2.0 \
  --progress-period 1.0 \
  --cmd-topic /motor_cmd_array_applied

echo "${EP},home_bridge_hold,no contact during can_bridge home PD hold,${LOG_DIR}/${EP}.csv,nc,can_bridge internal home policy; tau_ff=0; no human contact" >> "${LOG_DIR}/manifest.csv"
```

다음 episode는 `EP`만 바꾼다.

```text
no_contact_home_bridge_hold_005
no_contact_home_bridge_hold_006
no_contact_home_bridge_hold_007
no_contact_home_bridge_hold_008
```

## No-Contact: Slow Sine

Terminal 2에서 motion node를 켠 뒤, Terminal 3에서 recorder를 실행한다.
기록 중에는 사람 손을 대지 않는다.

Terminal 2, j2:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy joint_sweep_node --ros-args \
  --params-file ~/idle_ws/install/phy/share/phy/config/slow_sine_j2.yaml
```

Terminal 2, j3:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy joint_sweep_node --ros-args \
  --params-file ~/idle_ws/install/phy/share/phy/config/slow_sine_j3.yaml
```

Terminal 2, j2+j3:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy joint_sweep_node --ros-args \
  --params-file ~/idle_ws/install/phy/share/phy/config/slow_sine_j23.yaml
```

Terminal 3 recorder:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/no_contact/${TODAY}
mkdir -p "${LOG_DIR}"
[ -f "${LOG_DIR}/manifest.csv" ] || printf "episode_id,type,description,csv_path,label,notes\n" > "${LOG_DIR}/manifest.csv"

EP=slow_sine_j23_003

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 20 \
  --residual-offset-duration 2.0 \
  --progress-period 1.0 \
  --cmd-topic /motor_cmd_array_applied

echo "${EP},slow_joint,j2+j3 slow sine,${LOG_DIR}/${EP}.csv,nc,no human contact" >> "${LOG_DIR}/manifest.csv"
```

`EP`와 설명만 motion에 맞게 바꾼다.

```text
j2 no-contact:
EP=slow_sine_j2_002
description=j2 slow sine

j3 no-contact:
EP=slow_sine_j3_002
description=j3 slow sine

j2+j3 no-contact:
EP=slow_sine_j23_003
description=j2+j3 slow sine
```

## Contact: Slow Sine

Terminal 2에서 원하는 slow sine motion을 켜고, Terminal 3에서 recorder를 실행한다.
slow sine contact는 앞으로 `5~13초`를 contact interval로 둔다.

```text
0~5초: 손 안 댐
5~6초: 천천히 힘 주기 시작
6~12초: 저항 유지
12~13초: 천천히 힘 빼기
13~20초: 손 안 댐
```

Terminal 3 recorder:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

TODAY=$(date +%Y%m%d)
LOG_DIR=contact_detection/real_logs/contact/${TODAY}
mkdir -p "${LOG_DIR}"
[ -f "${LOG_DIR}/manifest.csv" ] || printf "episode_id,type,description,csv_path,label,contact_intervals,notes\n" > "${LOG_DIR}/manifest.csv"

EP=contact_ee_slow_sine_j23_007

python3 contact_detection/record_real_log.py \
  --csv "${LOG_DIR}/${EP}.csv" \
  --sample-hz 100 \
  --duration 20 \
  --residual-offset-duration 2.0 \
  --contact-intervals-json '[[5.0, 13.0]]' \
  --progress-period 1.0 \
  --cmd-topic /motor_cmd_array_applied

echo "${EP},contact,ee push during j2+j3 slow sine,${LOG_DIR}/${EP}.csv,pc,\"[[5.0,13.0]]\",slow sine contact; manual interval" >> "${LOG_DIR}/manifest.csv"
```

`EP`와 설명만 motion에 맞게 바꾼다.

```text
j2 contact:
EP=contact_ee_slow_sine_j2_003
description=ee push during j2 slow sine

j3 contact:
EP=contact_ee_slow_sine_j3_003
description=ee push during j3 slow sine

j2+j3 contact:
EP=contact_ee_slow_sine_j23_007
description=ee push during j2+j3 slow sine
```

이미 기록한 `contact_ee_slow_sine_j2_001~002`, `contact_ee_slow_sine_j3_001~002`, `contact_ee_slow_sine_j23_001~006`은 그대로 둔다.
새로 추가하는 slow sine contact는 가능하면 `[[5.0, 13.0]]` interval을 사용한다.

## 학습 실행

수집 후 real detector sanity training은 두 갈래로 본다.

1. `residual_cmd_v1` 또는 `real_cmd_error_v1` 기반 real-only detector
2. 포스터 흐름용 `sim GRU pretrain -> real fine-tune`

첫 번째는 실제 로봇 마찰/중력보상 오차를 residual feature로 보기 위한 실험이다. 두 번째는 기존 시뮬레이션 기반 detector 위에 실제 로봇 데이터를 얹는 흐름을 보여주기 위한 실험이다.

### Real-only residual sanity training

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 contact_detection/train_real_detector.py \
  --epochs 35 \
  --batch-size 128 \
  --hidden-dim 64 \
  --dropout 0.1 \
  --feature-mode residual_cmd_v1 \
  --model both
```

출력은 다음에 저장된다.

```text
contact_detection/outputs_real/home_bridge_hold_residual_cmd_v1/
```

### Poster용: Sim GRU Pretrain -> Real Fine-Tune

포스터에서는 다음 흐름으로 설명한다.

```text
simulation pretraining:
  tau_ext command label이 있는 sim dataset으로 GRU 학습

real fine-tuning:
  같은 original_42 feature를 사용해서 실제 로봇 manual label CSV로 추가 학습

real validation:
  held-out real episodes에서 nc/pc confusion, precision, recall, F1, FPR 확인
```

중요한 점은 fine-tuning은 feature dimension과 의미가 같아야 한다는 것이다. 따라서 sim GRU가 `original_42`로 학습되었으면 real fine-tuning도 `--feature-mode original_42`를 사용한다. residual feature는 별도 real adaptation 실험으로 해석한다.

기본 sim GRU checkpoint에서 real fine-tuning:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

PYTHONUNBUFFERED=1 python3 contact_detection/train_real_detector.py \
  --output-dir contact_detection/outputs_real/finetune_sim_gru_original42_real_20260619 \
  --pretrained-checkpoint contact_detection/outputs_legacy_gru_mlp/randomized_sim/models/gru_detector.pt \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260618/static_hold_*.csv' \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260619/no_contact_home_bridge_hold_*.csv' \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260619/slow_sine_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_static_hold_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_home_bridge_hold_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_slow_sine_*.csv' \
  --feature-mode original_42 \
  --model gru \
  --split-policy stratified_by_condition \
  --epochs 20 \
  --batch-size 512 \
  --lr 0.0002 \
  --weight-decay 0.001 \
  --window-length 30 \
  --stride 1 \
  --transition-exclusion-s 0.5 \
  --exclude-transition-val \
  --seed 7
```

Optuna로 가장 높게 나온 sim GRU checkpoint를 fine-tuning하려면 checkpoint만 바꾼다.

```bash
--pretrained-checkpoint contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__gru_014/models/gru_detector.pt
```

false positive를 줄이는 trigger 관점의 threshold 정책까지 같이 보려면 아래 옵션을 추가한다.

```bash
  --threshold-selection-policy fpr_constrained_f1 \
  --max-validation-fpr 0.05
```

결과 확인:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("contact_detection/outputs_real/finetune_sim_gru_original42_real_20260619/metrics/real_train_summary.json")
d = json.loads(p.read_text())
row = d["models"]["gru"]
m = row["validation_metrics"]
print("pretrained:", d["pretrained_checkpoint"])
print("feature_mode:", d["feature_mode"])
print("best_epoch:", row["best_epoch"])
print("threshold:", row["decision_threshold"])
print("precision:", m["precision"])
print("recall:", m["recall"])
print("f1:", m["f1"])
print("fpr:", m["false_positive_rate"])
print("confusion:", row["confusion_matrix"])
PY
```

### Real-specific ablation: e_q-free GRU

실물 live test에서 아무 접촉이 없는데도 `P(contact)`가 높게 튀면, 먼저 `e_q`와 `delta_e_q`를 feature에서 제거한 모델을 확인한다.
이 실험은 `original_42` sim fine-tuning과는 별도다. 즉, feature dimension이 달라지므로 sim checkpoint를 그대로 이어받는 실험이 아니라, 실제 로봇 로그 기반 real-specific ablation으로 해석한다.

1차 실험은 `e_q`, `delta_e_q`만 제거하고 `delta_qdot`은 유지한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

PYTHONUNBUFFERED=1 python3 contact_detection/train_real_detector.py \
  --output-dir contact_detection/outputs_real/full_real_no_eq_v1_gru_20260619 \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260618/static_hold_*.csv' \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260619/no_contact_home_bridge_hold_*.csv' \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260619/slow_sine_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_static_hold_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_home_bridge_hold_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_slow_sine_*.csv' \
  --feature-mode real_no_eq_v1 \
  --model gru \
  --split-policy stratified_by_condition \
  --epochs 25 \
  --batch-size 512 \
  --hidden-dim 32 \
  --num-layers 1 \
  --dropout 0.25 \
  --lr 0.0005 \
  --weight-decay 0.001 \
  --window-length 30 \
  --stride 1 \
  --transition-exclusion-s 0.5 \
  --exclude-transition-val \
  --threshold-selection-policy fpr_constrained_f1 \
  --max-validation-fpr 0.05 \
  --seed 7
```

그래도 live에서 `delta_qdot`이 너무 민감하게 튀면, 2차로 `delta_qdot`까지 제거한 버전을 비교한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

PYTHONUNBUFFERED=1 python3 contact_detection/train_real_detector.py \
  --output-dir contact_detection/outputs_real/full_real_no_eq_no_dqdot_v1_gru_20260619 \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260618/static_hold_*.csv' \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260619/no_contact_home_bridge_hold_*.csv' \
  --no-contact-glob 'contact_detection/real_logs/no_contact/20260619/slow_sine_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_static_hold_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_home_bridge_hold_*.csv' \
  --contact-glob 'contact_detection/real_logs/contact/20260619/contact_ee_slow_sine_*.csv' \
  --feature-mode real_no_eq_no_dqdot_v1 \
  --model gru \
  --split-policy stratified_by_condition \
  --epochs 25 \
  --batch-size 512 \
  --hidden-dim 32 \
  --num-layers 1 \
  --dropout 0.25 \
  --lr 0.0005 \
  --weight-decay 0.001 \
  --window-length 30 \
  --stride 1 \
  --transition-exclusion-s 0.5 \
  --exclude-transition-val \
  --threshold-selection-policy fpr_constrained_f1 \
  --max-validation-fpr 0.05 \
  --seed 7
```

학습 결과 확인:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for name in [
    "full_real_no_eq_v1_gru_20260619",
    "full_real_no_eq_no_dqdot_v1_gru_20260619",
]:
    p = Path("contact_detection/outputs_real") / name / "metrics" / "real_train_summary.json"
    if not p.exists():
        print(name, "not trained yet")
        continue
    d = json.loads(p.read_text())
    row = d["models"]["gru"]
    m = row["validation_metrics"]
    print("\n", name)
    print("feature_mode:", d["feature_mode"])
    print("best_epoch:", row["best_epoch"])
    print("threshold:", row["decision_threshold"])
    print("precision/recall/f1:", m["precision"], m["recall"], m["f1"])
    print("fpr:", m["false_positive_rate"])
    print("confusion:", row["confusion_matrix"])
PY
```

live robot에서 확인할 때는 학습된 모델 경로만 바꿔서 실행한다.

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run phy contact_detector_node --ros-args \
  -p model_path:=$HOME/idle_ws/contact_detection/outputs_real/full_real_no_eq_v1_gru_20260619/models/gru_detector.pt \
  -p scaler_path:=$HOME/idle_ws/contact_detection/outputs_real/full_real_no_eq_v1_gru_20260619/models/scaler.pkl \
  -p command_topic:=/motor_cmd_array_applied \
  -p state_topic:=/motor_state_array \
  -p inference_hz:=100.0 \
  -p debug_log_enabled:=true \
  -p debug_log_period_s:=1.0 \
  -p rate_log_period_s:=3.0
```

포스터 해석 문장:

```text
On the real robot, q/e_q-based features can overreact to steady-state tracking residuals and command jitter caused by friction and imperfect gravity compensation.
Therefore, we additionally evaluate an e_q-free proprioceptive feature ablation for real-robot deployment.
This analysis separates the simulation-trained detector result from a practical real-robot robustness study.
```

### 30D/24D 모델 비교 그림 만들기

같은 real CSV에 여러 모델을 적용해서 probability overlay, nc/pc confusion matrix, summary table을 만든다.
이 비교는 checkpoint/threshold 선택용이 아니라 poster/debug용이다.

대표 slow sine j2+j3 contact episode:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 contact_detection/compare_real_models.py \
  --csv contact_detection/real_logs/contact/20260619/contact_ee_slow_sine_j23_006.csv \
  --contact-intervals-json '[[5.0,13.0]]' \
  --output-dir contact_detection/outputs_real/model_compare_20260619/j23_006
```

저장 결과:

```text
contact_detection/outputs_real/model_compare_20260619/j23_006/metrics/real_model_comparison_summary.json
contact_detection/outputs_real/model_compare_20260619/j23_006/metrics/real_model_comparison_table.csv
contact_detection/outputs_real/model_compare_20260619/j23_006/figures/real_model_probability_overlay.png
contact_detection/outputs_real/model_compare_20260619/j23_006/figures/real_model_confusion_comparison.png
```

현재 `j23_006` 기준 비교:

```text
30D real_no_eq_v1:
  F1=0.600, Precision=0.531, Recall=0.690, FPR=0.416

24D real_no_eq_no_dqdot_v1:
  F1=0.623, Precision=0.603, Recall=0.645, FPR=0.290
```

해석:

```text
24D 모델은 delta_qdot을 제거하면서 slow sine j2+j3 episode에서 no-contact false positive를 줄였다.
다만 contact recall은 30D보다 약간 낮아져, false alarm 감소와 missed contact 증가 사이의 trade-off가 보인다.
```

### 24D Optuna tuning

현재 live robot trigger 후보는 `real_no_eq_no_dqdot_v1` 24D 모델로 둔다.
고정 hyperparameter 모델은 sanity check용이므로, 실제 적용 후보로 쓰기 전 validation-only Optuna tuning을 한 번 돌린다.

중요:

```text
Optuna ranking은 real validation split만 사용한다.
live robot에서 잘 보였는지, 특정 review CSV에서 잘 나왔는지는 tuning 선택 기준으로 쓰지 않는다.
```

실행:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

PYTHONUNBUFFERED=1 python3 contact_detection/tune_real_detector.py \
  --study-name real_24d_optuna_20260619 \
  --feature-mode real_no_eq_no_dqdot_v1 \
  --n-trials 20 \
  --epochs 25 \
  --sampler optuna \
  --require-optuna
```

시간이 부담되면 먼저 8~12 trial만 돌린다.

```bash
PYTHONUNBUFFERED=1 python3 contact_detection/tune_real_detector.py \
  --study-name real_24d_optuna_20260619_quick \
  --feature-mode real_no_eq_no_dqdot_v1 \
  --n-trials 12 \
  --epochs 20 \
  --sampler optuna \
  --require-optuna
```

튜닝 결과 확인:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("contact_detection/outputs_real/hparam_tuning/real_24d_optuna_20260619/real_hparam_tuning_summary.json")
d = json.loads(p.read_text())
b = d["best_by_validation_only"]
print("best trial:", b["trial_index"])
print("feature_mode:", d["feature_mode"])
print("F1:", b["validation_f1"])
print("precision:", b["validation_precision"])
print("recall:", b["validation_recall"])
print("FPR:", b["validation_false_positive_rate"])
print("threshold:", b["decision_threshold"])
print("model_path:", b["model_path"])
print("scaler_path:", b["scaler_path"])
print("params:", b["params"])
PY
```

best trial을 live robot에 적용:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

MODEL=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("contact_detection/outputs_real/hparam_tuning/real_24d_optuna_20260619/real_hparam_tuning_summary.json")
print(json.loads(p.read_text())["best_by_validation_only"]["model_path"])
PY
)
SCALER=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("contact_detection/outputs_real/hparam_tuning/real_24d_optuna_20260619/real_hparam_tuning_summary.json")
print(json.loads(p.read_text())["best_by_validation_only"]["scaler_path"])
PY
)

ros2 run phy contact_detector_node --ros-args \
  -p model_path:="${MODEL}" \
  -p scaler_path:="${SCALER}" \
  -p command_topic:=/motor_cmd_array_applied \
  -p state_topic:=/motor_state_array \
  -p inference_hz:=100.0 \
  -p debug_log_enabled:=true \
  -p debug_log_period_s:=1.0 \
  -p rate_log_period_s:=3.0
```

튜닝 best와 fixed 24D/30D를 같은 CSV에서 비교하려면 `compare_real_models.py --model ...` 옵션을 반복해서 넣는다.

```bash
BEST_MODEL=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("contact_detection/outputs_real/hparam_tuning/real_24d_optuna_20260619/real_hparam_tuning_summary.json")
print(json.loads(p.read_text())["best_by_validation_only"]["model_path"])
PY
)
BEST_SCALER=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("contact_detection/outputs_real/hparam_tuning/real_24d_optuna_20260619/real_hparam_tuning_summary.json")
print(json.loads(p.read_text())["best_by_validation_only"]["scaler_path"])
PY
)

python3 contact_detection/compare_real_models.py \
  --csv contact_detection/real_logs/contact/20260619/contact_ee_slow_sine_j23_006.csv \
  --contact-intervals-json '[[5.0,13.0]]' \
  --output-dir contact_detection/outputs_real/model_compare_20260619/j23_006_with_optuna \
  --model 30D_fixed contact_detection/outputs_real/full_real_no_eq_v1_gru_20260619/models/gru_detector.pt contact_detection/outputs_real/full_real_no_eq_v1_gru_20260619/models/scaler.pkl \
  --model 24D_fixed contact_detection/outputs_real/full_real_no_eq_no_dqdot_v1_gru_20260619/models/gru_detector.pt contact_detection/outputs_real/full_real_no_eq_no_dqdot_v1_gru_20260619/models/scaler.pkl \
  --model 24D_optuna "${BEST_MODEL}" "${BEST_SCALER}"
```

### 공정한 30D vs 24D Optuna 비교

feature ablation 자체를 공정하게 비교하려면 30D와 24D를 둘 다 같은 trial 수/epoch/validation policy로 튜닝한다.
그 다음 각 feature mode의 validation-best 모델끼리 비교한다.

```text
나쁜 비교:
  fixed 30D vs Optuna 24D

공정한 비교:
  Optuna-best 30D vs Optuna-best 24D
```

30D Optuna:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

PYTHONUNBUFFERED=1 python3 contact_detection/tune_real_detector.py \
  --study-name real_30d_optuna_20260619 \
  --feature-mode real_no_eq_v1 \
  --n-trials 20 \
  --epochs 25 \
  --sampler optuna \
  --require-optuna
```

24D Optuna:

```bash
cd ~/idle_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

PYTHONUNBUFFERED=1 python3 contact_detection/tune_real_detector.py \
  --study-name real_24d_optuna_20260619 \
  --feature-mode real_no_eq_no_dqdot_v1 \
  --n-trials 20 \
  --epochs 25 \
  --sampler optuna \
  --require-optuna
```

각 best 모델 경로 확인:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for study in ["real_30d_optuna_20260619", "real_24d_optuna_20260619"]:
    p = Path("contact_detection/outputs_real/hparam_tuning") / study / "real_hparam_tuning_summary.json"
    d = json.loads(p.read_text())
    b = d["best_by_validation_only"]
    print("\n", study)
    print("feature_mode:", d["feature_mode"])
    print("trial:", b["trial_index"])
    print("F1:", b["validation_f1"])
    print("precision:", b["validation_precision"])
    print("recall:", b["validation_recall"])
    print("FPR:", b["validation_false_positive_rate"])
    print("model:", b["model_path"])
    print("scaler:", b["scaler_path"])
PY
```

Optuna-best 30D vs Optuna-best 24D를 같은 CSV에서 비교:

```bash
BEST30_MODEL=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("contact_detection/outputs_real/hparam_tuning/real_30d_optuna_20260619/real_hparam_tuning_summary.json")
print(json.loads(p.read_text())["best_by_validation_only"]["model_path"])
PY
)
BEST30_SCALER=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("contact_detection/outputs_real/hparam_tuning/real_30d_optuna_20260619/real_hparam_tuning_summary.json")
print(json.loads(p.read_text())["best_by_validation_only"]["scaler_path"])
PY
)
BEST24_MODEL=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("contact_detection/outputs_real/hparam_tuning/real_24d_optuna_20260619/real_hparam_tuning_summary.json")
print(json.loads(p.read_text())["best_by_validation_only"]["model_path"])
PY
)
BEST24_SCALER=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("contact_detection/outputs_real/hparam_tuning/real_24d_optuna_20260619/real_hparam_tuning_summary.json")
print(json.loads(p.read_text())["best_by_validation_only"]["scaler_path"])
PY
)

python3 contact_detection/compare_real_models.py \
  --csv contact_detection/real_logs/contact/20260619/contact_ee_slow_sine_j23_006.csv \
  --contact-intervals-json '[[5.0,13.0]]' \
  --output-dir contact_detection/outputs_real/model_compare_20260619/j23_006_optuna_best \
  --model 30D_optuna "${BEST30_MODEL}" "${BEST30_SCALER}" \
  --model 24D_optuna "${BEST24_MODEL}" "${BEST24_SCALER}"
```

포스터에서는 이 비교를 이렇게 표현한다.

```text
Both 30D and 24D real-robot GRU candidates were tuned with the same Optuna budget and selected using validation-only F1 under an FPR-constrained threshold policy.
The 24D feature removes delta_qdot to reduce motion-induced false positives, while the 30D feature retains delta_qdot and may improve sensitivity at the cost of more false alarms.
```

포스터 해석 문장:

```text
The GRU detector was first trained in simulation using external torque command labels.
For real-robot adaptation, the same sensorless 42D proprioceptive feature vector was used to fine-tune the simulation-pretrained GRU with manually labeled real robot logs.
This evaluates the sim-to-real gap caused by friction, gravity compensation error, and steady-state tracking residuals.
Residual-torque features were additionally explored as a real adaptation direction, but they are treated separately from the direct sim-pretrain fine-tuning experiment.
```

## 주의

- sample 단위로 섞지 말고 episode 단위로 나중에 split한다.
- contact interval은 weak label이다. 손을 천천히 대고 천천히 뗀다.
- 실수한 episode는 지우지 말고 manifest notes에 `timing uncertain`, `too strong`, `released fast`처럼 적는 편이 낫다.
- `tau_ext`나 외부 force ground truth는 실제 모델 입력으로 쓰지 않는다.
