# 센서리스 접촉 검출 실험 README

이 폴더는 **F/T 센서 없이 QDD 6-DOF 매니퓰레이터에서 contact / no-contact를 검출**하기 위한 시뮬레이션 기반 학습 파이프라인이다.

현재 코드는 기존 논문 흐름을 유지하면서, 비교 대상을 다음처럼 확장한다.

```text
기존: Threshold baseline vs GRU detector
확장: Threshold baseline vs MLP baseline vs GRU detector
```

전체 실행 흐름은 다음 순서다.

```text
1. generate_sim_dataset.py
2. diagnose_dataset.py
3. train_detectors.py
4. evaluate_detectors.py
```

각 스크립트가 논문/실험의 어느 단계에 해당하는지는 [PAPER_PIPELINE.md](/home/eomyunbeen/idle_ws/contact_detection/PAPER_PIPELINE.md)에 더 자세히 정리되어 있다.

## 1. 실험 목적

### 연구 목표

목표는 실제 F/T sensor 없이, QDD 기반 6자유도 로봇팔에서 접촉 여부를 이진 분류하는 것이다.

```text
output = contact 또는 no-contact
```

이 코드는 외력 크기 추정이나 접촉 위치 추정이 아니라, 접촉 여부만 판단하는 binary detector를 만든다.

### 왜 시뮬레이션을 쓰는가?

실제 로봇에서는 정확한 접촉 시점과 외력 크기에 대한 ground-truth label을 얻기 어렵다.

그래서 이 프로젝트에서는 URDF 기반 명목 동역학 시뮬레이션에서 외란 토크 `tau_ext`를 주입하고, 그 값을 이용해 contact label을 만든다.

중요한 점은 다음이다.

```text
tau_ext는 label 생성과 분석에만 사용한다.
tau_ext는 모델 입력에 절대 포함하지 않는다.
```

### 왜 MLP baseline을 추가했는가?

기존 논문은 Threshold와 GRU를 비교했다.

이번 확장은 GRU가 좋은 이유를 더 잘 분리해서 보기 위한 것이다.

```text
Threshold:
  단순 tracking-error 기반 rule baseline

MLP:
  현재 시점의 42D feature만 사용하는 learning-based baseline

GRU:
  최근 30 sample의 시간적 패턴을 보는 sequence model
```

해석 목적은 다음과 같다.

```text
MLP가 Threshold보다 좋다:
  learning-based detector가 단순 threshold보다 정상 추종 오차와 접촉을 더 잘 구분한다.

GRU가 MLP보다 좋다:
  시간적 패턴 학습이 contact/no-contact 구분에 도움이 된다.

MLP와 GRU가 비슷하다:
  현재 시점 feature만으로도 상당히 구분 가능하며, MLP가 더 가벼운 실시간 후보가 될 수 있다.
```

### Main experiment와 low-torque analysis 구분

현재 실험은 두 종류로 나눈다.

```text
randomized_sim:
  기존 논문 조건에 가까운 main/original-range experiment

low_torque_analysis:
  약한 외란 조건에서 detector 민감도를 보는 추가 분석
```

`low_torque_analysis` 결과는 논문 메인 결과를 대체하지 않는다. 약한 접촉에서 recall과 false positive trade-off를 내부적으로 확인하기 위한 추가 분석이다.

## 2. 데이터와 label 정의

### Label 정의

시뮬레이션 label은 다음 기준으로 만든다.

```text
label_t = 1 if ||tau_ext,t|| > epsilon else 0
```

여기서 `tau_ext`는 시뮬레이션에서 주입한 외란 토크다.

`tau_ext`는 다음 용도로만 사용한다.

```text
label 생성
dataset diagnosis
토크 크기 구간별 분석
결과 해석용 확인
```

`tau_ext`는 다음 경로에는 들어가지 않는다.

```text
feature matrix
scaler fitting
MLP input
GRU input
threshold score
real robot inference input
ROS online detector input
```

### Feature 정의

현재 기본 입력 feature는 다음이다.

```text
x_t = [q_t, qdot_t, e_q,t, tau_cmd,t, delta_e_q,t, delta_qdot_t, delta_tau_cmd,t]
e_q,t = q_d,t - q_t
```

6-DOF 기준 feature dimension은 다음과 같다.

```text
6 joints * 7 feature blocks = 42
```

각 항의 의미는 다음과 같다.

```text
q:
  joint position

qdot:
  joint velocity

e_q:
  desired position과 실제 position 사이의 tracking error

tau_cmd:
  commanded torque 또는 controller input

delta_e_q:
  episode 내부에서 계산한 tracking error difference

delta_qdot:
  episode 내부에서 계산한 velocity difference

delta_tau_cmd:
  episode 내부에서 계산한 commanded torque difference
```

feature를 만드는 핵심 함수는 다음이다.

```text
utils.py::build_input_features
```

이 함수는 인자로 `tau_ext`를 받지 않는다. 그래서 `tau_ext`가 feature/scaler/model로 흘러 들어가는 경로가 없다.

### 실제 로봇 residual feature 확장

실제 로봇 로그에서는 `record_real_log.py`가 `tau_meas`, `tau_ff`, `kp`, `kd`를 진단용으로 함께 저장한다. 기본 42D feature 모델은 기존 논문 흐름을 유지하기 위해 그대로 둔다.

실제 로봇 residual 실험을 할 때의 기본 정의는 다음이다.

```text
tau_cmd = kp * (q_des - q) + kd * (qdot_des - qdot) + tau_ff
tau_residual = tau_meas - tau_cmd
tau_residual_corrected = tau_residual - episode 초기 no-contact 평균 offset
```

현재 ROS command 구조에서는 `tau_ff`가 이미 중력보상을 포함한다. 따라서 residual을 만들 때 gravity를 한 번 더 더하지 않는다.

지원되는 feature mode는 다음이다.

```text
original_42:
  기존 42D feature. 기본값.

residual_v1:
  [qdot, tau_cmd, tau_residual_corrected, delta_tau_residual, delta_e_q, delta_qdot]

residual_v2:
  [tau_residual_corrected, delta_tau_residual, qdot, delta_qdot]

residual_v3:
  [q, qdot, tau_cmd, tau_residual_corrected, delta_tau_residual, delta_e_q, delta_qdot]

residual_cmd_v1:
  [tau_cmd, delta_tau_cmd, tau_residual_corrected, delta_tau_residual, qdot, delta_qdot]
```

config에서는 다음처럼 선택한다.

```yaml
dataset:
  feature_mode: original_42
```

중요한 해석은 다음이다.

```text
tau_ext는 label/분석용이다.
tau_ext는 residual feature에도 넣지 않는다.
residual feature는 실제 로봇에서 읽을 수 있는 tau_meas와 command 정보로만 만든다.
```

현재 실제 로봇 단계에서는 `original_42` 모델을 무리하게 튜닝하기보다, 먼저 real no-contact CSV의 residual 분포를 확인한 뒤 residual 기반 모델로 확장하는 것이 더 타당하다. 실제 로봇은 마찰과 중력보상 오차 때문에 목표 관절값에 완전히 도달하지 못할 수 있고, 이 경우 raw `e_q = q_des - q`는 contact가 아니라 정상 잔류 오차를 크게 반영할 수 있다.

따라서 실물 적용용 1차 후보 feature는 `e_q`를 직접 넣지 않는 residual 계열로 둔다.
초기 home PD hold 데이터에서는 `tau_residual_corrected`만으로 contact/no-contact가 잘 분리되지 않았고,
사람이 밀 때 controller가 버티면서 `tau_cmd` 변화가 더 뚜렷하게 나타났다. 그래서 첫 real sanity training은
`residual_cmd_v1`을 사용한다.

```text
residual_cmd_v1:
  [tau_cmd, delta_tau_cmd, tau_residual_corrected, delta_tau_residual, qdot, delta_qdot]
```

`original_42`는 기존 시뮬레이션 논문 흐름과 비교 baseline으로 유지한다. `residual_v2/v3` 모델은 real contact positive CSV까지 확보한 뒤 별도 artifact로 학습/평가한다.

실제 로봇 home PD hold 조건에서 수집한 CSV로만 sanity training을 돌리는 명령은 다음이다.

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

현재 첫 sanity run의 저장 위치는 다음이다.

```text
contact_detection/outputs_real/home_bridge_hold_residual_cmd_v1/
```

이 run은 test set을 따로 주장하지 않고, episode-wise train/validation split으로만 확인한 초기 실험이다.
validation F1 기준 best checkpoint와 threshold를 선택했다.

```text
GRU validation:
  Precision = 0.909
  Recall    = 0.770
  F1        = 0.834
  FPR       = 0.020

MLP validation:
  Precision = 0.936
  Recall    = 0.757
  F1        = 0.837
  FPR       = 0.013
```

해석:
이 결과는 “실제 로봇 residual/tau_cmd 기반 신호가 contact/no-contact를 어느 정도 분리할 수 있다”는 sanity check이다.
아직 데이터가 작고 조건도 home PD hold에 치우쳐 있으므로 논문식 최종 성능으로 해석하면 안 된다.
다음 단계는 slow sine/contact, PTP/contact까지 추가해서 real validation 조건을 넓히는 것이다.

### 실제 로봇 데이터 수집 runbook

실제 로봇 구현용 데이터 수집 명령은 별도 파일로 분리했다.

```text
contact_detection/REAL_DATA_COLLECTION_RUNBOOK.md
```

수집 중에는 이 파일만 보고 진행한다. 기존 시뮬레이션 논문용 학습 흐름과 섞지 않기 위해 real data 수집/학습 명령을 따로 관리한다.

현재 수집한 real no-contact 로그의 residual 분포는 다음 명령으로 확인한다.

```bash
python3 contact_detection/analyze_real_residuals.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --manifest contact_detection/real_logs/no_contact/20260618/manifest.csv \
  --manifest contact_detection/real_logs/no_contact/20260619/manifest.csv \
  --probability-summary-dir contact_detection/outputs_legacy_gru_mlp/randomized_sim/real_inference/no_contact_check
```

출력은 다음 위치에 저장된다.

```text
contact_detection/outputs_legacy_gru_mlp/randomized_sim/real_inference/no_contact_residual_analysis/
```

실물 no-contact 수집 절차와 명령어는 [REAL_NO_CONTACT_COLLECTION.md](/home/eomyunbeen/idle_ws/contact_detection/REAL_NO_CONTACT_COLLECTION.md)에 정리되어 있다.

### 추가 저장 metadata

새 데이터셋에는 평가와 분석을 위해 다음 metadata도 저장한다.

```text
trajectory_mode
active_event_id
active_event_magnitude
event_table_id
event_table_episode_id
event_table_mode
event_table_start_step
event_table_end_step
event_table_start_index
event_table_end_index
event_table_magnitude
event_table_kind
```

이 값들은 mode별 성능과 토크 크기 구간별 성능을 계산하기 위한 것이다. 모델 입력에는 사용하지 않는다.

## 3. 데이터 처리 원칙

### Episode-wise split

train / validation / test는 episode 단위로 생성한다.

sample 단위 random split은 사용하지 않는다. 같은 episode 안의 인접 sample은 서로 매우 비슷하기 때문에, sample 단위로 섞으면 temporal leakage가 생길 수 있다.

### GRU window

GRU 입력 window는 같은 episode 내부에서만 만든다.

```text
X_t = [x_{t-L+1}, ..., x_t]
y_t = label at timestep t
```

현재 기본값은 다음이다.

```text
window_length = 30
dt = 0.002 s
window time = 0.06 s
```

### Delta feature

delta feature도 episode 내부에서만 계산한다.

episode 첫 sample은 이전 sample이 없으므로 delta를 0으로 둔다.

### Scaler

`scaler.pkl`은 train split에만 fit한다.

validation / test / real log / ROS online inference에서는 저장된 train scaler를 transform 용도로만 사용한다.

## 4. 실험 조건

### Main/original-range experiment

main experiment는 다음 stage를 사용한다.

```text
stage = randomized_sim
```

기본 조건은 `config.yaml`에 있다.

```text
disturbance magnitude = 0.8 ~ 2.5 Nm
disturbance duration = 0.5 ~ 1.2 s
trajectory modes = hold, slow_sine
train / validation / test = 100 / 20 / 20 episodes
window length = 30
feature dimension = 42
```

### Low-torque analysis

약한 외란 분석은 다음 stage를 사용한다.

```text
stage = low_torque_analysis
```

기본 조건은 다음이다.

```text
disturbance magnitude = 0.2 ~ 1.0 Nm
disturbance duration = 0.5 ~ 1.2 s
trajectory modes = hold, slow_sine
```

이 결과는 main paper condition과 분리해서 해석해야 한다.

## 5. 모델 설명

### Threshold baseline

Threshold baseline은 단순 rule-based detector다.

```text
r_t = alpha * ||e_q|| + beta * ||delta_e_q||
alpha = 1.0
beta = 2.0
```

판단 기준은 다음이다.

```text
r_t > gamma 이면 contact
```

`gamma`는 validation set에서 선택한다. 기본 설정에서는 validation F1-score가 최대가 되는 값을 사용한다.

저장 파일은 다음이다.

```text
outputs/<stage>/models/threshold.json
```

### MLP baseline

MLP는 현재 시점의 42D feature만 사용하는 single-time learning baseline이다.

```text
input shape = (batch, 42)
output = binary logit 1개
```

MLP는 window를 보지 않는다.

다만 GRU와 공정하게 비교하기 위해, MLP도 GRU window의 마지막 시점 index와 같은 sample set만 사용한다.

저장 파일은 다음이다.

```text
outputs/<stage>/models/mlp_detector.pt
outputs/<stage>/metrics/mlp_train_log.json
```

### GRU detector

GRU는 최근 30 sample의 feature window를 입력으로 받는다.

```text
input shape = (batch, 30, 42)
output = binary logit 1개
```

GRU의 목적은 단일 시점 오차 크기만 보는 것이 아니라, 최근 상태와 명령 정보의 시간적 변화 패턴을 학습하는 것이다.

저장 파일은 기존과 동일하다.

```text
outputs/<stage>/models/gru_detector.pt
```

이 파일명은 real log inference와 ROS online detector가 기대하는 이름이므로 유지한다.

### 학습 설정

MLP와 GRU는 최대한 같은 조건으로 학습한다.

```text
loss = BCEWithLogitsLoss
pos_weight = N_negative / N_positive
optimizer = Adam 기본값
best checkpoint = validation F1-score 최고 epoch 기준
decision threshold = validation set에서 선택
test evaluation = validation에서 선택한 threshold 고정 사용
```

MLP는 GRU와 비교하기 위해 추가한 baseline이므로, MLP 학습이 GRU의 random initialization이나 shuffled batch order를 바꾸면 안 된다. 그래서 기본 설정에서는 각 모델 학습 직전에 seed를 다시 고정한다.

```yaml
training:
  isolate_model_random_seed: true
  mlp_seed: null
  gru_seed: null
  mlp_seed_offset: 0
  gru_seed_offset: 0
```

`mlp_seed`와 `gru_seed`가 `null`이면 config의 top-level `seed`를 그대로 사용한다. 이 설정 때문에 MLP를 단순 추가해도 GRU 학습 random state가 MLP 학습에 의해 밀리지 않는다.

여기서 두 가지를 구분해야 한다.

```text
model checkpoint 선택:
  어떤 epoch의 weight를 저장할지 결정한다.
  기본값은 val_f1이 가장 높은 epoch이다.
  val_f1이 완전히 같으면 val_loss가 더 낮은 epoch를 고른다.

decision threshold 선택:
  저장된 모델의 probability를 몇 이상이면 contact로 볼지 결정한다.
  validation set에서 F1-score 또는 config의 selection policy 기준으로 고른다.
```

Label delay ablation을 해석할 때는 threshold 기준을 특히 구분해야 한다.

각 ablation run의 checkpoint와 decision threshold는 해당 run의 selection label 기준 validation F1-score가 최대가 되도록 선택하였다. 이후 선택된 checkpoint와 동일 threshold를 고정한 상태에서 original external force command label 기준 validation/test sample-level metric과 event-level metric을 계산하였다. 따라서 original-label metric은 threshold를 다시 탐색한 결과가 아니라, 해당 label policy로 선택된 detector가 실제 command 기준 contact event에 대해 어떻게 동작하는지를 평가한 결과이다.

이 해석을 남기기 위해 metrics에는 다음 필드를 저장한다.

```text
decision_threshold_value
threshold_selection_metric
threshold_selected_on
checkpoint_selected_on
selection_label_basis
label_basis_for_training
label_basis_for_checkpoint_selection
label_basis_for_threshold_search
label_basis_for_original_evaluation
threshold_applied_to_original_label_metrics
original_label_metric_threshold_policy
```

### Event-level latency와 label ablation

sample-level F1-score는 각 window를 맞췄는지 보는 지표다. 하지만 real-time contact detector에서는 contact가 발생한 뒤 얼마나 빨리 감지하는지도 중요하다.

그래서 evaluator는 external force command onset을 기준으로 event-level latency를 추가 계산한다.

```text
force onset:
  event_table_start_index에 기록된 external force command 시작 시점

detection time:
  P(contact) >= decision_threshold가 K sample 연속 유지되기 시작한 시점

detection latency:
  detection time - force onset time
```

기본값은 다음이다.

```yaml
evaluation:
  event_detection_consecutive_samples: 3
  detection_margin_ms: 50
  feature_response_window_pre_ms: 100
  feature_response_window_post_ms: 200
```

event detection search interval은 force onset부터 force offset + `detection_margin_ms`까지다. false alarm은 contact event interval과 detection margin을 제외한 pure no-contact region에서 발생한 threshold-crossing segment 수로 계산한다.

Feature response analysis는 test dataset의 raw signal을 force onset 기준으로 정렬한다. 기본적으로 onset 전 100 ms, 후 200 ms window를 사용하며, 이 값은 `evaluation.feature_response_window_pre_ms`, `evaluation.feature_response_window_post_ms` 또는 evaluator CLI 옵션으로 바꿀 수 있다.

Label delay ablation은 external force command label과 실제 proprioceptive response 사이의 시간 차이를 보기 위한 실험이다.

```text
label_delay_ms = 0:
  기존 command label 그대로 학습

label_delay_ms > 0:
  학습/validation selection label을 해당 ms만큼 뒤로 shift
```

Transition-aware exclusion은 contact onset/release 주변의 애매한 transition window를 학습에서 제외하는 ablation이다.

```text
transition_exclusion_ms = 0:
  기존 방식과 동일

transition_exclusion_ms > 0:
  command label transition 주변 window를 train에서 제외
```

validation transition exclusion은 `--exclude-transition-val`을 줄 때만 적용한다. 기본 validation/test 평가는 전체 original command label 기준을 유지한다.

transition exclusion을 사용한 run에서는 test 전체 metric은 그대로 저장하고, transition 주변 window를 제외한 참고용 metric은 `sim_test_metrics.json`의 `transition_excluded_sample_metrics`에 별도로 저장한다.

중요한 선택 기준은 다음이다.

```text
checkpoint / threshold / ablation 추천:
  validation 기준만 사용

test set:
  최종 보고용 성능 확인에만 사용
```

train_loss는 계속 내려가는데 val_loss가 올라가면 과적합 신호일 수 있다. 다만 현재 실험의 checkpoint 선택 기준은 `checkpoint_selection_metric: "val_f1"`이다.

즉, `val_loss` warning은 참고용 경고이고 최종 test 평가에 쓰이는 모델은 validation F1이 가장 높았던 epoch의 checkpoint다.

관련 config key는 다음이다.

```yaml
training:
  optimizer: "adam"
  batch_size: 256
  epochs: 50
  lr: 0.001
  weight_decay: 0.0
  hidden_dim: 64
  num_layers: 1
  dropout: 0.1
  mlp_hidden_dim: 64
  mlp_num_layers: 2
  mlp_dropout: 0.1
  mlp_optimizer: "adam"
  checkpoint_selection_metric: "val_f1"
```

매 epoch마다 다음 값이 log에 저장된다.

```text
train_loss
val_loss
val_precision
val_recall
val_f1
val_threshold
```

best checkpoint의 epoch 번호와 metric 요약은 다음 파일에도 저장된다.

```text
outputs/<stage>/metrics/checkpoint_summary.json
```

test set은 checkpoint 선택, threshold 선택, early stopping에 절대 사용하지 않는다.

배포용으로 train+validation을 합쳐 final model을 다시 학습할 수도 있다. 이때는 앞선 실험에서 저장된 `best_epoch` 횟수만큼만 학습하고, decision threshold는 validation에서 고른 값을 그대로 사용한다.

이 final model은 논문/포스터 비교용 artifact를 덮어쓰지 않는다.

```text
outputs/<stage>/models/mlp_detector_trainval_final.pt
outputs/<stage>/models/gru_detector_trainval_final.pt
outputs/<stage>/metrics/final_trainval_summary.json
```

## 6. 평가 지표

`evaluate_detectors.py`는 다음 지표를 계산한다.

```text
Precision
Recall
F1-score
Accuracy
TP
FP
TN
FN
False positive rate
False negative rate
Detection delay
```

### Precision

모델이 contact라고 예측한 것 중 실제 contact인 비율이다.

Precision이 낮으면 false positive가 많다는 뜻이다.

### Recall

실제 contact 중 모델이 잡아낸 비율이다.

Recall이 낮으면 missed contact가 많다는 뜻이다.

### F1-score

Precision과 Recall의 균형을 보는 지표다.

기본 threshold 선택 기준으로 사용한다.

### Detection delay

contact event onset 이후 prediction이 처음 contact로 바뀌는 시점까지의 지연이다.

현재 코드는 episode 경계를 고려해서 delay를 계산한다.

### Mode별 성능

다음 결과를 따로 저장한다.

```text
전체 test set 성능
hold mode 성능
slow_sine mode 성능
hold / slow_sine 참고용 평균 성능
```

현재 실험 해석에서는 우선 `hold`와 `slow_sine` 각각의 성능을 보면 된다.

참고용 평균 성능은 두 mode를 한 줄로 요약하고 싶을 때만 보면 된다. 필수로 해석할 필요는 없다.

저장 파일은 다음이다.

```text
outputs/<stage>/metrics/mode_split_metrics.json
```

### 토크 크기 구간별 성능

여기서 `bin`은 어려운 말이 아니라 **구간**이라는 뜻이다.

예를 들어 `0.0~0.5 Nm`, `0.5~1.0 Nm`처럼 토크 크기를 몇 개 구간으로 나누고, 각 구간에서 detector 성능을 따로 보는 것이다.

이 프로젝트에서는 event 대표 `||tau_ext||` 값을 기준으로 토크 크기 구간을 나눈다.

기본 구간은 다음이다.

```yaml
evaluation:
  torque_bins:
    - [0.0, 0.5]
    - [0.5, 1.0]
    - [1.0, 1.5]
    - [1.5, 2.5]
```

각 구간에서는 다음을 계산한다.

```text
Precision
Recall
F1-score
TP
FP
TN
FN
```

저장 파일은 다음이다.

```text
outputs/<stage>/metrics/torque_bin_metrics.json
```

## 7. 결과 파일 구조

각 stage는 다음 구조로 결과를 저장한다.

```text
outputs/<stage>/
  datasets/
  models/
  metrics/
  figures/
  real_inference/
```

대표 파일은 다음이다.

```text
outputs/<stage>/experiment_config_used.yaml
outputs/<stage>/datasets/sim_train.npz
outputs/<stage>/datasets/sim_val.npz
outputs/<stage>/datasets/sim_test.npz
outputs/<stage>/models/scaler.pkl
outputs/<stage>/models/threshold.json
outputs/<stage>/models/mlp_detector.pt
outputs/<stage>/models/gru_detector.pt
outputs/<stage>/models/mlp_detector_trainval_final.pt
outputs/<stage>/models/gru_detector_trainval_final.pt
outputs/<stage>/metrics/dataset_summary.json
outputs/<stage>/metrics/dataset_diagnosis.json
outputs/<stage>/metrics/checkpoint_summary.json
outputs/<stage>/metrics/train_log.json
outputs/<stage>/metrics/mlp_train_log.json
outputs/<stage>/metrics/final_trainval_summary.json
outputs/<stage>/metrics/sim_test_metrics.json
outputs/<stage>/metrics/sim_test_metrics_trainval_final.json
outputs/<stage>/metrics/validation_original_metrics.json
outputs/<stage>/metrics/event_latency_metrics.json
outputs/<stage>/metrics/feature_response_analysis.json
outputs/<stage>/metrics/feature_response_aligned.csv
outputs/<stage>/metrics/confusion_matrix_summary.json
outputs/<stage>/metrics/ablation_summary.json
outputs/<stage>/metrics/mode_split_metrics.json
outputs/<stage>/metrics/torque_bin_metrics.json
outputs/<stage>/figures/training_curve.png
outputs/<stage>/figures/mlp_training_curve.png
outputs/<stage>/figures/sim_metric_bar.png
outputs/<stage>/figures/confusion_matrix_threshold.png
outputs/<stage>/figures/confusion_matrix_gru.png
outputs/<stage>/figures/confusion_matrix_mlp.png
outputs/<stage>/figures/confusion_matrix_comparison.png
outputs/<stage>/figures/gru_threshold_tradeoff.png
outputs/<stage>/figures/precision_recall_curve_mlp_gru.png
outputs/<stage>/figures/sim_prediction_example.png
outputs/<stage>/figures/sim_prediction_examples.png
outputs/<stage>/figures/feature_response_aligned.png
```

`experiment_config_used.yaml`은 해당 실험에서 실제 적용된 config를 저장한다. 나중에 같은 조건을 재현할 때 사용한다.

`*_trainval_final.*` 파일은 `--train-final-trainval` 또는 `--model-variant trainval_final`을 사용했을 때만 생성된다.

ablation 실행 결과는 기본 결과를 덮지 않고 다음처럼 분리 저장된다.

```text
outputs/<stage>/ablations/<ablation_tag>/models/
outputs/<stage>/ablations/<ablation_tag>/metrics/
outputs/<stage>/ablations/<ablation_tag>/figures/
```

## 8. 스크립트별 역할

### generate_sim_dataset.py

시뮬레이션 dataset을 만든다.

저장되는 주요 값은 다음이다.

```text
q
qdot
q_des
qdot_des
tau_cmd_raw
tau_cmd
tau_ext
label
episode_id
trajectory_mode
event metadata
```

### diagnose_dataset.py

학습 전에 dataset sanity check를 수행한다.

확인하는 내용은 다음이다.

```text
contact ratio
event duration
||tau_ext||
||e_q||
||delta_e_q||
||qdot||
saturation ratio
label과 tau_ext 동기화 여부
```

### train_detectors.py

다음 detector를 학습하거나 선택한다.

```text
Threshold baseline
MLP baseline
GRU detector
```

### evaluate_detectors.py

test split에서 정량 평가를 수행한다.

저장하는 결과는 다음이다.

```text
전체 성능
mode별 성능
참고용 mode 평균 성능
토크 크기 구간별 성능
confusion matrix
metric bar
PR curve
prediction example
```

포스터용으로는 `nc/pc` confusion matrix를 별도 저장한다.

```text
nc = no contact
pc = physical contact

              Predicted
              nc      pc
True nc       TN      FP
True pc       FN      TP
```

`confusion_matrix_summary.json`에는 raw count와 true-label row 기준 percentage가 함께 저장된다. 그림은 `confusion_matrix_threshold.png`, `confusion_matrix_mlp.png`, `confusion_matrix_gru.png`, `confusion_matrix_comparison.png`로 저장된다.

포스터에서는 단일 F1-score만 제시하기보다, nc/pc confusion matrix를 통해 false positive와 false negative의 구조를 함께 보여준다. Threshold 방식은 높은 recall을 보일 수 있지만 no-contact sample을 contact로 오검출하는 false positive가 많을 수 있다. pHRI trigger 관점에서는 이러한 false positive가 불필요한 모드 전환을 유발할 수 있으므로, confusion matrix와 F1-score를 함께 해석한다.

Optuna best MLP와 Optuna best GRU가 서로 다른 ablation 폴더에 저장되어 있을 때는 `evaluate_detectors.py`에 checkpoint 경로를 직접 넘겨 하나의 poster output으로 묶을 수 있다.

```bash
python3 contact_detection/evaluate_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --mlp-model-path contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__mlp_009/models/mlp_detector.pt \
  --gru-model-path contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__gru_014/models/gru_detector.pt \
  --scaler-path contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__mlp_009/models/scaler.pkl \
  --threshold-path contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__mlp_009/models/threshold.json \
  --output-suffix poster_optuna_best
```

이 명령은 기존 결과를 덮지 않고 `*_poster_optuna_best` suffix가 붙은 metrics/figures를 생성한다.

### tune_detectors.py

이미 생성된 같은 dataset을 고정한 상태에서 MLP/GRU 하이퍼파라미터를 여러 trial로 학습한다.

선택 기준은 validation F1이며, test set은 하이퍼파라미터 선택에 사용하지 않는다.

### plot_results.py

이미 저장된 metrics와 example data를 다시 읽어서 figure를 재생성한다.

### run_training_pipeline.py

여러 스크립트를 한 번에 순서대로 실행하는 묶음 실행 스크립트다.

즉, 아래 네 명령을 직접 하나씩 치는 대신:

```text
generate_sim_dataset.py
diagnose_dataset.py
train_detectors.py
evaluate_detectors.py
```

`run_training_pipeline.py` 하나로 같은 순서를 자동 실행한다.

## 9. 실행 방법

먼저 workspace root로 이동한다.

```bash
cd ~/idle_ws
```

### 방법 A: 단계별로 하나씩 실행

여기서 “단계별 실행”은 각 스크립트를 직접 하나씩 실행한다는 뜻이다.

장점은 중간 결과를 확인하면서 멈추기 쉽다는 점이다.

main/original-range experiment:

```bash
python3 contact_detection/generate_sim_dataset.py --config contact_detection/config.yaml --stage randomized_sim
python3 contact_detection/diagnose_dataset.py --config contact_detection/config.yaml --stage randomized_sim
python3 contact_detection/train_detectors.py --config contact_detection/config.yaml --stage randomized_sim
python3 contact_detection/evaluate_detectors.py --config contact_detection/config.yaml --stage randomized_sim
```

기존 GRU-only 논문 결과를 보존한 상태에서 MLP만 추가하려면 `config_legacy_gru_mlp.yaml`을 사용한다. 이 설정은 0.799 F1이 나온 legacy GRU checkpoint를 재학습하지 않고 그대로 재사용하며, 같은 legacy dataset/scaler 조건에서 MLP baseline만 추가 학습한다.

```bash
python3 contact_detection/train_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim

python3 contact_detection/evaluate_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim
```

결과는 기본 `outputs/randomized_sim`을 덮지 않고 아래에 저장된다.

```text
outputs_legacy_gru_mlp/randomized_sim/
```

이 경로의 GRU 결과는 legacy checkpoint 기준이며, `train_log.json`에 `reused_gru_checkpoint: true`로 기록된다.

### 같은 dataset 기준 하이퍼파라미터 튜닝

GRU와 MLP를 공정하게 비교하려면 dataset을 새로 만들면서 비교하면 안 된다. dataset이 바뀌면 성능 차이가 모델 구조 때문인지, 시뮬레이션 데이터가 달라졌기 때문인지 섞인다.

따라서 튜닝은 이미 만들어진 같은 dataset을 고정해서 수행한다.

```text
고정 dataset:
outputs_legacy_gru_mlp/randomized_sim/datasets/sim_train.npz
outputs_legacy_gru_mlp/randomized_sim/datasets/sim_val.npz
outputs_legacy_gru_mlp/randomized_sim/datasets/sim_test.npz
```

`tune_detectors.py`는 `generate_sim_dataset.py`를 호출하지 않는다. 즉, trial마다 데이터를 다시 만들지 않고 같은 `sim_train.npz`와 `sim_val.npz`를 반복해서 읽는다.

튜닝 기준은 validation set이다.

```text
primary metric:
  best_val_f1_selection_label

tie-breaker:
  lower best_val_loss_selection_label

test set:
  hyperparameter 선택에는 사용하지 않음
```

Optuna가 설치되어 있으면 `--sampler optuna`를 사용할 수 있다. 현재 환경에 Optuna가 없으면 기본적으로 같은 validation 기준의 random search로 동작한다. Optuna가 꼭 필요하면 먼저 `optuna`를 설치한 뒤 `--sampler optuna --require-optuna`를 사용한다.

MLP 튜닝 예시는 다음이다. 이 경우 legacy GRU checkpoint는 재사용하고 MLP 하이퍼파라미터만 바꿔가며 학습한다.

```bash
python3 contact_detection/tune_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --models mlp \
  --n-trials-mlp 12 \
  --sampler random \
  --study-name legacy_dataset_mlp_tuning
```

GRU 튜닝 예시는 다음이다. 이 경우 trial config에서 GRU reuse를 끄고 GRU를 다시 학습한다.

```bash
python3 contact_detection/tune_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --models gru \
  --n-trials-gru 12 \
  --sampler random \
  --study-name legacy_dataset_gru_tuning
```

MLP와 GRU를 모두 같은 dataset에서 튜닝하려면 다음처럼 실행한다.

```bash
python3 contact_detection/tune_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --models mlp gru \
  --n-trials-mlp 12 \
  --n-trials-gru 12 \
  --sampler random \
  --study-name legacy_dataset_mlp_gru_tuning
```

Optuna가 설치되어 있는 환경에서는 다음처럼 바꿀 수 있다.

```bash
python3 contact_detection/tune_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --models mlp gru \
  --n-trials-mlp 20 \
  --n-trials-gru 20 \
  --sampler optuna \
  --require-optuna \
  --study-name legacy_dataset_optuna_tuning
```

튜닝 trial 결과는 기본 artifact를 덮지 않고 아래에 저장된다.

```text
outputs_legacy_gru_mlp/randomized_sim/ablations/tune_<study_name>__mlp_000/
outputs_legacy_gru_mlp/randomized_sim/ablations/tune_<study_name>__gru_000/
```

튜닝 요약은 다음 파일에 저장된다.

```text
outputs_legacy_gru_mlp/randomized_sim/metrics/hparam_tuning_<study_name>.json
```

이 요약 파일의 `best_by_model_validation_only`가 validation F1 기준으로 고른 후보이다. test metric은 이 선택에 쓰지 않는다. 선택된 best trial을 test set에서 최종 확인하고 싶을 때만 `--evaluate-best`를 추가한다.

```bash
python3 contact_detection/tune_detectors.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --models mlp gru \
  --n-trials-mlp 12 \
  --n-trials-gru 12 \
  --sampler random \
  --study-name legacy_dataset_mlp_gru_tuning \
  --evaluate-best
```

위 흐름에서 `train_detectors.py`는 validation F1이 가장 높은 epoch의 MLP/GRU checkpoint를 저장하고, `evaluate_detectors.py`는 마지막 epoch가 아니라 그 checkpoint를 불러 test set에서 평가한다.

배포용 final train+val 모델까지 따로 만들고 싶으면 학습 명령에 옵션을 추가한다.

```bash
python3 contact_detection/train_detectors.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --train-final-trainval
```

final train+val 모델을 test set에서 확인할 때는 기본 결과와 섞이지 않도록 별도 variant로 평가한다.

```bash
python3 contact_detection/evaluate_detectors.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --model-variant trainval_final
```

Event latency 설정을 바꿔 평가하려면 다음처럼 실행한다.

```bash
python3 contact_detection/evaluate_detectors.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --event-detection-consecutive-samples 3 \
  --detection-margin-ms 50
```

Label delay ablation 예시는 다음이다. 아래 명령은 기본 `outputs/randomized_sim/models`를 덮지 않고 `outputs/randomized_sim/ablations/label_delay_20ms/` 아래에 저장한다.

```bash
python3 contact_detection/train_detectors.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --label-delay-ms 20

python3 contact_detection/evaluate_detectors.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --label-delay-ms 20
```

Transition-aware exclusion ablation 예시는 다음이다.

```bash
python3 contact_detection/train_detectors.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --transition-exclusion-ms 20

python3 contact_detection/evaluate_detectors.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --transition-exclusion-ms 20
```

validation에서도 transition 주변 window를 제외해 checkpoint/threshold를 고르고 싶을 때만 `--exclude-transition-val`을 추가한다.

여러 ablation을 따로 실행한 뒤 root summary를 다시 만들고 싶으면 다음 명령을 사용한다.

```bash
python3 contact_detection/evaluate_detectors.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --scan-ablations
```

low-torque analysis는 현재 메인 필수 파이프라인이 아니라 약한 외란 민감도 확인용 추가 옵션이다.

```bash
python3 contact_detection/generate_sim_dataset.py --config contact_detection/config.yaml --stage low_torque_analysis
python3 contact_detection/diagnose_dataset.py --config contact_detection/config.yaml --stage low_torque_analysis
python3 contact_detection/train_detectors.py --config contact_detection/config.yaml --stage low_torque_analysis
python3 contact_detection/evaluate_detectors.py --config contact_detection/config.yaml --stage low_torque_analysis
```

### 방법 B: 묶음 실행 스크립트 사용

`run_training_pipeline.py`는 wrapper라고도 부를 수 있다.

여기서 wrapper는 특별한 모델이 아니라, 여러 실행 단계를 한 번에 묶어서 호출해주는 편의용 Python 스크립트라는 뜻이다.

main/original-range experiment:

```bash
python3 contact_detection/run_training_pipeline.py --config contact_detection/config.yaml --stage randomized_sim
```

low-torque analysis:

```bash
python3 contact_detection/run_training_pipeline.py --config contact_detection/config.yaml --stage low_torque_analysis
```

주의: `run_training_pipeline.py`는 기본 비교용 흐름을 실행한다. train+val final 모델은 다음처럼 `train_detectors.py --train-final-trainval`을 별도로 실행한다.

### Recall-prioritized 실험

기존 `config_recall.yaml`은 recall을 더 중요하게 보는 operating point 실험이다.

```bash
python3 contact_detection/run_training_pipeline.py --config contact_detection/config_recall.yaml --stage randomized_sim
```

더 강한 recall-priority 설정은 다음 config를 사용한다.

```bash
python3 contact_detection/run_training_pipeline.py --config contact_detection/config_recall_high.yaml --stage randomized_sim
```

### Figure 재생성

평가 결과가 이미 저장되어 있을 때 figure만 다시 만들고 싶으면 다음을 실행한다.

```bash
python3 contact_detection/plot_results.py --config contact_detection/config.yaml --stage randomized_sim
```

## 10. Real log inference와 ROS online detector

현재 MLP는 offline baseline이다.

실제 로봇 로그와 ROS online detector는 기존처럼 GRU를 사용한다.

실제 로봇 적용은 다음 순서로 진행하는 것을 기본 방향으로 둔다.

```text
1. simulation-trained GRU를 zero-shot으로 실제 로봇에 적용한다.
2. no-contact hold / slow motion 로그에서 false positive baseline을 확인한다.
3. 사람이 EE 또는 gripper를 잡거나 밀어 P(contact)가 올라가는지 확인한다.
4. 필요하면 실물 validation log로 decision threshold만 먼저 보정한다.
5. 그래도 부족하면 실험 프로토콜 기반 weak label로 light fine-tuning을 수행한다.
```

이때 test log는 최종 보고용으로만 사용하고, threshold나 checkpoint를 다시 고르는 데 사용하지 않는다.

### Real log inference

```bash
python3 contact_detection/infer_real_log.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --csv path/to/real_log.csv
```

Optuna best GRU처럼 기본 `outputs/<stage>/models/gru_detector.pt`가 아닌 checkpoint를 실제 로그에 적용할 때는 model/scaler 경로를 직접 지정한다.

```bash
python3 contact_detection/infer_real_log.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --csv path/to/real_log.csv \
  --model-path contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__gru_014/models/gru_detector.pt \
  --scaler-path contact_detection/outputs_legacy_gru_mlp/randomized_sim/ablations/tune_legacy_dataset_optuna_tuning__gru_014/models/scaler.pkl
```

추론 후에는 실제 로그 review 스크립트로 no-contact false alarm, contact interval detection, feature out-of-distribution 여부를 요약한다.

```bash
python3 contact_detection/review_real_contact_log.py \
  --config contact_detection/config_legacy_gru_mlp.yaml \
  --stage randomized_sim \
  --real-csv path/to/real_log.csv \
  --contact-intervals-json '[[5.0, 7.0], [12.0, 14.0]]'
```

출력:

```text
outputs/<stage>/real_inference/real_contact_review.json
outputs/<stage>/figures/real_contact_review.png
```

`real_contact_review.json`은 정량 논문 metric이라기보다 실물 적용 디버그용이다. contact interval은 weak label로만 사용하며 모델 입력에는 들어가지 않는다.

### ROS online detector

```bash
ros2 run phy contact_detector_node --ros-args \
  -p model_path:=/home/eomyunbeen/idle_ws/contact_detection/outputs/randomized_sim/models/gru_detector.pt \
  -p scaler_path:=/home/eomyunbeen/idle_ws/contact_detection/outputs/randomized_sim/models/scaler.pkl
```

ROS node는 monitoring 및 preliminary feasibility check 용도다. 안전 정지나 토크 차단 같은 기능에 직접 연결하면 안 된다.

## 11. 결과 해석 가이드

### Threshold 결과

Threshold가 recall은 높고 precision이 낮으면, 실제 contact는 많이 잡지만 정상 추종 오차도 contact로 오검출한다는 뜻이다.

### MLP 결과

MLP가 Threshold보다 좋으면, learning-based detector가 단순 tracking-error threshold보다 정상 추종 오차와 contact를 더 잘 구분한다는 뜻이다.

### GRU 결과

GRU가 MLP보다 좋으면, 최근 30 sample의 시간적 패턴이 contact/no-contact 구분에 도움이 된다는 뜻이다.

MLP와 GRU가 비슷하면, 현재 시점 42D feature만으로도 상당한 구분이 가능하며 MLP가 더 가벼운 실시간 후보일 수 있다.

### Low-torque 결과

Low-torque 결과는 약한 접촉 검출 가능성과 false positive trade-off를 보기 위한 추가 분석이다.

weak contact recall이 높아져도 false positive가 증가할 수 있으므로, precision / recall / F1-score를 함께 봐야 한다.

## 12. 주의사항

다음 규칙은 반드시 지켜야 한다.

```text
sample 단위 random split 금지
GRU window가 episode 경계를 넘는 것 금지
validation/test에서 scaler fit 금지
tau_ext/contact force/tau_mouse를 model input에 포함 금지
main result와 low-torque analysis 결과를 섞어서 해석 금지
gru_detector.pt와 scaler.pkl 파일명 임의 변경 금지
```

## 13. 자주 보이는 warning

### Matplotlib Axes3D warning

다음 메시지는 보통 치명적인 오류가 아니다.

```text
UserWarning: Unable to import Axes3D
```

이 프로젝트의 figure 저장은 2D plot 위주라서, 3D projection을 못 불러와도 dataset 생성이나 학습은 계속 진행될 수 있다.

### gripper joint lock warning

다음 메시지도 gripper가 URDF에 포함된 경우 예상 가능한 warning이다.

```text
Locked extra URDF movable joints for 6-DOF contact detection: ['finger_l', 'finger_r']
```

현재 연구 범위는 6-DOF arm의 contact/no-contact 검출이므로, config에 없는 gripper movable joint는 neutral position에서 lock하고 6-DOF reduced model로 시뮬레이션한다는 뜻이다.

### 실행 중 멈춘 것처럼 보일 때

`generate_sim_dataset.py --stage randomized_sim`는 기본 설정에서 train/validation/test 총 140 episode를 생성한다.

그래서 warning 이후 한동안 새 출력이 없어도 실패가 아닐 수 있다. 정상적으로 끝나면 다음과 비슷한 출력이 나온다.

```text
[sim_train] generated ...
[sim_val] generated ...
[sim_test] generated ...
Dataset generation complete ...
```

프롬프트가 돌아오기 전에는 아직 실행 중인 상태다.

## 14. 빠른 확인 순서

실험 후에는 다음 순서로 확인하는 것이 좋다.

```text
1. dataset_diagnosis.json
2. threshold.json
3. mlp_train_log.json
4. train_log.json
5. checkpoint_summary.json
6. sim_test_metrics.json
7. validation_original_metrics.json
8. event_latency_metrics.json
9. feature_response_analysis.json
10. mode_split_metrics.json
11. torque_bin_metrics.json
12. ablation_summary.json
13. experiment_config_used.yaml
```

특히 논문이나 포스터에 결과를 적을 때는 `experiment_config_used.yaml`을 먼저 확인해야 한다.
