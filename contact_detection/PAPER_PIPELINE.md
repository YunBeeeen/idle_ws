# Sensorless Contact Detection 논문용 코드 흐름 정리

이 문서는 현재 repo의 contact detection 코드가 논문 흐름에서 어디에 대응되는지
정리한 것이다. 연구 범위는 **6-DOF QDD 로봇팔의 contact/no-contact binary
classification**이다.

중요한 연구 정의:

- 외력 크기 추정(force estimation)이 아니다.
- 접촉 위치 추정(contact localization)이 아니다.
- `tau_ext`, external force, MuJoCo contact force는 label/diagnosis 전용이다.
- 모델 입력은 실제 로봇에서도 얻을 수 있는 값만 사용한다.

모델 입력 feature:

```text
x_t = [q_t, qdot_t, e_q_t, tau_cmd_t, delta_e_q_t, delta_qdot_t, delta_tau_cmd_t]
e_q_t = q_des_t - q_t
tau_cmd_t = commanded torque / control input
```

현재 `randomized_sim` 모델은 delta feature를 사용하므로 총 feature dimension은
`6 * 7 = 42`이다.

## 1. 데이터 생성

관련 코드:

```text
generate_sim_dataset.py
utils.py
config.yaml
```

논문에서 대응되는 부분:

```text
F/T sensor가 없는 실제 로봇에서는 외력 ground truth를 얻기 어렵다.
따라서 URDF 기반 명목 동역학 시뮬레이션에서 외란 토크를 주입해 label을 만든다.
```

동역학 개념:

```text
M(q) qddot + C(q, qdot) qdot + g(q) = tau_cmd + tau_ext
```

제어 입력:

```text
tau_cmd_raw = Kp*(q_des - q) + Kd*(qdot_des - qdot) + g(q)
tau_cmd = clip(tau_cmd_raw, -torque_limit, torque_limit)
```

label:

```text
label = 1 if ||tau_ext|| > eps else 0
```

저장되는 핵심 값:

```text
q, qdot, q_des, qdot_des, tau_cmd_raw, tau_cmd, tau_ext, label, episode_id
```

주의:

- `tau_ext`는 npz에 저장되지만 모델 입력에는 들어가지 않는다.
- train/val/test는 episode 단위로 생성된다.
- `episode_id`가 train/val/test 사이에 겹치면 코드가 에러를 낸다.

## 2. Dataset diagnosis

관련 코드:

```text
diagnose_dataset.py
```

논문에서 대응되는 부분:

```text
학습 전에 label과 feature 사이에 관측 가능한 신호가 있는지 확인한다.
```

확인하는 것:

- contact ratio
- episode별 event 수
- 평균 contact duration
- `||tau_ext||`
- `||e_q||`
- `||delta_e_q||`
- `||qdot||`
- saturation ratio
- label과 tau_ext 동기화

출력:

```text
outputs/<stage>/metrics/dataset_diagnosis.json
outputs/<stage>/figures/dataset_diagnosis_example.png
```

이 단계에서 contact/no-contact의 `||e_q||`, `||delta_e_q||` 분포가 거의 같으면
GRU가 배울 신호가 약하다는 뜻이다.

## 3. Sliding window dataset

관련 코드:

```text
contact_dataset.py
utils.py::build_input_features
```

논문에서 대응되는 부분:

```text
관절 상태 시계열을 이용해 GRU 입력 window를 만든다.
```

window:

```text
X_t = [x_{t-L+1}, ..., x_t]
y_t = label at current timestep
```

현재 기본값:

```text
window_length = 30
dt = 0.002 s
window time = 0.06 s
```

중요한 validity rule:

- window는 episode 경계를 넘지 않는다.
- delta feature도 episode 경계를 넘지 않는다.
- scaler는 train set feature dimension별 mean/std만 사용한다.

## 4. Threshold baseline

관련 코드:

```text
train_detectors.py
utils.py::threshold_score_from_data
evaluate_detectors.py
```

논문에서 대응되는 부분:

```text
GRU가 정말 의미 있는지 보기 위해 단순 threshold baseline과 비교한다.
```

지원 metric:

```text
error_norm: ||e_q||
delta_error_norm: ||delta_e_q||
combined: alpha*||e_q|| + beta*||delta_e_q||
```

validation set에서 F1이 최대가 되는 gamma를 찾고 저장한다.

출력:

```text
outputs/<stage>/models/threshold.json
```

## 5. GRU 학습

관련 코드:

```text
models.py
train_detectors.py
```

논문에서 대응되는 부분:

```text
관절 상태 시계열로 P(contact)를 출력하는 GRU detector를 학습한다.
```

학습 설정:

```text
loss = BCEWithLogitsLoss(pos_weight=N_negative/N_positive)
optimizer = Adam
early stopping = validation F1 기준
```

출력:

```text
outputs/<stage>/models/gru_detector.pt
outputs/<stage>/models/scaler.pkl
outputs/<stage>/metrics/train_log.json
outputs/<stage>/figures/training_curve.png
```

checkpoint에는 다음 metadata가 들어간다.

```text
model_type = binary
feature_names
input_dim
window_length
use_delta_features
decision_threshold
```

이 metadata는 ROS2 online node가 feature mismatch를 막는 데 사용한다.

## 6. 시뮬레이션 정량 평가

관련 코드:

```text
evaluate_detectors.py
plot_results.py
```

논문에서 대응되는 부분:

```text
시뮬레이션 test set은 정답 label이 있으므로 Precision, Recall, F1-score로 평가한다.
```

출력:

```text
outputs/<stage>/metrics/sim_test_metrics.json
outputs/<stage>/figures/sim_metric_bar.png
outputs/<stage>/figures/confusion_matrix_gru.png
outputs/<stage>/figures/sim_prediction_example.png
outputs/<stage>/figures/gru_threshold_tradeoff.png
```

현재 `randomized_sim` test 결과:

```text
GRU       Precision 0.836 / Recall 0.765 / F1 0.799
Threshold Precision 0.458 / Recall 0.859 / F1 0.597
```

해석:

- GRU가 threshold보다 F1/Precision 측면에서 좋다.
- FN이 남아 있으므로 "모든 접촉을 완벽히 맞힌다"는 과장된 결과가 아니다.

## 7. MuJoCo demo

관련 코드:

```text
mujoco_contact_demo.py
src/sim/robot.xml
```

논문/발표에서 대응되는 부분:

```text
실제 로봇에 적용하기 전 MuJoCo에서 외란을 주었을 때 P(contact)가 올라가는지 확인한다.
```

실행:

```bash
python3 contact_detection/mujoco_contact_demo.py \
  --config contact_detection/config.yaml \
  --stage randomized_sim \
  --mouse \
  --duration 60
```

주의:

- mouse force는 viewer 시연용이다.
- 모델 입력으로 force를 넣는 것이 아니다.
- 바닥 collision은 `src/sim/robot.xml`에 추가되어 있다.

## 8. 실제 로봇 CSV inference

관련 코드:

```text
infer_real_log.py
record_real_log.py
```

논문에서 대응되는 부분:

```text
학습된 모델을 실제 로봇 로그에 적용해 의도적 접촉 구간에서 P(contact)가 증가하는지 본다.
```

실제 로봇에는 보통 contact ground truth가 없으므로 기본 결과는 정량 accuracy가 아니다.
`contact_marker`가 있고 사용자가 명시적으로 marker metrics를 켤 때만 real F1을 계산한다.

출력:

```text
outputs/<stage>/real_inference/real_contact_probability.csv
outputs/<stage>/figures/real_contact_probability.png
```

## 9. ROS2 online detector

관련 코드:

```text
src/phy/phy/contact_detector_node.py
src/idle_launch/launch/contact_detector.launch.py
```

논문/실험에서 대응되는 부분:

```text
학습된 GRU를 ROS2 node로 띄워 실시간 P(contact)를 publish한다.
```

Subscribe:

```text
/motor_state_array
/motor_cmd_array
```

Publish:

```text
/contact_probability
/contact_state
/contact_detector_ready
```

중요:

- monitoring/preliminary feasibility check용이다.
- certified safety function이 아니다.
- `state.tau`나 외력 추정값을 쓰지 않는다.
- motor_id 순서로 joint order를 강제한다.

## 실제 실행했던 권장 학습 순서

1. `easy_hold`

```bash
python3 contact_detection/generate_sim_dataset.py --config contact_detection/config.yaml --stage easy_hold
python3 contact_detection/diagnose_dataset.py --config contact_detection/config.yaml --stage easy_hold
python3 contact_detection/train_detectors.py --config contact_detection/config.yaml --stage easy_hold
python3 contact_detection/evaluate_detectors.py --config contact_detection/config.yaml --stage easy_hold
```

2. `sine_no_randomization`

```bash
python3 contact_detection/generate_sim_dataset.py --config contact_detection/config.yaml --stage sine_no_randomization
python3 contact_detection/diagnose_dataset.py --config contact_detection/config.yaml --stage sine_no_randomization
python3 contact_detection/train_detectors.py --config contact_detection/config.yaml --stage sine_no_randomization
python3 contact_detection/evaluate_detectors.py --config contact_detection/config.yaml --stage sine_no_randomization
```

3. `randomized_sim`

```bash
python3 contact_detection/generate_sim_dataset.py --config contact_detection/config.yaml --stage randomized_sim
python3 contact_detection/diagnose_dataset.py --config contact_detection/config.yaml --stage randomized_sim
python3 contact_detection/train_detectors.py --config contact_detection/config.yaml --stage randomized_sim
python3 contact_detection/evaluate_detectors.py --config contact_detection/config.yaml --stage randomized_sim
```

논문 본문 결과와 ROS2 node는 `randomized_sim` 모델을 기준으로 쓰는 것이 가장 깔끔하다.

## 평가자가 의심할 수 있는 부분에 대한 답

1. tau_ext leakage?

`feature_names`에 `tau_ext`가 없고, feature builder가 tau_ext를 인자로 받지 않는다.

2. sample-level split leakage?

데이터 생성 시 train/val/test episode_id가 서로 겹치면 에러를 낸다.

3. window crossing?

`build_window_end_indices()`는 episode slice 내부에서만 window end index를 만든다.

4. threshold baseline?

`threshold.json`과 `sim_test_metrics.json`에 함께 저장된다.

5. 너무 완벽한 결과?

현재 randomized_sim test에는 FN이 존재한다. 즉 어려운/작은 외란 case를 놓치는
구간이 있어 결과가 과도하게 완벽하지 않다.
