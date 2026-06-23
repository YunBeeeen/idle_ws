# 포스터 Figure 구성 가이드

이 문서는 포스터에 어떤 그림을 어디에 넣을지 정리한 것이다.
현재 포스터는 **시뮬레이션 결과를 메인 성과**로 두고, 실제 로봇 결과는 **초기 적용 실험과 sim-to-real gap 분석**으로 작게 배치하는 방향이 가장 안전하다.

## 1. 포스터 메시지

추천 메시지:

```text
F/T 센서 없이 로봇 내부 신호만으로 contact를 검출하는 sensorless contact detection pipeline을 구축하였다.
시뮬레이션에서는 external force command 기반 label로 GRU detector가 안정적인 성능을 보였고,
실제 로봇 적용에서는 friction, gravity compensation error, tracking residual, command jitter로 인한 sim-to-real gap이 관찰되었다.
```

짧게 말하면:

```text
Simulation에서 잘 동작하는 sensorless contact detector를 만들었고,
실제 로봇에 적용하면서 어떤 feature와 데이터가 추가로 필요한지 확인했다.
```

## 2. 권장 포스터 배치

### 왼쪽 위: Background and Objective

내용:

```text
F/T 센서 없이 pHRI/hand guiding trigger로 사용할 수 있는 contact detector가 필요하다.
본 연구는 simulation에서 detector를 학습하고, 실제 로봇 적용을 통해 sim-to-real gap을 분석한다.
```

강조할 표현:

```text
완성된 실물 배포 모델이 아니라, sensorless contact detection의 feasibility와 sim-to-real gap 분석이다.
```

### 왼쪽 중간: Detection Pipeline

사용 그림:

```text
contact_detection/poster_assets/poster_pipeline_kr.png
```

캡션 예시:

```text
Fig. 1. Sensorless contact detection pipeline. τ_ext는 simulation label 생성에만 사용하고, 모델 입력에는 포함하지 않는다.
```

### 오른쪽 위: Simulation Results

사용 그림:

```text
contact_detection/poster_assets/poster_simulation_summary_kr.png
```

넣을 문장:

```text
시뮬레이션에서 GRU detector는 nc/pc sample-level classification과 event detection 모두에서 안정적인 결과를 보였다.
Threshold는 validation F1 기준으로 선택했으며, test set은 최종 보고용으로만 사용하였다.
```

핵심 숫자:

```text
Precision = 0.836
Recall = 0.765
F1-score = 0.799
Event detection = 41 / 48
Mean latency = 5.7 ms
```

### 오른쪽 중간: Real-Robot Preliminary Study

사용 그림:

```text
contact_detection/poster_assets/poster_real_ablation_summary_kr.png
```

넣을 문장:

```text
실제 로봇에서는 no-contact hard negative와 manual EE contact interval을 수집하였다.
직접 적용 결과, no-contact motion에서도 contact-like feature pattern이 발생하여 sim-to-real gap이 확인되었다.
```

30D/24D 해석:

```text
30D와 24D real-specific feature set을 validation-only Optuna로 비교하였다.
현재 데이터에서는 30D가 validation-best였으며, 24D는 Δqdot 제거의 보수적 ablation으로 해석한다.
```

핵심 숫자:

```text
30D: F1 = 0.687, Recall = 0.596, FPR = 0.024
24D: F1 = 0.657, Recall = 0.594, FPR = 0.036
```

주의 문장:

```text
Real-robot 결과는 데이터 수가 제한된 preliminary result이며, live/test episode는 모델 선택에 사용하지 않았다.
```

### 오른쪽 아래: Future Work

넣을 항목:

```text
- balanced real contact dataset
- contact direction / magnitude label
- residual torque compensation
- observer-based external torque estimation
- pHRI mode switching trigger integration
```

한국어 문장:

```text
향후에는 접촉 방향과 세기가 균형 잡힌 real contact dataset을 구축하고,
residual/observer 기반 feature를 추가하여 실제 로봇에서 안정적인 pHRI trigger로 확장할 계획이다.
```

## 3. 기존 그림 중 쓸 만한 후보

시뮬레이션 prediction 예시:

```text
contact_detection/outputs/randomized_sim/figures/sim_prediction_examples.png
```

시뮬레이션 metric bar:

```text
contact_detection/outputs/randomized_sim/figures/sim_metric_bar.png
```

시뮬레이션 GRU confusion matrix:

```text
contact_detection/outputs/randomized_sim/figures/confusion_matrix_gru.png
```

실제 로그에서 30D/24D probability overlay:

```text
contact_detection/outputs_real/model_compare_20260619/j23_006/figures/real_model_probability_overlay.png
```

실제 로그에서 30D/24D confusion comparison:

```text
contact_detection/outputs_real/model_compare_20260619/j23_006/figures/real_model_confusion_comparison.png
```

## 4. 지금 포스터에서 피해야 할 표현

피할 표현:

```text
실제 로봇 contact detection 완성
최종 배포 모델
30D가 최적 모델
Optuna best가 진짜 최적
```

추천 표현:

```text
current validation-best real-robot candidate
preliminary real-robot study
sim-to-real feasibility analysis
real-specific feature ablation
```

한국어 추천 표현:

```text
현재 데이터 기준 validation-best 후보
초기 실제 로봇 적용 실험
sim-to-real gap 분석
실물 로봇용 feature ablation
```

## 5. 한 문단 요약

포스터 결론에 넣기 좋은 문단:

```text
본 연구는 시뮬레이션에서 contact/no-contact label이 명확한 sensorless contact detector를 학습하고,
이를 실제 QDD 로봇에 적용하여 sim-to-real gap을 분석하였다.
시뮬레이션에서는 GRU detector가 안정적인 nc/pc 검출 성능을 보였으며,
실제 로봇에서는 tracking residual, friction, command jitter에 의해 false positive와 missed contact가 발생하였다.
제한된 real dataset에서 30D feature set이 validation-best 후보로 나타났지만,
강건한 pHRI trigger로 확장하기 위해서는 균형 잡힌 real contact data와 residual/observer 기반 feature가 필요하다.
```

