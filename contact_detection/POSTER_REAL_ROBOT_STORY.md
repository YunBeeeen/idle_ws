# Real Robot Contact Detection Poster Story

이 문서는 포스터에 지금까지 한 일을 어떻게 넣을지 정리한 것이다.
데이터를 앞으로 어떻게 더 쌓을지는 `REAL_DATA_QUALITY_PLAN.md`에 따로 정리한다.

## 1. Core Message

목표는 F/T 센서 없이 로봇 내부 신호만으로 physical contact를 감지하고, 이후 hand guiding 또는 pHRI mode switching의 trigger로 사용할 수 있는 sensorless contact detector를 만드는 것이다.

포스터의 핵심 메시지:

```text
Simulation-trained sensorless contact detection is feasible, but real-robot deployment reveals a sim-to-real gap caused by friction, imperfect gravity compensation, tracking residuals, and motion-induced command dynamics.
```

조금 더 발표용으로 말하면:

```text
We first trained a contact detector in simulation and then tested how it transfers to a real robot. The real experiments showed that contact detection is possible, but robust deployment requires real-robot feature adaptation and better contact data.
```

## 2. Story Flow

포스터 흐름은 아래 순서가 가장 자연스럽다.

```text
1. Simulation contact detection pipeline 구축
2. External force command 기반 label 생성
3. GRU/MLP/threshold baseline 비교
4. Validation F1 기준 checkpoint/threshold 선택
5. Sim-trained GRU를 실제 로봇 로그와 live robot에 적용
6. Real robot에서 false positive와 missed contact 확인
7. Real-specific feature ablation 수행
8. 30D vs 24D 후보 비교
9. 현재 한계와 다음 단계 제시
```

여기서 중요한 점:

- `tau_ext`는 simulation label 생성에만 사용했다.
- `tau_ext`는 모델 입력 feature에 넣지 않았다.
- test/live 결과는 checkpoint, threshold, Optuna 선택에 사용하지 않았다.
- real robot 결과는 최종 배포 모델 성능이 아니라 sim-to-real feasibility 분석이다.

## 3. What We Did

### Simulation

```text
Generated simulated contact episodes with known external force command labels.
Trained binary contact detectors using proprioceptive features.
Compared threshold baseline, MLP, and GRU.
Selected checkpoints and thresholds using validation F1-score.
```

포스터 문장:

```text
A GRU-based binary contact detector was trained in simulation using external force command labels. The external torque signal was used only for labeling and was never included in the model input.
```

### Real Robot Transfer

실제 로봇에서는 다음 데이터를 기록했다.

```text
q
qdot
q_des
tau_cmd
tau_meas
tau_ff
kp/kd
manual contact intervals
```

실제 적용 중 관찰한 문제:

```text
No-contact motion sometimes produced contact-like probability peaks.
Some contact directions were detected better than others.
The detector could be stable in home/no-contact but miss or delay contact under slow sine motion.
```

포스터 문장:

```text
When the simulation-trained detector was applied to the real robot, no-contact motion sometimes produced contact-like responses. This suggests a sim-to-real gap caused by friction, imperfect gravity compensation, steady-state tracking residuals, and command jitter.
```

### Real Feature Ablation

기존 실물 후보 feature:

```text
30D: q, qdot, tau_cmd, delta_qdot, delta_tau_cmd
```

더 보수적인 실물 후보 feature:

```text
24D: q, qdot, tau_cmd, delta_tau_cmd
```

해석:

```text
30D retains delta_qdot and can be more sensitive.
24D removes delta_qdot to reduce motion-induced false positives.
```

포스터 문장:

```text
To reduce false positives from motion-induced derivative features, we evaluated real-specific feature ablations. Removing delta_qdot reduced no-contact false alarms, but introduced a sensitivity-recall trade-off.
```

## 4. What To Claim

강하게 말해도 되는 것:

- Sensorless contact detection pipeline을 만들었다.
- Simulation에서는 label이 명확한 contact dataset을 생성했다.
- `tau_ext`를 모델 입력으로 쓰지 않는 규칙을 지켰다.
- 실제 로봇 로그 수집과 online detector 적용까지 진행했다.
- Real robot 적용에서 sim-to-real gap을 확인했다.
- 24D feature는 real robot trigger 후보로 볼 수 있다.
- Real robot에서 no-contact false positive와 contact miss 사이 trade-off가 관찰되었다.

조심해야 하는 표현:

- "최종 배포 모델"이라고 말하지 않는다.
- "실제 로봇 contact detection 완성"이라고 말하지 않는다.
- "24D가 최적"이라고 말하지 않는다.
- "Optuna best = true best"라고 말하지 않는다.

추천 표현:

```text
24D was selected as a preliminary real-robot trigger candidate after validation-only tuning.
```

또는:

```text
The real-robot experiments revealed that reducing derivative-sensitive features can improve no-contact stability, but more balanced contact data is required for robust direction-invariant detection.
```

## 5. Figure Plan

### Figure A: Pipeline

```text
Simulation generation
  -> external force command label
  -> GRU/MLP/threshold training
  -> validation F1 checkpoint selection
  -> real robot log inference
  -> real feature ablation
```

그림에서 강조:

```text
tau_ext: label only
input: proprioceptive + command features
selection: validation only
```

### Figure B: Simulation Result

nc/pc confusion matrix로 보여준다.

```text
nc = no contact
pc = physical contact
```

가능한 구성:

```text
Threshold baseline vs GRU
또는
Threshold baseline vs MLP vs GRU
```

### Figure C: Real Transfer Issue

실제 robot no-contact/slow sine에서 P(contact)가 튀는 probability plot을 보여준다.

caption 예시:

```text
Direct real-robot application showed contact-like probability peaks during no-contact motion, indicating sim-to-real mismatch.
```

### Figure D: 30D vs 24D Ablation

보여줄 것:

```text
probability overlay
nc/pc confusion matrix
F1, Recall, FPR
```

caption 예시:

```text
The 24D candidate reduced no-contact false positives by removing delta_qdot, while 30D retained higher sensitivity.
```

### Figure E: Limitation / Next Step

```text
Current limitation:
  manual interval labels
  limited contact directions
  uncalibrated force magnitude
  no observer-based external torque estimate yet

Next step:
  balanced real contact dataset
  residual/observer torque features
  direction-aware evaluation
```

## 6. Poster Text Draft

### Motivation

```text
For pHRI and hand guiding, the robot must detect physical contact without relying on external F/T sensors. We investigate a sensorless contact detector using proprioceptive signals and commanded motor torques.
```

### Method

```text
We first trained a GRU contact detector in simulation using external force command labels. The external torque signal was used only for labeling and was never included in the model input.
```

### Real Robot Study

```text
The simulation-trained detector was then applied to real robot logs and live robot motion. Real no-contact motion revealed false positives caused by tracking residuals, gravity compensation mismatch, friction, and command dynamics.
```

### Ablation

```text
We evaluated real-specific feature ablations to reduce motion-induced false positives. A 24D candidate that removes delta_qdot showed more stable trigger behavior, while contact recall remained sensitive to direction and label timing.
```

### Conclusion

```text
The results support the feasibility of sensorless contact detection as a pHRI trigger, while showing that robust real-world deployment requires balanced real contact data and improved residual/observer-based features.
```

## 7. Final Poster Position

이 포스터는 "완성된 실물 contact detector"가 아니라 아래를 보여주는 포스터로 잡는 것이 좋다.

```text
Simulation-to-real sensorless contact detection pipeline
+
real robot feasibility check
+
feature ablation and limitation analysis
```

즉 핵심 기여는:

```text
We built the pipeline, transferred it to a real robot, identified the sim-to-real failure modes, and proposed a real-specific feature/data direction for robust pHRI contact triggering.
```

