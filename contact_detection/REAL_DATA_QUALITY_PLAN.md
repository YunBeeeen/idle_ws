# Real Data Quality Plan

이 문서는 앞으로 실제 로봇 contact/no-contact 데이터를 어떻게 더 좋게 쌓을지 정리한 것이다.
포스터에 지금까지 한 일을 어떻게 말할지는 `POSTER_REAL_ROBOT_STORY.md`에 따로 정리한다.

## 1. Current Problem

현재 real detector가 과적합처럼 보이는 이유는 모델만의 문제가 아니라 데이터 품질과 다양성의 문제일 가능성이 크다.

현재 데이터의 약점:

- contact episode 수가 아직 부족하다.
- 손으로 미는 방향과 위치가 비슷한 episode가 많다.
- manual contact interval의 시작/끝이 실제 힘이 들어간 순간과 정확히 맞지 않는다.
- slow sine no-contact와 slow sine contact가 motion 자체는 비슷하고, 접촉 패턴만 약하게 다르다.
- contact 힘의 크기가 episode마다 일정하지 않다.
- 같은 조건의 episode들이 서로 너무 비슷해서 모델이 일반적인 contact보다 episode-specific pattern을 외울 수 있다.

따라서 Optuna에서 대부분 train loss만 내려가고 validation이 흔들리면, 이것은 hyperparameter tuning 실패라기보다 real contact dataset이 아직 충분히 일반적이지 않다는 신호로 해석한다.

## 2. Data Collection Goal

목표는 단순히 많은 데이터를 쌓는 것이 아니라, 아래 조건을 균형 있게 포함하는 것이다.

```text
motion condition
contact direction
contact strength
contact location
label confidence
```

좋은 dataset은 모델에게 다음을 알려줘야 한다.

```text
정상 no-contact tracking error는 contact가 아니다.
정상 slow sine 방향 전환 마찰은 contact가 아니다.
약한 접촉도 contact일 수 있다.
운동 방향 반대 접촉만 contact가 아니다.
운동 방향 같은 방향, 옆 방향, 위/아래 방향 접촉도 contact다.
```

## 3. Label Protocol

기본 label:

```text
nc = no contact
pc = physical contact
```

추천 episode 길이:

```text
duration = 15 s
0-4 s: no contact
5-8 s: stable contact
9-15 s: no contact
```

명령 예시:

```bash
--contact-intervals-json '[[5.0,8.0]]'
```

이전처럼 `[[5.0,13.0]]`를 너무 넓게 잡으면 다음 구간이 contact label에 섞일 수 있다.

```text
손이 닿기 전 접근 구간
힘이 아직 약한 ramp-in 구간
손을 떼는 ramp-out 구간
손 뗀 뒤 로봇이 다시 움직이는 구간
```

따라서 다음부터는 contact를 조금 짧고 확실하게 유지하는 편이 좋다.

주의:

- contact 시작 전에 미리 손을 대지 않는다.
- 5초 근처에서 천천히 접촉을 시작하되, 실제 힘이 들어가는 구간을 5-8초에 맞춘다.
- 8초 이후에는 천천히 떼고, 9초 이후는 완전히 no-contact로 만든다.
- 애매하게 실패한 episode는 지우기보다 manifest에 `label_uncertain`으로 표시한다.

## 4. No-Contact Dataset

no-contact는 hard negative 역할을 한다.
즉, 모델이 contact로 오해하기 쉬운 정상 로봇 움직임을 많이 보여줘야 한다.

각 조건당 최소 5 episode:

```text
home hold: 5
gravity compensation hold: 5
slow sine j2: 5
slow sine j3: 5
slow sine j2+j3: 5
PTP move and hold: 5
```

포함해야 할 no-contact 현상:

- steady-state tracking error
- gravity compensation mismatch
- slow sine 방향 전환 마찰
- qdot noise
- tau_cmd fluctuation
- PTP 도착 후 잔류 오차

목표:

```text
모델이 "정상적으로 못 따라가는 것"과 "사람이 접촉한 것"을 구분하도록 만든다.
```

## 5. Contact Dataset

contact는 motion condition, direction, strength를 나눠서 모은다.

조건:

```text
home hold contact
gravity compensation hold contact
slow sine j2 contact
slow sine j3 contact
slow sine j2+j3 contact
PTP hold contact
```

각 조건당 최소 5-10 episode를 목표로 한다.

방향:

```text
against motion direction
with motion direction
side direction
up/down direction
```

세기:

```text
weak
medium
strong but safe
```

위치:

```text
EE contact 우선
가능하면 link/body contact는 별도 그룹으로 기록
```

지금 hand guiding trigger 관점에서는 EE contact가 우선이다.
다만 pHRI accident detection까지 확장하려면 link/body contact도 나중에 필요하다.

## 6. Recommended Round-2 Amount

현실적인 1차 개선 목표:

```text
No-contact:
  6 conditions x 5 episodes = 30 episodes

Contact:
  6 conditions x 6 episodes = 36 episodes

Total:
  about 60-70 new episodes
```

더 좋은 목표:

```text
No-contact: 50 episodes
Contact: 50 episodes
Total: 100 episodes
```

단, 같은 조건만 반복해서 100개를 만드는 것보다, 조건을 균형 있게 나누는 것이 훨씬 중요하다.

## 7. Episode Naming

추천 naming:

```text
no_contact_home_hold_001.csv
no_contact_gc_hold_001.csv
no_contact_slow_sine_j2_001.csv
no_contact_slow_sine_j3_001.csv
no_contact_slow_sine_j23_001.csv
no_contact_ptp_hold_001.csv

contact_home_hold_against_001.csv
contact_home_hold_side_001.csv
contact_slow_sine_j2_against_001.csv
contact_slow_sine_j2_with_001.csv
contact_slow_sine_j3_side_001.csv
contact_slow_sine_j23_down_001.csv
contact_ptp_hold_side_001.csv
```

## 8. Manifest Fields

manifest는 현재보다 조금 더 자세히 쓰는 것이 좋다.

추천 header:

```csv
episode_id,type,motion_condition,contact_direction,contact_strength,contact_location,csv_path,label,contact_intervals,label_confidence,notes
```

예시:

```csv
contact_slow_sine_j23_against_003,contact,slow_sine_j23,against_motion,medium,ee,contact_detection/real_logs/contact/20260619/contact_slow_sine_j23_against_003.csv,pc,"[[5.0,8.0]]",high,"clean push, stable interval"
```

label confidence 기준:

```text
high: 접촉 시작/끝이 명확하고 interval이 잘 맞음
medium: 접촉은 맞지만 시작/끝이 약간 애매함
low: 실수했거나 timing이 많이 어긋남
```

low confidence episode는 학습에서 제외하거나 별도 ablation으로 둔다.

## 9. Collection Checklist

각 episode 기록 전에 확인:

- CAN state가 모든 motor에서 들어오는가
- command topic이 `/motor_cmd_array_applied`로 기록되는가
- 사람이 접촉하지 않는 구간이 확실한가
- contact interval 시간을 소리내서 맞출 수 있는가
- 로봇이 안전한 자세인가

각 episode 기록 후 확인:

- CSV row 수가 duration x sample_hz와 비슷한가
- contact_label positive row가 예상 개수인가
- q/qdot/tau_cmd가 NaN 없이 기록됐는가
- manifest에 방향/세기/위치/신뢰도를 적었는가

## 10. Training Policy After Round 2

계속 유지할 원칙:

- episode-wise split
- train-only scaler
- validation-only checkpoint selection
- validation-only threshold selection
- Optuna도 validation metric만 사용
- live/test/review CSV는 모델 선택에 사용하지 않음

추천 ranking:

```text
1. validation F1
2. lower validation FPR
3. higher validation recall
4. lower validation loss
```

pHRI trigger 관점 해석:

```text
False positive -> 불필요한 mode switch
False negative -> 접촉을 놓침
```

따라서 F1 하나만 보지 말고 nc/pc confusion matrix와 FPR/Recall trade-off를 같이 본다.

## 11. What Good Data Should Improve

좋은 데이터가 쌓이면 기대되는 변화:

```text
train_loss와 val_loss gap 감소
validation F1 안정화
validation FPR 감소
contact direction별 recall 편차 감소
live no-contact에서 P(contact) 안정화
slow sine contact에서 missed detection 감소
```

만약 데이터가 늘어도 계속 과적합이면 다음을 의심한다.

```text
feature 자체가 부족함
tau_meas/tau_cmd residual 정리가 필요함
observer 기반 external torque estimate가 필요함
manual interval label이 너무 noisy함
```

## 12. Next Technical Step

데이터 개선 후의 기술적 개선 방향:

```text
1. residual torque feature 정리
2. tau_meas - tau_cmd offset correction 강화
3. observer-based external torque estimate 추가
4. contact direction / magnitude label 추가
5. event-level latency metric 추가
```

observer는 지금 당장 필수는 아니지만, 논문 흐름과 더 가까워지려면 중요한 다음 단계다.

