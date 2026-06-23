# Wafer Map Failure Pattern ML Project Plan

## 1. 프로젝트 방향 정정

본 프로젝트는 프로젝트 가이드라인의 지침을 우선하여 **CNN 기반 딥러닝 분류기가 아니라 XGBoost, Random Forest, SVM, Logistic Regression 등 전통 머신러닝 모델**을 중심으로 설계한다. 선행 연구 중 CNN 계열 논문은 직접 구현 대상이 아니라, wafer map 문제에서 중요한 **기하학적 불변성, 데이터 불균형, 특징 표현 방식**을 도출하기 위한 참고 연구로 활용한다.

따라서 본 프로젝트의 핵심 차별점은 다음과 같다.

1. wafer map 이미지를 그대로 CNN에 넣지 않고, 결함 위치와 형태를 설명하는 수치 특징으로 변환한다.
2. XGBoost와 Random Forest를 중심으로 분류 모델을 만들고, feature importance와 SHAP으로 판단 근거를 설명한다.
3. Radon transform, radial profile, connected component 등 선행 연구에서 강조된 구조 정보를 머신러닝용 특징으로 재해석한다.
4. 단순 정확도 경쟁이 아니라, 불균형 데이터와 회전/반전 변화에서 안정적인 성능을 검증한다.
5. 분류 결과와 함께 유사 wafer map 검색을 제공하여 실패 원인 추론에 활용할 수 있는 분석형 프로젝트로 만든다.

## 2. 목표 정의

### 2.1 문제 정의

wafer bin map에서 defect die의 공간적 분포를 분석하여 failure pattern class를 예측한다. 입력 wafer map은 이미지 분류 문제가 아니라 **공간 패턴 데이터**로 간주하고, 각 wafer map에서 결함의 밀도, 위치, 거리 분포, 군집성, 선형성, 대칭성, radial 구조를 추출한다.

### 2.2 최종 목표

입력 wafer map에 대해 다음 결과를 제공하는 머신러닝 기반 failure pattern analysis pipeline을 구축한다.

- failure pattern 예측 class
- 예측 확률 또는 confidence score
- 예측에 크게 기여한 상위 특징
- 같은 class 또는 유사 구조를 가진 wafer map top-k
- 회전/반전 변형에 대한 robustness 평가 결과

### 2.3 연구 질문

1. handcrafted feature만으로 wafer map failure pattern을 얼마나 잘 분류할 수 있는가?
2. XGBoost와 Random Forest 중 어떤 모델이 불균형 wafer map 데이터에서 더 안정적인가?
3. radial/Radon/connected-component 특징을 추가하면 edge, ring, scratch, donut 계열 패턴 구분이 개선되는가?
4. feature importance와 SHAP을 통해 각 failure pattern의 판단 근거를 설명할 수 있는가?
5. 유사도 검색 결과가 예측 class의 신뢰도를 보조하는 근거로 활용될 수 있는가?

## 3. 선행 연구 활용 전략

| 선행 연구 | 직접 사용할 내용 | 본 프로젝트에서의 차별화 적용 |
| --- | --- | --- |
| Boosted Stacking Ensemble Machine Learning Method for Wafer Map Pattern Classification | handcrafted feature와 ensemble ML의 유효성 | XGBoost/Random Forest 중심 baseline 및 stacking 후보 설계 |
| Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-scale Datasets | 분류와 유사도 ranking 결합 | 예측 class와 함께 top-k similar wafer map 제공 |
| Wafer map failure pattern classification using geometric transformation-invariant convolutional neural network | 회전/반전 불변성, Radon 기반 표현 | CNN은 제외하고 Radon projection을 ML feature로 변환 |
| Iterative Cluster Harvesting for Wafer Map Defect Patterns | 라벨 부족 상황에서 clustering을 통한 패턴 탐색 | UMAP/PCA + clustering으로 feature space 검증 및 라벨 검토 보조 |

## 4. 데이터 계획

### 4.1 데이터셋 후보

- 1순위: WM-811K wafer map dataset
- 초기 실험 class: `center`, `donut`, `edge-loc`, `edge-ring`, `loc`, `near-full`, `random`, `scratch` 중 가이드라인과 데이터 가용성에 맞춰 결정
- 프로젝트 초반에는 라벨 수가 극단적으로 적거나 정의가 모호한 class를 제외한 6~8 class 실험을 우선한다.

### 4.2 전처리 원칙

1. wafer map의 die 상태를 숫자 matrix로 변환한다.
2. valid die 영역과 defect die 영역을 분리한다.
3. wafer 크기가 서로 다를 경우 feature 기반 방식에서는 원본 좌표를 정규화하고, 필요할 때만 64x64 또는 96x96 grid로 재표현한다.
4. class imbalance를 기록하고 train/validation/test split이 class distribution을 보존하도록 stratified split을 사용한다.
5. 데이터 누수를 막기 위해 feature scaling, PCA, sampling은 train split에 fit한 뒤 validation/test에 transform한다.

## 5. 특징 설계

### 5.1 기본 통계 특징

- wafer 내 전체 die 수
- defect die 수
- defect ratio
- valid die ratio
- defect x/y 좌표 평균과 표준편차
- defect bounding box width, height, area ratio

### 5.2 위치 기반 특징

- wafer 중심으로부터 defect까지의 평균 거리
- 중심 거리의 표준편차, 왜도, 첨도
- center region defect ratio
- edge region defect ratio
- quadrant별 defect density
- radial bin별 defect histogram
- angular bin별 defect histogram

### 5.3 패턴별 해석 특징

- ringness score: 특정 반지름 구간에 defect가 집중되는 정도
- donut score: 중심부 결함이 낮고 중간 반지름 영역 결함이 높은 정도
- edge-localization score: wafer 외곽 특정 각도 구간에 defect가 몰리는 정도
- scratch linearity score: PCA 첫 번째 주성분 설명력 또는 Hough/Radon peak strength
- random dispersion score: defect 분포의 균일성, nearest-neighbor distance variance
- local cluster score: connected component 개수와 largest component ratio

### 5.4 기하학적 불변 특징

- radial density profile: 회전에 덜 민감한 1차원 반지름 분포
- sorted angular density statistics: 각도 구간의 순서보다 분포 자체를 보는 통계
- Radon projection max/mean/std/entropy
- Radon peak angle과 peak strength
- horizontal/vertical/diagonal symmetry score
- rotated feature consistency score

### 5.5 특징 선택 계획

1. correlation heatmap으로 중복 특징 제거
2. Random Forest feature importance로 1차 후보 선정
3. XGBoost gain/cover/weight importance 비교
4. permutation importance로 검증
5. SHAP summary plot으로 최종 해석

## 6. 모델 계획

### 6.1 필수 baseline

| 모델 | 역할 |
| --- | --- |
| DummyClassifier | class imbalance 상황에서 최소 기준선 |
| Logistic Regression | 선형 분류 기준선 및 feature scaling 영향 확인 |
| SVM 또는 LinearSVM | 중간 규모 feature space 기준선 |
| Random Forest | 해석 가능한 tree ensemble baseline |
| XGBoost | 최종 주력 모델 후보 |

### 6.2 최종 proposed model 후보

1. **XGBoost + full handcrafted features**
   - 주요 제출 모델 후보
   - class weight 또는 sample weight 적용
   - Optuna 또는 GridSearchCV로 hyperparameter tuning

2. **Random Forest + selected interpretable features**
   - feature importance와 설명 가능성 중심 모델
   - XGBoost 대비 과적합 여부 비교

3. **Stacking ML ensemble**
   - Logistic Regression, SVM, Random Forest, XGBoost의 예측 확률을 meta feature로 사용
   - 프로젝트 시간이 충분할 경우 확장 실험으로 수행

### 6.3 CNN 제외 원칙

CNN, ResNet, EfficientNet, Vision Transformer 등 딥러닝 이미지 분류 모델은 구현 대상에서 제외한다. 다만 CNN 논문에서 제안한 회전/반전 불변성 아이디어는 Radon feature, radial profile, symmetry score 등 전통 머신러닝 feature로 변환하여 반영한다.

## 7. 평가 계획

### 7.1 기본 평가 지표

- accuracy
- macro F1-score
- weighted F1-score
- balanced accuracy
- class-wise precision/recall/F1
- confusion matrix

불균형 데이터 특성상 최종 비교는 accuracy보다 **macro F1-score와 balanced accuracy**를 중심으로 한다.

### 7.2 robustness 평가

| 평가셋 | 목적 |
| --- | --- |
| original test | 기본 일반화 성능 |
| rotated test | 회전 변화에 대한 안정성 확인 |
| flipped test | 좌우/상하 반전에 대한 안정성 확인 |
| small-label split | 라벨 부족 상황에서 모델 안정성 확인 |
| imbalanced split | 실제 class imbalance 상황 반영 |

### 7.3 ablation study

| 실험 | 비교 목적 |
| --- | --- |
| basic features only | 기본 통계 특징의 한계 확인 |
| basic + spatial features | 위치 정보의 효과 확인 |
| basic + spatial + component features | 군집/형상 정보의 효과 확인 |
| full features without Radon | Radon 제외 성능 확인 |
| full features with Radon | 기하학적 특징의 추가 효과 확인 |

## 8. 유사도 검색 계획

유사도 검색은 분류 모델과 독립적으로 feature vector 기반으로 구현한다.

1. train wafer map의 feature vector를 저장한다.
2. StandardScaler 또는 RobustScaler로 정규화한다.
3. cosine similarity 또는 Euclidean distance로 query wafer와 가까운 wafer를 찾는다.
4. top-k similar wafer map의 실제 label, 예측 label, similarity score를 반환한다.
5. 예측 class와 top-k majority class가 일치하는지 confidence 보조 지표로 사용한다.

## 9. 프로젝트 산출물

### 9.1 코드 산출물

- data loader
- preprocessing script
- feature extraction module
- baseline training script
- XGBoost/Random Forest training script
- evaluation script
- similarity retrieval script
- visualization notebook

### 9.2 보고서 산출물

- 선행 연구 비교표
- 데이터 EDA 결과
- feature 설계 근거
- 모델별 성능 비교
- robustness 평가 결과
- feature importance/SHAP 분석
- 유사도 검색 사례 분석

## 10. 단계별 일정

| 단계 | 목표 | 산출물 |
| --- | --- | --- |
| 1단계 | 목표 정의 및 선행 연구 정리 | 연구 질문, 차별화 포인트, 비교표 |
| 2단계 | 데이터 로딩 및 EDA | class distribution, sample visualization |
| 3단계 | feature extraction 구현 | feature table, feature description |
| 4단계 | baseline 모델 학습 | Dummy/LogReg/SVM/RF 성능표 |
| 5단계 | XGBoost 튜닝 | best parameter, validation score |
| 6단계 | robustness/ablation 평가 | 변형 테스트 및 feature group별 성능 |
| 7단계 | 설명 가능성 분석 | feature importance, SHAP plot |
| 8단계 | 유사도 검색 구현 | top-k retrieval example |
| 9단계 | 최종 보고서 작성 | 결과 해석 및 한계점 |

## 11. 발표용 핵심 메시지

본 프로젝트는 wafer map을 단순 이미지가 아니라 결함의 공간 분포 데이터로 해석한다. CNN을 사용하지 않고도 radial, edge, component, Radon, symmetry feature를 설계하여 XGBoost와 Random Forest로 실패 패턴을 분류하며, feature importance와 유사도 검색을 통해 예측 결과의 근거를 제시한다. 이를 통해 정확도뿐 아니라 설명 가능성과 현장 활용성을 강조하는 머신러닝 기반 wafer failure analysis를 목표로 한다.
