# My_project_0318

`My_project_0318`는 2026년 3월 18일 머신러닝 프로그래밍 수업 실습 내용을 정리한 폴더입니다.

이 폴더에서는 주로 로지스틱 회귀와 분류 성능 평가 개념을 예제 코드로 다룹니다.

주요 주제:
- 시그모이드 함수
- 가설 함수
- 비용 함수
- 경사하강법
- 혼동행렬
- 정확도
- 정밀도, 재현율, F1-score
- ROC Curve, AUC

## Quick Start

터미널에서 아래처럼 실행하면 됩니다.

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ML_Lecture/My_project_0318

python3 page14_sigmoid_sample.py
python3 page16_hypothesis_sample.py
python3 page20_cost_sample.py
python3 page22_gradient_descent_sample.py
python3 page27_confusion_matrix_sample.py
python3 page29_accuracy_sample.py
python3 page33_metrics_sample.py
python3 page36_roc_curve_sample.py
```

## 폴더 구조

```text
My_project_0318/
├── README.md
├── page14_sigmoid_sample.py
├── page16_hypothesis_sample.py
├── page20_cost_sample.py
├── page22_gradient_descent_sample.py
├── page27_confusion_matrix_sample.py
├── page29_accuracy_sample.py
├── page33_metrics_sample.py
├── page36_roc_curve_sample.py
├── z_Explanation_page14_sigmoid_sample.py
├── z_Explanation_page16_hypothesis_sample.py
├── z_Explanation_page20_cost_sample.py
├── z_Explanation_page22_gradient_descent_sample.py
├── z_Explanation_page27_confusion_matrix_sample.py
├── z_Explanation_page29_accuracy_sample.py
├── z_Explanation_page33_metrics_sample.py
├── z_Explanation_page36_roc_curve_sample.py
└── page14_sigmoid_sample.png
```

## 파일 설명

### 주제별 파일 정리

- 시그모이드 함수
  - `page14_sigmoid_sample.py`: 시그모이드 함수의 기본 형태와 출력 변화를 확인하는 예제입니다.
  - `z_Explanation_page14_sigmoid_sample.py`: 시그모이드 함수의 개념과 동작을 설명합니다.
  - `page14_sigmoid_sample.png`: 시그모이드 함수 시각화 결과 이미지입니다.

- 가설 함수
  - `page16_hypothesis_sample.py`: 로지스틱 회귀의 가설 함수를 예제로 보여주는 코드입니다.
  - `z_Explanation_page16_hypothesis_sample.py`: 가설 함수의 의미와 입력-출력 관계를 설명합니다.

- 비용 함수
  - `page20_cost_sample.py`: 비용 함수 계산 과정을 실습하는 코드입니다.
  - `z_Explanation_page20_cost_sample.py`: 비용 함수의 필요성과 계산 의미를 설명합니다.

- 경사하강법
  - `page22_gradient_descent_sample.py`: 경사하강법을 이용해 비용을 줄여가는 과정을 구현한 코드입니다.
  - `z_Explanation_page22_gradient_descent_sample.py`: 경사하강법의 업데이트 원리와 학습률 개념을 설명합니다.

- 혼동행렬
  - `page27_confusion_matrix_sample.py`: 혼동행렬 계산 및 분류 결과 분석 예제입니다.
  - `z_Explanation_page27_confusion_matrix_sample.py`: 혼동행렬의 각 구성요소(TP, TN, FP, FN)를 설명합니다.

- 정확도
  - `page29_accuracy_sample.py`: 정확도(Accuracy)를 계산하는 예제입니다.
  - `z_Explanation_page29_accuracy_sample.py`: 정확도의 해석과 한계를 설명합니다.

- 정밀도/재현율/F1-score
  - `page33_metrics_sample.py`: 정밀도, 재현율, F1-score를 계산하는 예제입니다.
  - `z_Explanation_page33_metrics_sample.py`: 각 평가지표의 의미를 설명합니다.

- ROC Curve / AUC
  - `page36_roc_curve_sample.py`: ROC Curve와 AUC 개념을 실습하는 예제입니다.
  - `z_Explanation_page36_roc_curve_sample.py`: ROC Curve와 AUC 해석 방법을 설명합니다.

### 파일 구분

- `page*.py` 파일은 직접 실행용 예제입니다.
- `z_Explanation_*.py` 파일은 개념 설명 중심 코드입니다.

## 실행 결과 예시

- 시그모이드 함수 그래프 확인
- 가설 함수 출력값 변화 확인
- 비용 함수와 경사하강법 계산 흐름 확인
- 분류 성능 평가 지표 계산 결과 확인

## 사용 라이브러리

- `pandas`
- `numpy`
- `matplotlib`
- `pathlib`

## 비고

- `page*.py` 파일은 직접 실행용 예제입니다.
- `z_Explanation_*.py` 파일은 개념 설명 중심 코드입니다.
- 시각화 관련 코드는 실행 환경에 따라 그래프 창이 열릴 수 있습니다.
