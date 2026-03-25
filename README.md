# 2026 1st Term Machine Learning Programming

충북대학교 산업인공지능학과 `머신러닝 프로그래밍(8884005-01)` 수업 실습 자료와 코드, 결과물을 정리한 저장소입니다.

이 저장소는 `ML_Lecture` 폴더 기준으로 구성되어 있으며, 강의 실습 코드와 생성 결과 파일을 날짜별 프로젝트 폴더로 관리합니다.

## 1. 강의 자료 링크

- EIS 강의 자료: [https://eisn.cbnu.ac.kr/nxui/index.html?OBSC_YN=0&LNG=ko#main](https://eisn.cbnu.ac.kr/nxui/index.html?OBSC_YN=0&LNG=ko#main)

## 2. GitHub 저장소 정보

- GitHub 계정: `draonfe73@chungbuk.ac.kr`
- 저장소 이름: `2026_1st_term_Machine-Learning-Programming`
- Git URL: [https://github.com/CBNU-JYC/2026_1st_term_Machine-Learning-Programming.git](https://github.com/CBNU-JYC/2026_1st_term_Machine-Learning-Programming.git)

## 3. 개발 환경

- Python 실행 환경: `Anaconda Navigator`의 `PyCharm` 가상환경
- 프로젝트 열기 경로: `/Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ML_Lecture/`
- 학습 및 코드 보조: `Codex AI`

## 4. 강의 노트 및 원본 파일 위치

- 맥북 로컬 경로:
  `/Users/jeong-yongcheol/Desktop/00_CBNU_AI/2_(대학원, 본교 E10-401)머신러닝 프로그래밍 (8884005-01) , 채민선교수`

## 5. 저장소 폴더 구조

```text
ML_Lecture/
├── README.md
├── Homework_0311/
│   ├── README.md
│   ├── page13_code.py
│   ├── page14_code.py
│   ├── page16_code.py
│   ├── page19_code.py
│   ├── stockdata.csv
│   ├── stockdata_processed.csv
│   ├── stockdata_standardized.csv
│   ├── stockdata_normalized.csv
│   ├── stock_train.csv
│   ├── stock_valid.csv
│   └── stock_test.csv
├── My_project_0311/
│   ├── datasets/
│   │   ├── housing.csv
│   │   ├── housing_train.csv
│   │   ├── housing_valid.csv
│   │   └── housing_test.csv
│   ├── page13_code.py
│   ├── page14_code.py
│   ├── page16_code.py
│   └── page19_code.py
└── My_project_0318/
    ├── page14_sigmoid_sample.py
    ├── page16_hypothesis_sample.py
    ├── page20_cost_sample.py
    ├── page22_gradient_descent_sample.py
    ├── page27_confusion_matrix_sample.py
    ├── page29_accuracy_sample.py
    ├── page33_metrics_sample.py
    ├── page36_roc_curve_sample.py
    ├── Explanation_page14_sigmoid_sample.py
    ├── Explanation_page16_hypothesis_sample.py
    ├── Explanation_page20_cost_sample.py
    ├── Explanation_page22_gradient_descent_sample.py
    ├── Explanation_page27_confusion_matrix_sample.py
    ├── Explanation_page29_accuracy_sample.py
    ├── Explanation_page33_metrics_sample.py
    ├── Explanation_page36_roc_curve_sample.py
    └── page14_sigmoid_sample.png
```

## 6. 폴더별 설명

### `Homework_0311`

- `stockdata.csv`를 기반으로 13, 14, 16, 19페이지 실습 코드를 재구성한 폴더입니다.
- 결측치 처리, 표준화, 정규화, 데이터 분할 결과까지 함께 저장되어 있습니다.

### `My_project_0311`

- `housing.csv`를 기반으로 13, 14, 16, 19페이지 실습을 정리한 폴더입니다.
- 데이터 로드, 결측치 처리, 스케일링, 데이터 분할 과정을 포함합니다.

### `My_project_0318`

- 3월 18일 실습 내용을 정리한 폴더입니다.
- 시그모이드 함수, 가설 함수, 비용 함수, 경사하강법, 혼동행렬, 정확도, 정밀도/재현율/F1, ROC Curve 등 분류 모델 평가 관련 예제가 포함되어 있습니다.
- `Explanation_*.py` 파일은 설명용 코드, `page*.py` 파일은 실행용 예제 코드입니다.

세부 파일 설명:

- `page14_sigmoid_sample.py`: 시그모이드 함수의 기본 형태와 입력값에 따른 출력 변화를 확인하는 예제입니다.
- `Explanation_page14_sigmoid_sample.py`: 시그모이드 함수의 의미와 동작 원리를 설명 중심으로 정리한 코드입니다.
- `page14_sigmoid_sample.png`: 시그모이드 함수 시각화 결과 이미지입니다.

- `page16_hypothesis_sample.py`: 로지스틱 회귀에서 사용하는 가설 함수 예제입니다.
- `Explanation_page16_hypothesis_sample.py`: 가설 함수가 입력 특성으로부터 예측 확률을 만드는 과정을 설명합니다.

- `page20_cost_sample.py`: 비용 함수의 계산 흐름과 예측 오차 반영 방식을 확인하는 예제입니다.
- `Explanation_page20_cost_sample.py`: 비용 함수가 왜 필요한지와 학습 과정에서의 역할을 설명합니다.

- `page22_gradient_descent_sample.py`: 경사하강법을 이용해 비용을 줄여가는 과정을 구현한 예제입니다.
- `Explanation_page22_gradient_descent_sample.py`: 경사하강법의 업데이트 원리와 학습률 개념을 설명합니다.

- `page27_confusion_matrix_sample.py`: 혼동행렬을 통해 분류 결과를 표 형태로 분석하는 예제입니다.
- `Explanation_page27_confusion_matrix_sample.py`: TP, TN, FP, FN의 의미와 해석 방법을 설명합니다.

- `page29_accuracy_sample.py`: 정확도(Accuracy)를 계산하는 기본 예제입니다.
- `Explanation_page29_accuracy_sample.py`: 정확도가 어떤 상황에서 유용하고 한계가 무엇인지 설명합니다.

- `page33_metrics_sample.py`: 정밀도, 재현율, F1-score 등 주요 분류 평가지표를 계산하는 예제입니다.
- `Explanation_page33_metrics_sample.py`: 각 평가지표의 의미와 활용 상황을 설명합니다.

- `page36_roc_curve_sample.py`: ROC Curve와 분류 임계값 변화에 따른 성능 비교 예제입니다.
- `Explanation_page36_roc_curve_sample.py`: ROC Curve와 AUC의 개념, 해석 방법을 설명합니다.

## 7. 실행 방법

터미널 또는 PyCharm에서 아래 예시처럼 실행할 수 있습니다.

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ML_Lecture

python3 Homework_0311/page13_code.py
python3 Homework_0311/page14_code.py
python3 Homework_0311/page16_code.py
python3 Homework_0311/page19_code.py

python3 My_project_0311/page13_code.py
python3 My_project_0311/page14_code.py
python3 My_project_0311/page16_code.py
python3 My_project_0311/page19_code.py
```

## 8. GitHub 업로드 참고

- 참고 유튜브: [기존 프로젝트 Github(Remote Repository)에 올리기](https://youtu.be/AOn6UUscqQw?si=eo5WK_GSbmj-4LSi)

체크리스트:

- [ ] 준비물: `git scm`, `pycharm`, `github.com id`
- [ ] PyCharm에서 git에 올릴 프로젝트 불러오기
- [ ] GitHub에서 새 repository 만들기
- [ ] `git init` 하기
- [ ] `.py` 파일 하나 만들기
- [ ] `git add` 하기
- [ ] `git commit` 하기
- [ ] `git remote`에 GitHub 새 저장소 주소 등록하기
- [ ] `git push` 하기
- [ ] 올라갔는지 확인하기
- [ ] collaborator 추가하기

## 9. 교과목 정보

- 개설연도-학기: `2026년 1학기`
- 개설학과: `산업인공지능학`
- 교과목번호-분반번호: `8884005-01`
- 교과목명: `머신러닝 프로그래밍`
- 이수구분: `전공심화`
- 학점/시수: `3-3-0`
- 강의시간/강의실: `수 11, 12, 13 / E10-401`
- 담당교수: `채민선(초빙교원)`
- E-mail: `mschae@cbnu.ac.kr`
- 학과전화: `043-249-1257`

## 10. 교과목 개요

머신러닝의 기본 개념과 주요 알고리즘을 이론과 실습을 통해 학습합니다.  
분류, 회귀, 서포트 벡터 머신, 결정트리, 앙상블 학습, 차원 축소, 군집 분석, 이상탐지 등 전통적 머신러닝 기법을 단계적으로 학습하며, 각 알고리즘의 적용 목적과 한계를 이해하는 것을 목표로 합니다.

또한 모델 훈련, 성능 평가, 하이퍼파라미터 튜닝 등 실무 및 연구 현장에서 필요한 머신러닝 개발 절차를 경험하고, 직접 구현한 결과를 해석하는 능력을 기릅니다.

## 11. 학습 목표

- 머신러닝의 기본 개념과 학습 유형을 이해하고 문제 유형에 맞는 접근 방법을 설명할 수 있다.
- 머신러닝 알고리즘의 원리와 특징을 이해하고 활용할 수 있다.
- 주어진 데이터와 문제 상황에 대해 적절한 모델을 선택하고 프로젝트 형태로 문제를 해결할 수 있다.

## 12. 교재

1. 주교재: `핸즈온 머신러닝(3판)`, 오렐리앙 제롱, 한빛미디어, 2023
2. 부교재: `데싸노트의 실전에서 통하는 머신러닝`, 권시현, 골든래빗, 2022
3. 부교재: `머신러닝 프로그래밍`, 김성수, 2021

## 13. 비고

- 이 저장소는 수업 실습 결과물과 코드 버전을 날짜별 폴더로 관리합니다.
- 일부 코드는 `scikit-learn` 없이 `numpy`와 `pandas`만으로 실행 가능하도록 수정되어 있습니다.
- 생성된 CSV 파일과 결과물도 함께 저장하여 재현성을 높였습니다.
