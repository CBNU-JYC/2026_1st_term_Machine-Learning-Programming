# 2026 1학기 머신러닝 프로그래밍

![GitHub repo size](https://img.shields.io/github/repo-size/CBNU-JYC/2026_1st_term_Machine-Learning-Programming)
![GitHub last commit](https://img.shields.io/github/last-commit/CBNU-JYC/2026_1st_term_Machine-Learning-Programming)
![Python](https://img.shields.io/badge/Python-Anaconda%20Environment-3776AB?logo=python&logoColor=white)
![PyCharm](https://img.shields.io/badge/Editor-PyCharm-000000?logo=pycharm&logoColor=white)
![Course](https://img.shields.io/badge/Course-Machine%20Learning%20Programming-0A7E8C)

충북대학교 산업인공지능학 전공 `머신러닝 프로그래밍` 강의의 실습 코드, 설명 파일, 실행 결과물, 환경 설정을 정리한 저장소입니다.

이 저장소는 강의 실습 내용을 날짜별 프로젝트 형태로 정리하고, Python 실습 코드와 설명용 파이썬 파일, 생성된 데이터셋 결과물까지 함께 관리하는 것을 목표로 합니다.

## 한눈에 보기

- 강의명: `머신러닝 프로그래밍`
- 학기: `2026년 1학기`
- 저장소: <https://github.com/CBNU-JYC/2026_1st_term_Machine-Learning-Programming.git>
- 개발 환경: `Anaconda Navigator + PyCharm`
- 실습 핵심 주제:
  - 데이터 불러오기와 기초 확인
  - 결측치 처리
  - 표준화와 정규화
  - 데이터 분할
  - 시그모이드 함수, 비용 함수, 경사하강법
  - 혼동행렬, 정확도, 정밀도, 재현율, F1-score, ROC Curve
  - 선형회귀, 최소자승법, OLS와 SGD 비교
  - 확률적/미니배치/전체배치 경사하강법 비교
  - Ridge, Lasso 정규화와 회귀 성능 해석
- 하위 폴더 바로가기:
  - [Homework_0311](./Homework_0311/README.md)
  - [My_project_0311](./My_project_0311/README.md)
  - [My_project_0318](./My_project_0318/README.md)
  - [My_project_0401](./My_project_0401/)

## 1. 저장소 정보

- 저장소 이름: `2026_1st_term_Machine-Learning-Programming`
- GitHub 주소: <https://github.com/CBNU-JYC/2026_1st_term_Machine-Learning-Programming.git>
- GitHub 계정: `draonfe73@chungbuk.ac.kr`

## 2. 강의 자료 링크

- 충북대학교 EISN 강의 자료 링크:
  <https://eisn.cbnu.ac.kr/nxui/index.html?OBSC_YN=0&LNG=ko#main>

## 3. 교과목 정보

- 개설연도-학기: `2026년 1학기`
- 개설학과: `산업인공지능학`
- 교과목번호-분반번호: `8884005-01`
- 교과목명: `머신러닝 프로그래밍`
- 이수구분: `전공심화`
- 학점/시수: `3-3-0`
- 강의시간/강의실: `수 11, 12, 13 [E10-401]`
- 담당교수: `채민선(초빙교원)`
- 전화: `0000000000`
- 이메일: `mschae@cbnu.ac.kr`
- 학과전화: `043-249-1257`

## 4. 교과목 개요

### 강의 개요

머신러닝의 기본 개념과 주요 알고리즘을 이론과 실습을 통해 학습합니다.

분류, 회귀, 서포트 벡터 머신, 결정트리, 앙상블 학습, 차원 축소, 군집 분석, 이상탐지 등 전통적 머신러닝 기법을 단계적으로 학습하며, 각 알고리즘의 적용 목적과 한계를 이해하는 것을 목표로 합니다.

또한 모델 훈련, 성능 평가, 하이퍼파라미터 튜닝 등 실무 및 연구 현장에서 필요한 머신러닝 개발 절차를 경험하고, 직접 구현한 결과를 해석하는 능력을 기릅니다.

### 학습목표

1. 머신러닝의 기본 개념과 학습 유형을 이해하고 문제 유형에 맞는 접근 방법을 설명할 수 있다.
2. 머신러닝 알고리즘의 원리와 특징을 이해하고 활용할 수 있다.
3. 주어진 데이터와 문제 상황에 대해 적절한 모델을 선택하고 프로젝트 형태로 문제를 해결할 수 있다.

## 5. 개발 및 실행 환경

- Python 환경: `Anaconda Navigator` 기반의 `PyCharm` 가상환경
- 프로젝트 열기 경로:
  `/Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ML_Lecture/`
- 실습 중 `Codex AI`의 도움을 받아 코드 설명 파일 작성, 실행 확인, README 정리 작업을 보조함

## 6. 빠른 시작

### 저장소 열기

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ML_Lecture
```

### 권장 실행 환경

- `Anaconda Navigator`에서 사용하는 Python 환경을 활성화한 뒤 실행
- `PyCharm`에서 위 프로젝트 경로를 그대로 열어 실습

## 설명용 코드 작성 지침

초보자용 파이썬 코드에는 원본 실행 파일과 별도로 설명 전용 파일을 만들어 정리합니다.

- 설명용 파일명은 원본 파일명 앞에 `z_explanation_`을 붙여 작성합니다.
- 설명용 파일에는 초보자도 이해할 수 있도록 자세한 한국어 주석을 추가합니다.
- 주석은 `PEP 8` 스타일에 맞게 `#` 뒤를 한 칸 띄워 작성합니다.

### 설명용 코드 작성 요구사항

1. 각 줄 또는 각 구문마다 `#` 주석으로 해당 코드가 무슨 일을 하는지 초등학생도 이해할 수 있게 설명합니다.
2. 함수 위에는 함수의 역할, 매개변수 설명, 반환값 설명을 포함한 docstring을 작성합니다.
3. 파일 맨 위에는 프로그램 전체 흐름을 간단히 설명하는 안내를 적습니다.
4. 복잡한 부분은 왜 이렇게 작성했는지도 함께 설명합니다.
5. 응답 또는 정리 결과는 주석이 추가된 완성 코드 기준으로 제공합니다.

### 대표 실습 예시 1: 주택 데이터 불러오기

```bash
cd My_project_0311
python3 page13_code.py
```

### 대표 실습 예시 2: 주가 데이터 전처리

```bash
cd Homework_0311
python3 page16_code.py
```

### 대표 실습 예시 3: 경사하강법 실습

```bash
cd My_project_0318
python3 page22_gradient_descent_sample.py
```

### 대표 실습 예시 4: 선형회귀 심화 실습

```bash
cd My_project_0401
python3 5-1_ordinary_least_squares.py
python3 5-2_comparison.py
python3 5-3_comparison.py
python3 5-4_lasso_ridge.py
```

## 7. 대표 실행 예시

### 주택 데이터 불러오기 예시

```text
---- Housing Data Head ----
longitude  latitude  ...  median_house_value  ocean_proximity
...

--- Housing Data Info ---
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
...
```

### 주가 데이터 정규화 결과 예시

- `page16_code.py` 실행 시 `stockdata_standardized.csv`와 `stockdata_normalized.csv`가 생성됩니다.
- `page19_code.py` 실행 시 `stock_train.csv`, `stock_valid.csv`, `stock_test.csv`가 생성됩니다.

### 분류 개념 실습 예시

- `page14_sigmoid_sample.py` 실행 시 시그모이드 함수 개념을 확인할 수 있습니다.
- `page22_gradient_descent_sample.py` 실행 시 경사하강법의 파라미터 업데이트 과정을 확인할 수 있습니다.
- `page27_confusion_matrix_sample.py`, `page33_metrics_sample.py`, `page36_roc_curve_sample.py`를 통해 분류 성능 평가 흐름을 확인할 수 있습니다.

### 선형회귀 실습 예시

- `5-1_ordinary_least_squares.py` 실행 시 OLS 기반 선형회귀, `RMSE`, `R-squared`를 확인할 수 있습니다.
- `5-2_comparison.py` 실행 시 `Stochastic`, `Mini-batch`, `Full-batch` 경사하강법 차이를 비교할 수 있습니다.
- `5-3_comparison.py` 실행 시 `OLS`와 `SGD`의 초기 상태 및 수렴 결과를 비교할 수 있습니다.
- `5-4_lasso_ridge.py` 실행 시 `L1(Lasso)`와 `L2(Ridge)`의 가중치 분포 차이를 확인할 수 있습니다.

## 8. 강의 노트 및 원본 자료 위치

- 맥북 로컬 참고 경로:
  `/Users/jeong-yongcheol/Desktop/00_CBNU_AI/2_(대학원, 본교 E10-401)머신러닝 프로그래밍 (8884005-01) , 채민선교수`

## 9. 주차별 학습 정리

| 주차/폴더 | 핵심 주제 | 주요 파일 | 결과물 |
|---|---|---|---|
| `My_project_0311` | 데이터 불러오기, 결측치 처리, 스케일링, 데이터 분할 | `page13_code.py`, `page14_code.py`, `page16_code.py`, `page19_code.py` | `housing_train.csv`, `housing_valid.csv`, `housing_test.csv` |
| `Homework_0311` | 주가 데이터 기반 전처리 및 분할 실습 | `page13_code.py`, `page14_code.py`, `page16_code.py`, `page19_code.py` | `stockdata_processed.csv`, `stockdata_standardized.csv`, `stockdata_normalized.csv`, `stock_train.csv`, `stock_valid.csv`, `stock_test.csv` |
| `My_project_0318` | 로지스틱 회귀 개념 및 분류 성능 평가 | `page14_sigmoid_sample.py`, `page22_gradient_descent_sample.py`, `page27_confusion_matrix_sample.py`, `page33_metrics_sample.py`, `page36_roc_curve_sample.py` | `page14_sigmoid_sample.png` |
| `My_project_0401` | 선형회귀, OLS, SGD 비교, Ridge/Lasso 정규화 | `5-1_ordinary_least_squares.py`, `5-2_comparison.py`, `5-3_comparison.py`, `5-4_lasso_ridge.py`, `z_explanation_*.py` | 회귀선 시각화, 배치 방식 비교 그래프, 정규화 비교 그래프 |
| `Homework_0325` | 후속 실습용 폴더 | 현재 비어 있음 | 추후 추가 예정 |

## 10. 현재 저장소 구조

```text
ML_Lecture/
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
├── Homework_0325/
├── My_project_0311/
│   ├── README.md
│   ├── page13_code.py
│   ├── page14_code.py
│   ├── page16_code.py
│   ├── page19_code.py
│   └── datasets/
│       ├── housing.csv
│       ├── housing_train.csv
│       ├── housing_valid.csv
│       └── housing_test.csv
├── My_project_0318/
│   ├── README.md
│   ├── page14_sigmoid_sample.py
│   ├── page16_hypothesis_sample.py
│   ├── page20_cost_sample.py
│   ├── page22_gradient_descent_sample.py
│   ├── page27_confusion_matrix_sample.py
│   ├── page29_accuracy_sample.py
│   ├── page33_metrics_sample.py
│   ├── page36_roc_curve_sample.py
│   ├── z_Explanation_page14_sigmoid_sample.py
│   ├── z_Explanation_page16_hypothesis_sample.py
│   ├── z_Explanation_page20_cost_sample.py
│   ├── z_Explanation_page22_gradient_descent_sample.py
│   ├── z_Explanation_page27_confusion_matrix_sample.py
│   ├── z_Explanation_page29_accuracy_sample.py
│   ├── z_Explanation_page33_metrics_sample.py
│   ├── z_Explanation_page36_roc_curve_sample.py
│   └── page14_sigmoid_sample.png
├── My_project_0401/
│   ├── 5-1_ordinary_least_squares.py
│   ├── 5-2_comparison.py
│   ├── 5-3_comparison.py
│   ├── 5-4_lasso_ridge.py
│   ├── z_explanation_5-1_ordinary_least_squares.py
│   ├── z_explanation_5-2_comparison.py
│   ├── z_explanation_5-3_comparison.py
│   └── z_explanation_5-4_lasso_ridge.py
├── .gitignore
└── README.md
```

## 11. 디렉토리별 설명

### `Homework_0311`

- `stockdata.csv`를 기반으로 13, 14, 16, 19페이지 실습을 재구성한 폴더입니다.
- 결측치 처리, 표준화, 정규화, 데이터 분할을 실습합니다.
- 실행 결과로 여러 CSV 파일이 생성되어 함께 저장되어 있습니다.
- 자세한 설명은 [Homework_0311/README.md](./Homework_0311/README.md)를 참고합니다.

### `My_project_0311`

- `housing.csv`를 기반으로 13, 14, 16, 19페이지 실습을 정리한 폴더입니다.
- 데이터 로드, 결측치 처리, 스케일링, 데이터 분할 과정을 포함합니다.
- `datasets/` 폴더에 원본과 분할 결과 파일이 함께 정리되어 있습니다.
- 자세한 설명은 [My_project_0311/README.md](./My_project_0311/README.md)를 참고합니다.

### `My_project_0318`

- 3월 18일 분류 개념 중심 실습을 정리한 폴더입니다.
- 시그모이드 함수, 가설 함수, 비용 함수, 경사하강법, 혼동행렬, 정확도, 정밀도/재현율/F1, ROC Curve 예제가 포함되어 있습니다.
- `page*.py`는 실행용 예제, `z_Explanation_*.py`는 설명용 코드입니다.
- 자세한 설명은 [My_project_0318/README.md](./My_project_0318/README.md)를 참고합니다.

### `Homework_0325`

- 3월 25일 이후 실습을 위한 작업 폴더입니다.
- 현재는 비어 있으며, 후속 실습 자료가 추가될 예정입니다.

### `My_project_0401`

- 4월 1일 선형회귀 심화 실습을 정리한 폴더입니다.
- `5-1_ordinary_least_squares.py`는 최소자승법 기반 선형회귀와 성능지표 계산을 다룹니다.
- `5-2_comparison.py`와 `5-3_comparison.py`는 `SGD`, `Mini-batch`, `Full-batch`, `OLS`의 차이를 비교합니다.
- `5-4_lasso_ridge.py`는 `Ridge(L2)`와 `Lasso(L1)` 정규화의 가중치 특성을 비교합니다.
- `z_explanation_*.py` 파일은 같은 코드를 쉬운 한국어 주석으로 풀어 쓴 복습용 버전입니다.

## 12. 포함 파일 안내

- 이 저장소에는 코드 파일뿐 아니라 실습 중 생성된 CSV 결과물도 함께 포함되어 있습니다.
- 일부 폴더에는 실행 중 자동 생성된 `__pycache__`가 로컬에 존재할 수 있으나, Git 추적 대상은 아닙니다.
- 설명용 파일은 강의 복습과 코드 이해를 돕기 위해 별도로 작성한 버전입니다.
- 설명 파일 이름은 폴더에 따라 `z_Explanation_*` 또는 `z_explanation_*` 형식을 사용합니다.

## 13. GitHub 업로드 관련 메모

- 참고 유튜브:
  <https://youtu.be/AOn6UUscqQw?si=eo5WK_GSbmj-4LSi>
- 주제: `기존 프로젝트 Github(Remote Repository)에 올리기`

체크리스트:

- [ ] 준비물: `git scm`, `pycharm`, `github.com id`
- [ ] pycharm에서 git에 올릴 프로젝트 불러오기(새 project 만들기)
- [ ] github.com에서 새 repository 만들기
- [ ] `git init` 하기
- [ ] `.py` 파일 하나 만들기
- [ ] `git add` 하기
- [ ] `git commit` 하기
- [ ] `git remote`에 GitHub 새 저장소 주소 등록하기
- [ ] `git push` 하기
- [ ] 업로드 확인하기
- [ ] collaborator 추가하기

## 14. 참고 사항

- 강의 실습 환경과 로컬 경로 설명은 작성 당시 사용 환경을 기준으로 기록했습니다.
- `My_project_0401`의 예제는 `Salary_Data.csv`를 기준으로 실행되도록 구성했습니다.
- 이후 주차 실습이 추가되면 README의 저장소 구조와 주차별 학습 정리 표도 함께 갱신하는 것을 권장합니다.
