# My_project_0311

`My_project_0311`는 `housing.csv`를 사용해 머신러닝 프로그래밍 수업의 13, 14, 16, 19페이지 실습 내용을 정리한 폴더입니다.

주요 내용은 아래와 같습니다.
- 데이터 불러오기
- 결측치 확인 및 처리
- 표준화와 정규화
- 학습/검증/테스트 데이터 분할

## Quick Start

터미널에서 아래처럼 실행하면 됩니다.

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ML_Lecture/My_project_0311

python3 page13_code.py
python3 page14_code.py
python3 page16_code.py
python3 page19_code.py
```

## 폴더 구조

```text
My_project_0311/
├── README.md
├── page13_code.py
├── page14_code.py
├── page16_code.py
├── page19_code.py
└── datasets/
    ├── housing.csv
    ├── housing_train.csv
    ├── housing_valid.csv
    └── housing_test.csv
```

## 파일 설명

### 코드 파일

- `page13_code.py`
  - `housing.csv`를 불러옵니다.
  - 데이터 상위 5개 행과 데이터프레임 기본 정보를 출력합니다.

- `page14_code.py`
  - 결측치 개수를 확인합니다.
  - `total_bedrooms` 컬럼의 결측치를 중앙값으로 대체합니다.

- `page16_code.py`
  - 수치형 특성만 선택합니다.
  - 표준화(Standardization)와 Min-Max 정규화(Normalization)를 수행합니다.
  - 일부 환경에서는 `scikit-learn` 없이도 실행되도록 작성되었습니다.

- `page19_code.py`
  - 전체 데이터를 `6:2:2` 비율로 분할합니다.
  - 학습용, 검증용, 테스트용 CSV 파일을 생성합니다.

### 데이터 및 결과 파일

- `datasets/housing.csv`
  - 원본 주택 데이터 파일입니다.

- `datasets/housing_train.csv`
  - 학습용 데이터 파일입니다.

- `datasets/housing_valid.csv`
  - 검증용 데이터 파일입니다.

- `datasets/housing_test.csv`
  - 테스트용 데이터 파일입니다.

## 실행 결과 예시

- `page13_code.py`: 원본 데이터 구조와 컬럼 정보를 확인
- `page14_code.py`: 결측치 처리 전후 상태 확인
- `page16_code.py`: 스케일링 및 정규화 결과 확인
- `page19_code.py`: train/valid/test 분할 파일 저장

## 사용 라이브러리

- `pandas`
- `numpy`
- `pathlib`

## 비고

- 모든 파일 경로는 현재 `My_project_0311` 폴더 기준의 상대 경로로 구성되어 있습니다.
- 폴더 구조를 유지하면 다른 환경에서도 같은 방식으로 실행할 수 있습니다.
