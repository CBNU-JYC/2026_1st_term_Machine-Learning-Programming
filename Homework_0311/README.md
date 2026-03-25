# Homework_0311

`Homework_0311`는 `stockdata.csv`를 사용해 머신러닝 프로그래밍 수업의 13, 14, 16, 19페이지 실습 내용을 다시 구성한 폴더입니다.

주요 내용은 아래와 같습니다.
- 데이터 불러오기
- 결측치 확인 및 처리 흐름 실습
- 표준화와 정규화
- 학습/검증/테스트 데이터 분할

## Quick Start

터미널에서 아래처럼 실행하면 됩니다.

```bash
cd /Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/ML_Lecture/Homework_0311

python3 page13_code.py
python3 page14_code.py
python3 page16_code.py
python3 page19_code.py
```

## 폴더 구조

```text
Homework_0311/
├── README.md
├── page13_code.py
├── page14_code.py
├── page16_code.py
├── page19_code.py
├── stockdata.csv
├── stockdata_processed.csv
├── stockdata_standardized.csv
├── stockdata_normalized.csv
├── stock_train.csv
├── stock_valid.csv
└── stock_test.csv
```

## 파일 설명

### 코드 파일

- `page13_code.py`
  - `stockdata.csv`를 불러옵니다.
  - 데이터 상위 몇 행과 기본 정보를 확인합니다.

- `page14_code.py`
  - 결측치 개수를 확인합니다.
  - 예시로 `MSFT` 컬럼 기준 결측치 처리 흐름을 실습합니다.
  - 처리 결과를 `stockdata_processed.csv`로 저장합니다.

- `page16_code.py`
  - 수치형 컬럼을 선택합니다.
  - 표준화(Standardization)와 정규화(Normalization)를 수행합니다.
  - 결과를 각각 CSV 파일로 저장합니다.

- `page19_code.py`
  - 전체 데이터를 `6:2:2` 비율로 분할합니다.
  - 학습/검증/테스트용 CSV 파일을 생성합니다.

### 데이터 및 결과 파일

- `stockdata.csv`
  - 원본 주가 데이터 파일입니다.

- `stockdata_processed.csv`
  - 결측치 처리 예제를 실행한 결과 파일입니다.

- `stockdata_standardized.csv`
  - 표준화 결과가 저장된 파일입니다.

- `stockdata_normalized.csv`
  - 정규화 결과가 저장된 파일입니다.

- `stock_train.csv`
  - 학습용 데이터 파일입니다.

- `stock_valid.csv`
  - 검증용 데이터 파일입니다.

- `stock_test.csv`
  - 테스트용 데이터 파일입니다.

## 실행 결과 예시

- `page13_code.py`: 원본 데이터 구조와 컬럼 정보를 확인
- `page14_code.py`: 결측치 점검 및 처리 후 결과 저장
- `page16_code.py`: 표준화/정규화 값 계산 후 결과 저장
- `page19_code.py`: 데이터 분할 후 train/valid/test 파일 저장

## 사용 라이브러리

- `pandas`
- `numpy`
- `pathlib`

## 비고

- 모든 파일 경로는 현재 `Homework_0311` 폴더 기준의 상대 경로로 구성되어 있습니다.
- 폴더 구조를 유지하면 다른 환경에서도 같은 방식으로 실행할 수 있습니다.
