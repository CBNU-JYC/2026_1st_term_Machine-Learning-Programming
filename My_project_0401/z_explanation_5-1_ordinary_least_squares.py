from pathlib import Path  # 파일/폴더 경로를 운영체제에 맞게 안전하게 다루기 위한 도구

import matplotlib.pyplot as plt  # 그래프를 그릴 때 사용하는 라이브러리
import numpy as np  # 수치 계산용 라이브러리
import pandas as pd  # 표 형태 데이터 처리용 라이브러리
from sklearn.linear_model import LinearRegression  # 선형회귀 모델을 불러옴
from sklearn.metrics import mean_squared_error, r2_score  # 성능평가 지표 계산 함수
from sklearn.model_selection import train_test_split  # 데이터를 학습용/테스트용으로 나누는 함수


# 1. 파일 데이터 불러오기
base_dir = Path(__file__).resolve().parent  # 지금 이 파이썬 파일이 들어있는 폴더 위치를 구함
candidate_paths = [  # Salary_Data.csv가 있을 만한 후보 경로들을 리스트로 준비
    base_dir / "Salary_Data.csv",  # 현재 폴더 안에 있는 Salary_Data.csv를 먼저 찾음
    base_dir.parent / "My_project_0325" / "Salary_Data.csv",  # 바로 못 찾으면 이전 폴더도 확인
    Path("/Users/jeong-yongcheol/Desktop/00_CBNU_AI/2_(대학원, 본교 E10-401)머신러닝 프로그래밍 (8884005-01) , 채민선교수/2. 수업 자료/Salary_Data.csv"),  # 원본 수업자료 경로도 마지막 후보로 확인
]

file_path = next((path for path in candidate_paths if path.exists()), None)  # 실제로 존재하는 첫 번째 파일 경로를 선택
if file_path is None:  # 어떤 후보 경로에도 파일이 없으면
    raise FileNotFoundError("Salary_Data.csv 파일을 찾을 수 없습니다.")  # 에러를 발생시켜 사용자에게 알려줌

dataset = pd.read_csv(file_path)  # CSV 파일을 읽어서 표(DataFrame) 형태로 저장


# 데이터 구조 확인
print("데이터셋 정보")  # 어떤 데이터인지 출력 시작 안내문
print(dataset.info())  # 열 이름, 데이터 개수, 자료형 등의 요약 정보 출력
print("\n- 데이터 상위 5행 -")  # 위 5개 행을 보여주기 전 제목 출력
print(dataset.head())  # 데이터의 앞부분 5줄을 출력해서 내용 확인


# 독립 변수(X: 연차)와 종속 변수(y: 연봉) 설정
X = dataset.iloc[:, :-1].values  # 마지막 열을 제외한 모든 열을 입력값 X로 사용
y = dataset.iloc[:, -1].values  # 마지막 열만 정답값 y로 사용


# 2. 데이터 분리 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(  # 데이터를 학습용과 테스트용으로 나눔
    X, y, test_size=0.2, random_state=42  # 전체의 20%를 테스트용으로 쓰고, random_state는 결과 재현용
)


# 3. 모델 생성 및 학습 (내부적으로 최소자승법 OLS 실행)
model = LinearRegression()  # 선형회귀 모델 객체를 생성
model.fit(X_train, y_train)  # 학습용 데이터를 사용해 가장 잘 맞는 직선을 찾도록 학습


# 4. 결과 예측 및 평가
y_pred = model.predict(X_test)  # 테스트용 입력값에 대해 예측값을 계산

rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 평균제곱오차(MSE)에 루트를 씌워 RMSE 계산
r2 = r2_score(y_test, y_pred)  # 모델이 데이터를 얼마나 잘 설명하는지 R² 값 계산

print(f"\n--- [실습 결과] ---")  # 결과 출력 구역 시작 표시
print(f"기울기(Weight, w): {model.coef_[0]:.2f}")  # 회귀직선의 기울기 값 출력
print(f"절편(Bias, b): {model.intercept_:.2f}")  # 회귀직선의 절편 값 출력
print(f"RMSE: {rmse:.2f}")  # 예측 오차 크기를 RMSE로 출력
print(f"R-squared: {r2:.4f}")  # 설명력을 나타내는 R² 값을 출력


# 5. 시각화: 모델이 찾은 '최적의 직선' 확인
plt.figure(figsize=(10, 6))  # 그래프 창 크기를 가로 10, 세로 6으로 설정
plt.scatter(X, y, color="darkorange", label="Actual Data")  # 실제 데이터 점들을 산점도로 그림
plt.plot(  # 선형회귀가 찾은 직선을 그래프에 그림
    X,  # x축에 들어갈 값
    model.predict(X),  # 전체 X에 대한 예측값을 y축으로 사용
    color="royalblue",  # 선 색상을 파란색 계열로 지정
    linewidth=2,  # 선 두께를 2로 설정
    label="OLS Line",  # 범례에 표시할 이름
)
plt.title("Salary vs Experience (OLS Regression)")  # 그래프 제목 설정
plt.xlabel("Years of Experience")  # x축 이름 설정
plt.ylabel("Salary")  # y축 이름 설정
plt.legend()  # 범례 표시
plt.grid(visible=True, linestyle="--", alpha=0.6)  # 배경 격자를 점선 형태로 표시
plt.show()  # 그래프 창을 화면에 출력
