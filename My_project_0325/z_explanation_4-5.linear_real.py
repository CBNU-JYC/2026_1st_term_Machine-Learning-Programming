import pandas as pd  # 표 형태 데이터(csv 등)를 다루기 위해 pandas를 pd라는 이름으로 불러옵니다.
import numpy as np  # 숫자 계산용 numpy를 np라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 pyplot을 plt라는 이름으로 불러옵니다.
from pathlib import Path  # 현재 파일 위치를 기준으로 안전하게 경로를 만들기 위해 Path를 불러옵니다.
from sklearn.linear_model import LinearRegression  # 선형회귀 모델 클래스를 불러옵니다.
from sklearn.metrics import r2_score  # 모델의 설명력을 평가하는 R2 점수 함수를 불러옵니다.


# 1. 데이터 불러오기
file_path = Path(__file__).with_name("Salary_Data.csv")  # 현재 파이썬 파일과 같은 폴더에 있는 Salary_Data.csv 경로를 자동으로 만듭니다.
data = pd.read_csv(file_path)  # CSV 파일을 읽어서 표(DataFrame) 형태로 data에 저장합니다.


# 2. 데이터 분리 (X: 경력, y: 연봉)
X = data[["YearsExperience"]]  # 입력값 X로 YearsExperience 열만 선택합니다. 이중 대괄호는 표 형태를 유지하기 위해 사용합니다.
y = data["Salary"]  # 정답값 y로 Salary 열을 선택합니다.


# 3. 모델 생성 및 학습
model = LinearRegression()  # 선형회귀 모델 객체를 만듭니다.
model.fit(X, y)  # X와 y 데이터를 이용해 가장 잘 맞는 직선을 학습시킵니다.


# 4. 학습 결과 확인
w1 = model.coef_[0]  # 학습된 기울기를 꺼냅니다. 경력이 1년 늘 때 연봉이 얼마나 늘어나는지 뜻합니다.
w0 = model.intercept_  # 학습된 절편을 꺼냅니다. 경력이 0년일 때의 시작 연봉처럼 볼 수 있습니다.

print(f"=== AI 분석 결과 ===")  # 출력 구간 제목을 보여줍니다.
print(f"공식: Salary = {w1:.2f} * Experience + {w0:.2f}")  # 학습된 선형회귀 직선 공식을 출력합니다.
print(f"[1] 연봉 인상액: ${w1:.2f}")  # 기울기를 보기 쉽게 출력합니다.
print(f"[2] 예상 초봉: ${w0:.2f}")  # 절편을 보기 쉽게 출력합니다.


# 5. 모델 평가 및 미래 예측
y_pred = model.predict(X)  # 학습된 모델로 기존 X에 대한 예측 연봉을 계산합니다.
print(f"[3] 정확도(R2): {r2_score(y, y_pred):.4f}")  # 실제값과 예측값을 비교해 R2 점수를 계산해 출력합니다.

# 경력 15년차 예측 (Feature names warning 방지를 위해 DataFrame 사용)
X_new = pd.DataFrame([[15]], columns=["YearsExperience"])  # 새 입력값 15년 경력을 기존 X와 같은 열 이름의 DataFrame으로 만듭니다.
pred_15 = model.predict(X_new)  # 15년 경력자의 예상 연봉을 모델로 예측합니다.
print(f"[4] 15년차 예상 연봉: ${pred_15[0]:,.2f}")  # 예측 결과를 돈 형식처럼 보기 좋게 출력합니다.


# 6. 시각화
plt.figure(figsize=(10, 6))  # 가로 10, 세로 6 크기의 그래프 창을 만듭니다.
plt.scatter(X, y, color="blue", label="Actual Data")  # 실제 데이터를 파란 점으로 그래프에 표시합니다.
plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")  # 모델이 학습한 회귀직선을 빨간 선으로 그립니다.

plt.title("Salary vs Experience Analysis", fontsize=15)  # 그래프 제목을 설정합니다.
plt.xlabel("Years of Experience")  # x축 이름을 설정합니다.
plt.ylabel("Salary ($)")  # y축 이름을 설정합니다.
plt.legend()  # 점과 선이 무엇을 의미하는지 알려주는 범례를 표시합니다.
plt.grid(True, alpha=0.3)  # 배경 격자를 켜서 그래프를 보기 쉽게 만듭니다.
plt.show()  # 완성된 그래프를 화면에 띄웁니다.
