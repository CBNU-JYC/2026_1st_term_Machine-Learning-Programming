import numpy as np  # 수치 계산용 라이브러리 numpy를 np라는 짧은 이름으로 불러옵니다.
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 pyplot을 plt라는 이름으로 불러옵니다.
from sklearn.linear_model import LinearRegression  # 선형회귀 모델 클래스를 가져옵니다.
from sklearn.metrics import mean_squared_error, r2_score  # 모델 성능을 평가할 때 쓸 지표 2개를 가져옵니다.


np.random.seed(42)  # 랜덤값이 매번 똑같이 나오도록 씨앗(seed)을 42로 고정합니다.

# np.random.rand(100, 1)은 0~1 사이의 랜덤 숫자를 100행 1열로 만듭니다.
# 여기에 100을 곱해서 0~100 범위의 입력값 X를 만듭니다.
X = 100 * np.random.rand(100, 1)

# y는 정답 데이터입니다.
# 5 * X + 10은 "기울기 5, 절편 10"인 직선을 뜻합니다.
# np.random.randn(100, 1) * 10은 약간의 잡음(noise)을 추가해서 현실 데이터처럼 만듭니다.
y = 5 * X + 10 + np.random.randn(100, 1) * 10


# LinearRegression()으로 선형회귀 모델 객체를 만듭니다.
model = LinearRegression()

# fit(X, y)는 입력 X와 정답 y를 보고 가장 잘 맞는 직선을 학습합니다.
model.fit(X, y)


# model.coef_는 학습된 기울기(가중치)를 담고 있습니다.
# 입력 특성이 1개이므로 첫 번째 행, 첫 번째 열의 값만 꺼냅니다.
w1 = model.coef_[0][0]

# model.intercept_는 학습된 절편(bias)을 담고 있습니다.
# 절편은 x가 0일 때의 y값이라고 생각하면 됩니다.
w0 = model.intercept_[0]


# 아래부터는 학습된 결과를 화면에 보기 좋게 출력합니다.
print(f"=== Training Results ===")  # 학습 결과 제목을 출력합니다.
print(f"Estimated Slope (w1): {w1:.2f}")  # 기울기 값을 소수 둘째 자리까지 출력합니다.
print(f"Estimated Intercept (w0): {w0:.2f}")  # 절편 값을 소수 둘째 자리까지 출력합니다.
print(f"Final Equation: y = {w1:.2f}x + {w0:.2f}")  # 최종 직선 식을 y = ax + b 형태로 보여줍니다.


# predict(X)는 학습된 직선을 이용해서 각 X에 대한 예측값을 계산합니다.
y_pred = model.predict(X)

# mean_squared_error는 실제값과 예측값의 차이를 제곱해서 평균낸 값입니다.
# 값이 작을수록 예측이 더 잘 되었다고 볼 수 있습니다.
mse = mean_squared_error(y, y_pred)

# r2_score는 결정계수(R^2)입니다.
# 1에 가까울수록 모델이 데이터를 잘 설명한다고 해석합니다.
r2 = r2_score(y, y_pred)


# 아래부터는 모델 평가 결과를 출력합니다.
print(f"\n=== Model Evaluation ===")  # 줄바꿈 후 평가 결과 제목을 출력합니다.
print(f"Mean Squared Error (MSE): {mse:.2f}")  # 평균제곱오차를 소수 둘째 자리까지 출력합니다.
print(f"R-squared (R2 Score): {r2:.2f}")  # R^2 점수를 소수 둘째 자리까지 출력합니다.


# figure(figsize=(10, 6))은 그래프 창 크기를 가로 10, 세로 6 비율로 정합니다.
plt.figure(figsize=(10, 6))

# scatter는 실제 데이터를 점으로 그립니다.
# color="blue"는 파란색, alpha=0.6은 약간 투명하게, label은 범례 이름입니다.
plt.scatter(X, y, color="blue", alpha=0.6, label="Actual Data")

# plot은 예측된 직선을 선으로 그립니다.
# color="red"는 빨간색, linewidth=2는 선 두께, label은 범례 이름입니다.
plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")

# title은 그래프 제목을 설정합니다.
plt.title("Linear Regression: Watcha Likes vs Audience", fontsize=14)

# xlabel은 x축 이름을 설정합니다.
plt.xlabel("Watcha 'Like' Count", fontsize=12)

# ylabel은 y축 이름을 설정합니다.
plt.ylabel("Total Audience Count", fontsize=12)

# legend는 그래프에 표시된 선과 점의 이름표를 보여줍니다.
plt.legend()

# grid는 배경 격자를 켭니다.
# linestyle="--"는 점선, alpha=0.6은 약간 투명하게 보이게 합니다.
plt.grid(True, linestyle="--", alpha=0.6)

# show는 완성된 그래프 창을 화면에 띄웁니다.
plt.show()
