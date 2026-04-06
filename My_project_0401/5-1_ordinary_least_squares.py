from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 1. 파일 데이터 불러오기
base_dir = Path(__file__).resolve().parent
candidate_paths = [
    base_dir / "Salary_Data.csv",
    base_dir.parent / "My_project_0325" / "Salary_Data.csv",
    Path("/Users/jeong-yongcheol/Desktop/00_CBNU_AI/2_(대학원, 본교 E10-401)머신러닝 프로그래밍 (8884005-01) , 채민선교수/2. 수업 자료/Salary_Data.csv"),
]

file_path = next((path for path in candidate_paths if path.exists()), None)
if file_path is None:
    raise FileNotFoundError("Salary_Data.csv 파일을 찾을 수 없습니다.")

dataset = pd.read_csv(file_path)


# 데이터 구조 확인
print("데이터셋 정보")
print(dataset.info())
print("\n- 데이터 상위 5행 -")
print(dataset.head())


# 독립 변수(X: 연차)와 종속 변수(y: 연봉) 설정
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# 2. 데이터 분리 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. 모델 생성 및 학습 (내부적으로 최소자승법 OLS 실행)
model = LinearRegression()
model.fit(X_train, y_train)


# 4. 결과 예측 및 평가
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- [실습 결과] ---")
print(f"기울기(Weight, w): {model.coef_[0]:.2f}")
print(f"절편(Bias, b): {model.intercept_:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.4f}")


# 5. 시각화: 모델이 찾은 '최적의 직선' 확인
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="darkorange", label="Actual Data")
plt.plot(
    X,
    model.predict(X),
    color="royalblue",
    linewidth=2,
    label="OLS Line",
)
plt.title("Salary vs Experience (OLS Regression)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(visible=True, linestyle="--", alpha=0.6)
plt.show()
