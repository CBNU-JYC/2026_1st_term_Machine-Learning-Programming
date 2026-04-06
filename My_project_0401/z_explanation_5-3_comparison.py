from pathlib import Path  # 파일 경로를 편하게 다루기 위한 도구

import matplotlib.pyplot as plt  # 그래프를 그리는 라이브러리
import pandas as pd  # 표 데이터를 다루는 라이브러리
from sklearn.linear_model import LinearRegression, SGDRegressor  # OLS와 SGD 회귀 모델
from sklearn.preprocessing import StandardScaler  # 입력값을 표준화하는 도구


# 1. 데이터 준비
base_dir = Path(__file__).resolve().parent  # 현재 파일이 있는 폴더 경로를 구함
candidate_paths = [  # Salary_Data.csv가 있을 수 있는 후보 경로 목록
    base_dir / "Salary_Data.csv",  # 현재 폴더의 CSV 파일
    base_dir.parent / "My_project_0325" / "Salary_Data.csv",  # 이전 실습 폴더의 CSV 파일
    Path("/Users/jeong-yongcheol/Desktop/00_CBNU_AI/2_(대학원, 본교 E10-401)머신러닝 프로그래밍 (8884005-01) , 채민선교수/2. 수업 자료/Salary_Data.csv"),  # 수업자료 원본 경로
]
file_path = next((path for path in candidate_paths if path.exists()), None)  # 실제로 존재하는 경로 하나를 선택
if file_path is None:  # 데이터 파일을 못 찾으면
    raise FileNotFoundError("Salary_Data.csv 파일을 찾을 수 없습니다.")  # 에러를 발생시킴

df = pd.read_csv(file_path)  # CSV 파일을 읽어서 데이터프레임으로 저장
X, y = df[["YearsExperience"]], df["Salary"]  # 입력은 연차, 출력은 연봉으로 분리


# 2. OLS (수학적 정답)
ols = LinearRegression().fit(X, y)  # 최소자승법으로 가장 잘 맞는 직선을 한 번에 계산


# 3. SGD 학습 준비 (표준화 필수)
scaler = StandardScaler()  # 표준화 도구 생성
X_s = scaler.fit_transform(X)  # 연차 값을 표준화해 SGD가 학습하기 쉽게 만듦

# [중간값] 약 1번만 업데이트한 모델
sgd_short = SGDRegressor(  # 아주 적게 학습해서 아직 불안정한 SGD 모델
    max_iter=1, tol=None, eta0=0.01, random_state=42
).fit(X_s, y)

# [오른쪽] 충분히 학습한 모델
sgd_long = SGDRegressor(  # 충분히 반복 학습해서 수렴에 가까운 SGD 모델
    max_iter=1000, tol=1e-3, eta0=0.01, random_state=42
).fit(X_s, y)


# 4. 그래프 출력
plt.figure(figsize=(18, 6))  # 가로로 3개 그래프를 보기 좋게 배치할 큰 그림 영역 생성

# [왼쪽] OLS: 수학적 정답
plt.subplot(1, 3, 1)  # 첫 번째 칸 선택
plt.scatter(X, y, color="orange", alpha=0.5)  # 실제 데이터를 점으로 그림
plt.plot(X, ols.predict(X), color="blue", linewidth=3)  # OLS가 구한 정답 직선을 그림
title_ols = f"OLS: Exact Solution\nw: {ols.coef_[0]:.2f}, b: {ols.intercept_:.2f}"  # 제목에 기울기와 절편 포함
plt.title(title_ols, fontsize=12, fontweight="bold")  # 왼쪽 그래프 제목 표시

# [중앙] SGD (1회 학습): 방황하는 단계
plt.subplot(1, 3, 2)  # 두 번째 칸 선택
plt.scatter(X_s, y, color="orange", alpha=0.5)  # 표준화된 입력 기준으로 데이터 점 그림
plt.plot(  # 짧게 학습한 SGD가 만든 직선을 그림
    X_s,
    sgd_short.predict(X_s),
    color="red",
    linestyle="--",
    label="1 Iteration",
)
title_short = (  # 제목 문자열을 만들어 현재 가중치 정보를 함께 보여줌
    f"SGD: 1 Iteration\nw: {sgd_short.coef_[0]:.2f}, "
    f"b: {sgd_short.intercept_[0]:.2f}"
)
plt.title(title_short, fontsize=12, color="red")  # 두 번째 그래프 제목 표시
plt.legend()  # 범례 표시

# [오른쪽] SGD (완료): 수렴한 단계
plt.subplot(1, 3, 3)  # 세 번째 칸 선택
plt.scatter(X_s, y, color="orange", alpha=0.5)  # 표준화된 입력 기준 실제 데이터 표시
plt.plot(  # 충분히 학습한 SGD가 만든 직선을 그림
    X_s,
    sgd_long.predict(X_s),
    color="green",
    linewidth=3,
    label="1000 Iterations",
)
title_long = (  # 마지막 제목 문자열 생성
    f"SGD: Final Convergence\nw: {sgd_long.coef_[0]:.2f}, "
    f"b: {sgd_long.intercept_[0]:.2f}"
)
plt.title(title_long, fontsize=12, color="green", fontweight="bold")  # 세 번째 그래프 제목 표시
plt.legend()  # 범례 표시

plt.tight_layout()  # 그래프 간격 자동 조정
plt.show()  # 화면에 그래프 출력
