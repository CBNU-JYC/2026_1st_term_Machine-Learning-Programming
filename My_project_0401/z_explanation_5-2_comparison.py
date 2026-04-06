from pathlib import Path  # 파일 경로를 안전하게 다루기 위한 도구

import matplotlib.pyplot as plt  # 그래프를 그릴 때 사용하는 라이브러리
import pandas as pd  # CSV 같은 표 데이터를 읽고 다룰 때 사용
from sklearn.linear_model import SGDRegressor  # 경사하강법 기반 회귀 모델
from sklearn.preprocessing import StandardScaler  # 입력값 크기를 비슷하게 맞춰주는 도구


# 1. 데이터 로드 및 표준화
base_dir = Path(__file__).resolve().parent  # 현재 파이썬 파일이 있는 폴더 위치를 구함
candidate_paths = [  # Salary_Data.csv가 있을 만한 후보 경로를 준비
    base_dir / "Salary_Data.csv",  # 현재 폴더 안의 데이터 파일 경로
    base_dir.parent / "My_project_0325" / "Salary_Data.csv",  # 이전 실습 폴더의 데이터 파일 경로
    Path("/Users/jeong-yongcheol/Desktop/00_CBNU_AI/2_(대학원, 본교 E10-401)머신러닝 프로그래밍 (8884005-01) , 채민선교수/2. 수업 자료/Salary_Data.csv"),  # 수업자료 원본 경로
]
file_path = next((path for path in candidate_paths if path.exists()), None)  # 실제로 존재하는 첫 번째 경로를 선택
if file_path is None:  # 어떤 경로에도 파일이 없으면
    raise FileNotFoundError("Salary_Data.csv 파일을 찾을 수 없습니다.")  # 에러를 발생시켜 알려줌

df = pd.read_csv(file_path)  # CSV 파일을 읽어서 표 형태 데이터로 저장
X, y = df[["YearsExperience"]], df["Salary"]  # 입력값은 연차, 정답값은 연봉으로 설정
scaler = StandardScaler()  # 표준화 도구를 준비
X_s = scaler.fit_transform(X)  # 연차 데이터를 평균 0, 표준편차 1 형태로 변환


# 2. 3가지 경사하강법 설정
# (1) Stochastic: 1개씩 업데이트
sgd_stochastic = SGDRegressor(  # 한 번에 데이터 1개 수준만 보고 빠르게 갱신하는 모델
    max_iter=1, learning_rate="constant", eta0=0.01, random_state=42
)

# (2) Mini-batch: 5개씩 묶어서 업데이트
sgd_minibatch = SGDRegressor(  # 여러 개를 조금씩 묶어서 중간 방식으로 학습하는 모델
    learning_rate="constant", eta0=0.01, random_state=42
)

# (3) Batch: 30개 전체를 보고 업데이트
sgd_batch = SGDRegressor(  # 전체 데이터를 충분히 여러 번 보며 안정적으로 학습하는 모델
    max_iter=1000,
    tol=1e-3,
    learning_rate="constant",
    eta0=0.01,
    random_state=42,
)


# 3. 시각화 비교
plt.figure(figsize=(18, 5))  # 가로로 3개 그래프를 놓기 위해 넓은 그림 영역을 만듦

# [그래프 1] Stochastic GD
plt.subplot(1, 3, 1)  # 1행 3열 중 첫 번째 칸에 그림을 그림
sgd_stochastic.partial_fit(X_s[:1], y[:1].values.ravel())  # 첫 샘플 1개만 사용해 한 번 학습
plt.scatter(X_s, y, color="orange", alpha=0.3)  # 실제 데이터 점들을 배경처럼 표시
plt.plot(  # 현재 모델이 만든 회귀선을 그림
    X_s,
    sgd_stochastic.predict(X_s),
    color="red",
    label="Stochastic (1 sample)",
)
plt.title("1. Stochastic GD", fontsize=13, color="red")  # 첫 번째 그래프 제목
plt.legend()  # 범례 표시

# [그래프 2] Mini-batch GD
plt.subplot(1, 3, 2)  # 1행 3열 중 두 번째 칸에 그림을 그림
sgd_minibatch.partial_fit(X_s[:5], y[:5].values.ravel())  # 처음 5개 샘플만 사용해 학습
plt.scatter(X_s, y, color="orange", alpha=0.3)  # 실제 데이터 점 표시
plt.plot(  # 미니배치 방식으로 얻은 회귀선을 그림
    X_s,
    sgd_minibatch.predict(X_s),
    color="blue",
    label="Mini-batch (5 samples)",
)
plt.title("2. Mini-batch GD", fontsize=13, color="blue")  # 두 번째 그래프 제목
plt.legend()  # 범례 표시

# [그래프 3] Full-Batch GD
plt.subplot(1, 3, 3)  # 1행 3열 중 세 번째 칸에 그림을 그림
sgd_batch.fit(X_s, y)  # 전체 데이터를 사용해 모델을 충분히 학습
plt.scatter(X_s, y, color="orange", alpha=0.3)  # 실제 데이터 점 표시
plt.plot(  # 전체 배치 방식으로 얻은 회귀선을 그림
    X_s,
    sgd_batch.predict(X_s),
    color="green",
    linewidth=3,
    label="Batch (ALL 30 samples)",
)
plt.title("3. Full-Batch GD", fontsize=13, color="green")  # 세 번째 그래프 제목
plt.legend()  # 범례 표시

plt.tight_layout()  # 그래프들이 겹치지 않게 간격 자동 조정
plt.show()  # 화면에 그래프를 출력
