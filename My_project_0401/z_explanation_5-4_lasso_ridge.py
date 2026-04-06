import matplotlib.pyplot as plt  # 그래프를 그릴 때 사용하는 라이브러리
import numpy as np  # 숫자 계산과 배열 연산에 사용하는 라이브러리
import pandas as pd  # 표 형태 데이터 비교용 라이브러리
from sklearn.datasets import make_regression  # 회귀용 가상 데이터를 만드는 함수
from sklearn.linear_model import Lasso, Ridge  # 라쏘와 리지 회귀 모델


# 1. 데이터 생성 (변수 100개 중 진짜 중요한 건 10개뿐인 상황)
X, y = make_regression(  # 회귀 실험용 가짜 데이터를 자동 생성
    n_samples=50,  # 데이터 개수는 50개
    n_features=100,  # 입력 변수는 총 100개
    n_informative=10,  # 그중 실제로 중요한 변수는 10개만 존재
    noise=1,  # 약간의 잡음을 추가해 현실적인 데이터처럼 만듦
    random_state=42,  # 실행할 때마다 같은 데이터가 나오도록 고정
)

# 2. L1(Lasso) 과 L2(Ridge) 학습
lasso = Lasso(alpha=1.0).fit(X, y)  # L1 정규화를 적용한 라쏘 회귀 학습
ridge = Ridge(alpha=1.0).fit(X, y)  # L2 정규화를 적용한 리지 회귀 학습


# [파트 1] 결과 수치 출력 (콘솔)
print("*" * 60)  # 구분선 출력
print("=== [가중치 값 비교 (상위 10개)] ===")  # 비교 제목 출력
print("*" * 60)  # 구분선 출력
df_comp = pd.DataFrame(  # 두 모델의 가중치를 표 형태로 묶음
    {
        "L2 (Ridge) 가중치": ridge.coef_,  # 리지 회귀의 가중치들
        "L1 (Lasso) 가중치": lasso.coef_,  # 라쏘 회귀의 가중치들
    }
)
print(df_comp.head(10))  # 위에서 10개 행만 출력해 빠르게 비교
print("\n" + "*" * 60)  # 줄바꿈 후 구분선 출력
# 결정적 차이: 가중치가 '정확히 0'인 변수의 개수 세기
print(f"L2 (Ridge)가 0으로 만든 변수 개수: {np.sum(ridge.coef_ == 0)}개 / 100개")  # 리지가 완전히 제거한 변수 개수 출력
print(f"L1 (Lasso)가 0으로 만든 변수 개수: {np.sum(lasso.coef_ == 0)}개 / 100개")  # 라쏘가 완전히 제거한 변수 개수 출력
print("*" * 60)  # 마지막 구분선 출력


# [파트 2] 결과 시각화 (그래프)
plt.figure(figsize=(15, 6))  # 가로로 긴 그림 영역 생성

# 왼쪽: Ridge (가중치가 빠글빼글)
plt.subplot(1, 2, 1)  # 1행 2열 중 왼쪽 그래프 위치 선택
# 가중치 분포를 막대 그래프(stem plot)로 표현
plt.stem(ridge.coef_, markerfmt=" ", basefmt="k-")  # 리지 가중치의 분포를 줄기 그래프로 그림
plt.title(  # 왼쪽 그래프 제목 지정
    "L2 (Ridge) Coefficients\n(Many small non-zero values)",
    fontsize=14,
)
plt.xlabel("Feature Index (0~99)")  # x축 이름 설정
plt.ylabel("Coefficient Value")  # y축 이름 설정
plt.ylim(-100, 100)  # y축 범위를 고정해 오른쪽 그래프와 비교하기 쉽게 만듦
plt.grid(visible=True, alpha=0.3)  # 격자를 약하게 표시

# 오른쪽: Lasso (대부분이 0이라 텅 빈 느낌)
plt.subplot(1, 2, 2)  # 1행 2열 중 오른쪽 그래프 위치 선택
plt.stem(lasso.coef_, markerfmt=" ", basefmt="k-")  # 라쏘 가중치의 분포를 줄기 그래프로 그림
plt.title(  # 오른쪽 그래프 제목 지정
    "L1 (Lasso) Coefficients\n(Sparse: Most are exactly 0)",
    fontsize=14,
)
plt.xlabel("Feature Index (0~99)")  # x축 이름 설정
plt.ylabel("Coefficient Value")  # y축 이름 설정
plt.ylim(-100, 100)  # y축 범위를 왼쪽과 같게 맞춤
plt.grid(visible=True, alpha=0.3)  # 격자를 약하게 표시
plt.tight_layout()  # 그래프 간격을 자동으로 정리
plt.show()  # 최종 그래프를 화면에 출력
