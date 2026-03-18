import numpy as np


# [1단계] 시그모이드 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# [2단계] 가설 함수 (Hypothesis Function)
def hypothesis_function(x, theta):
    z = np.dot(x, theta)
    return sigmoid(z)


# [3단계] 비용 함수 (Cost Function / Log Loss)
def compute_cost(x, y, theta):
    m = y.shape[0]  # 데이터의 개수
    h = hypothesis_function(x, theta)  # 모델의 예측 확률값

    # 로그 손실 공식 적용
    term1 = y.T.dot(np.log(h))
    term2 = (1 - y).T.dot(np.log(1 - h))

    j = (-1.0 / m) * (term1 + term2)
    return j


# [4단계] 실제 데이터 대입 및 결과 출력
x = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])

# 실제 정답 (1: 고장, 0: 정상, 1: 고장)
y = np.array([1, 0, 1])

# 임의로 설정한 가중치
theta = np.array([0.1, 0.2])

# 1. 먼저 예측 확률 확인하기
predictions = hypothesis_function(x, theta)
print("=== 1. 모델의 예측 확률 ===")
print(predictions)
print()

# 2. 이 예측이 얼마나 틀렸는지 비용 계산하기
cost = compute_cost(x, y, theta)
print("=== 2. 현재 모델의 비용(Cost) ===")
print(f"변형 점수: {cost:.4f}")
