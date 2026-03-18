import numpy as np


# [1단계] 기본 함수 생성 (시그모이드, 가설, 비용)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hypothesis_function(x, theta):
    z = np.dot(x, theta)
    return sigmoid(z)


def compute_cost(x, y, theta):
    m = y.shape[0]
    h = hypothesis_function(x, theta)
    # 안전한 로그 계산을 위해 아주 작은 값(1e-5)을 더해둡니다.
    term1 = y.T.dot(np.log(h + 1e-5))
    term2 = (1 - y).T.dot(np.log(1 - h + 1e-5))
    return (-1.0 / m) * (term1 + term2)


# [2단계] 경사하강법 함수 (가중치 업데이트)
def minimize_gradient(x, y, theta, iterations=1000, alpha=0.01):
    m = y.size
    cost_history = []

    for _ in range(iterations):
        # 예측값과 실제값의 차이(에러) 계산
        h = hypothesis_function(x, theta)
        loss = h - y

        # 기울기(Gradient) 계산
        gradient = x.T.dot(loss) / m

        # 가중치 업데이트
        theta = theta - (alpha * gradient)

        # 100번마다 현재 비용 기록
        if (_ % 100) == 0:
            current_cost = compute_cost(x, y, theta)
            cost_history.append(current_cost)
            print(f"[반복 횟수 {_:^4}] : 현재 비용(Cost) = {current_cost:.5f}")

    return theta, cost_history


# [3단계] 실제 데이터 대입 및 결과 확인
x_test = np.array([[1, 2], [3, 4], [5, 6]])
y_test = np.array([1, 0, 1])
initial_theta = np.array([0.0, 0.0])  # 초기 가중치

print("학습을 시작합니다...")  # 학습 실행
final_theta, history = minimize_gradient(
    x_test, y_test, initial_theta, iterations=1000, alpha=0.01
)
print("학습 완료!")
print(f"최종 가중치(theta): {final_theta}")
print(f"최종 비용(Cost): {compute_cost(x_test, y_test, final_theta):.5f}")
