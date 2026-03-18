import numpy as np  # 숫자 배열과 수학 계산을 쉽게 하기 위한 라이브러리입니다.


# [1단계] 기본 함수 생성 (시그모이드, 가설, 비용)
def sigmoid(z):  # z라는 점수를 0과 1 사이 값으로 바꾸는 함수를 정의합니다.
    return 1 / (1 + np.exp(-z))  # 시그모이드 공식입니다. 점수를 확률처럼 해석할 수 있게 바꿔줍니다.


def hypothesis_function(x, theta):  # 입력 데이터 x와 가중치 theta를 사용해 예측값을 계산하는 함수입니다.
    z = np.dot(x, theta)  # 입력값과 가중치를 곱해서 더한 종합 점수 z를 계산합니다.
    return sigmoid(z)  # 종합 점수를 시그모이드 함수에 넣어 0~1 사이 예측값으로 바꿉니다.


def compute_cost(x, y, theta):  # 현재 가중치가 얼마나 잘못 예측하는지 비용으로 계산하는 함수입니다.
    m = y.shape[0]  # 전체 데이터 개수를 구합니다.
    h = hypothesis_function(x, theta)  # 현재 가중치로 계산한 예측값들입니다.
    # 안전한 로그 계산을 위해 아주 작은 값(1e-5)을 더해둡니다.
    term1 = y.T.dot(np.log(h + 1e-5))  # 실제 정답이 1인 경우에 대한 손실 부분입니다.
    term2 = (1 - y).T.dot(np.log(1 - h + 1e-5))  # 실제 정답이 0인 경우에 대한 손실 부분입니다.
    return (-1.0 / m) * (term1 + term2)  # 평균 손실을 계산해 최종 비용값으로 반환합니다.


# [2단계] 경사하강법 함수 (가중치 업데이트)
def minimize_gradient(x, y, theta, iterations=1000, alpha=0.01):  # 여러 번 반복하며 가중치를 조금씩 더 좋은 방향으로 바꾸는 함수입니다.
    m = y.size  # 데이터 개수를 저장합니다.
    cost_history = []  # 학습 중 비용 변화를 저장할 리스트입니다.

    for _ in range(iterations):  # 정해진 횟수만큼 반복 학습합니다.
        # 예측값과 실제값의 차이(에러) 계산
        h = hypothesis_function(x, theta)  # 현재 가중치로 예측값을 계산합니다.
        loss = h - y  # 예측값과 실제값의 차이를 계산합니다.

        # 기울기(Gradient) 계산
        gradient = x.T.dot(loss) / m  # 비용을 줄이기 위해 어느 방향으로 얼마나 움직여야 하는지 계산합니다.

        # 가중치 업데이트
        theta = theta - (alpha * gradient)  # 학습률 alpha만큼 기울기 반대 방향으로 가중치를 수정합니다.

        # 100번마다 현재 비용 기록
        if (_ % 100) == 0:  # 100번 반복할 때마다 중간 결과를 확인합니다.
            current_cost = compute_cost(x, y, theta)  # 현재 가중치의 비용을 계산합니다.
            cost_history.append(current_cost)  # 비용 기록 리스트에 저장합니다.
            print(f"[반복 횟수 {_:^4}] : 현재 비용(Cost) = {current_cost:.5f}")  # 현재 비용을 화면에 출력합니다.

    return theta, cost_history  # 최종 가중치와 비용 기록을 함수 밖으로 돌려줍니다.


# [3단계] 실제 데이터 대입 및 결과 확인
x_test = np.array([[1, 2], [3, 4], [5, 6]])  # 특징 2개를 가진 샘플 3개를 입력 데이터로 만듭니다.
y_test = np.array([1, 0, 1])  # 각 샘플에 대한 실제 정답입니다.
initial_theta = np.array([0.0, 0.0])  # 학습 시작 전의 초기 가중치입니다.

print("학습을 시작합니다...")  # 학습 시작 안내 문장입니다.
final_theta, history = minimize_gradient(  # 경사하강법을 실행해 최종 가중치와 비용 변화 기록을 받습니다.
    x_test, y_test, initial_theta, iterations=1000, alpha=0.01
)
print("학습 완료!")  # 학습 종료 안내 문장입니다.
print(f"최종 가중치(theta): {final_theta}")  # 학습 후 얻은 최종 가중치를 출력합니다.
print(f"최종 비용(Cost): {compute_cost(x_test, y_test, final_theta):.5f}")  # 최종 가중치의 비용값을 출력합니다.
