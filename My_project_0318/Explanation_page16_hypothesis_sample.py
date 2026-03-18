import numpy as np  # 숫자 배열과 수학 계산을 쉽게 하기 위한 라이브러리입니다.


# [1단계] 이전에 만든 시그모이드 함수
def sigmoid(z):  # z라는 값을 입력받아 0과 1 사이의 값으로 바꾸는 함수를 정의합니다.
    return 1 / (1 + np.exp(-z))  # 시그모이드 공식입니다. 입력이 클수록 1에 가까워지고, 작을수록 0에 가까워집니다.


# [2단계] 가설 함수 (Hypothesis Function) 정의
# 공식: h(x) = sigmoid(theta^T * x)
def hypothesis_function(x, theta):  # 입력 데이터 x와 가중치 theta를 받아 최종 확률을 계산하는 함수를 만듭니다.
    # np.dot은 행렬의 곱(내적)을 계산해주는 함수입니다.
    # z = theta[0]*x[0] + theta[1]*x[1] ... 와 같은 계산을 한 줄로 끝냅니다.
    z = np.dot(x, theta)  # 입력값과 가중치를 각각 곱해서 더한 종합 점수 z를 계산합니다.
    return sigmoid(z)  # 계산한 z를 시그모이드 함수에 넣어 최종 확률값으로 바꿉니다.


# [3단계] 실제 데이터로 테스트
# 예: 온도(x1)=30도, 진동(x2)=0.8 일 때
x_data = np.array([30, 0.8])  # 실제 입력 데이터입니다. 여기서는 온도와 진동값 2개를 넣었습니다.
theta_data = np.array([0.1, 5.0])  # 각 입력값에 곱해질 가중치입니다. 모델이 중요도를 반영할 때 사용합니다.

# 최종 확률 계산
prob = hypothesis_function(x_data, theta_data)  # 실제 데이터와 가중치를 넣어 최종 확률을 계산합니다.

print(f"종합 점수(z): {np.dot(x_data, theta_data)}")  # 시그모이드에 들어가기 전의 종합 점수 z를 화면에 출력합니다.
print(f"최종 고장 확률: {prob * 100:.2f}%")  # 최종 확률을 백분율(%) 형태로 보기 좋게 출력합니다.
