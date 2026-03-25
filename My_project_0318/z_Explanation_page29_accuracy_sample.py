import numpy as np  # 숫자 배열을 쉽게 다루기 위한 라이브러리입니다.
from sklearn.metrics import accuracy_score  # 정확도(accuracy)를 계산해주는 함수를 불러옵니다.


# [1단계] 실험할 데이터 만들기
y_pred = np.array([0, 1, 1, 0])  # 모델이 예측한 값입니다.
y_true = np.array([0, 1, 0, 0])  # 실제 정답입니다.


# [2단계] 직접 정확도 계산하기
manual_accuracy = sum(y_true == y_pred) / len(y_true)  # 예측과 정답이 같은 개수를 세고, 전체 개수로 나누어 정확도를 구합니다.

print("1. 직접 계산한 정확도")  # 첫 번째 결과 제목을 출력합니다.
print(f"맞춘 개수(3) / 전체 개수(4) = {manual_accuracy}")  # 직접 계산한 정확도를 출력합니다.
print()  # 보기 좋게 한 줄 띄웁니다.


# [3단계] accuracy_score 함수로 계산하기
sklearn_accuracy = accuracy_score(y_true, y_pred)  # 사이킷런 함수를 사용해 정확도를 계산합니다.

print("2. 사이킷런 함수(accuracy_score)를 사용한 정확도")  # 두 번째 결과 제목을 출력합니다.
print(f"결과: {sklearn_accuracy}")  # 함수로 계산한 정확도를 출력합니다.
