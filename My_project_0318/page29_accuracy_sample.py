import numpy as np
from sklearn.metrics import accuracy_score


# [1단계] 실험할 데이터 만들기
y_pred = np.array([0, 1, 1, 0])  # 모델이 예측한 값
y_true = np.array([0, 1, 0, 0])  # 실제 정답


# [2단계] 직접 정확도 계산하기
manual_accuracy = sum(y_true == y_pred) / len(y_true)

print("1. 직접 계산한 정확도")
print(f"맞춘 개수(3) / 전체 개수(4) = {manual_accuracy}")
print()


# [3단계] accuracy_score 함수로 계산하기
sklearn_accuracy = accuracy_score(y_true, y_pred)

print("2. 사이킷런 함수(accuracy_score)를 사용한 정확도")
print(f"결과: {sklearn_accuracy}")
