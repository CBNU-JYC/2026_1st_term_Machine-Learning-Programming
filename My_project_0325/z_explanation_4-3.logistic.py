import numpy as np  # 숫자 계산을 쉽게 하기 위해 numpy를 np라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt  # 그래프를 그리기 위해 pyplot을 plt라는 이름으로 불러옵니다.
from sklearn import datasets  # sklearn 안에 있는 예제 데이터셋 모음을 불러옵니다.
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델 클래스를 불러옵니다.


# 1. 데이터 준비 (0, 1, 2 세 가지 숫자만 사용)
digits = datasets.load_digits()  # 손글씨 숫자 이미지 데이터셋을 불러옵니다.
X, y = digits.data, digits.target  # X에는 입력 이미지 데이터, y에는 정답 숫자 라벨을 저장합니다.
mask = np.isin(y, [0, 1, 2])  # y 값이 0, 1, 2 중 하나인지 검사해서 True/False 마스크를 만듭니다.
X_sub, y_sub = X[mask], y[mask]  # 마스크를 이용해 0, 1, 2에 해당하는 데이터만 따로 골라냅니다.


# --- [실습 1] 시그모이드 (Sigmoid): "이것은 0입니까?" ---
# 0이면 1(True), 1이나 2이면 0(False)으로
y_binary = (y_sub == 0).astype(int)  # 0인 경우만 1로, 나머지(1과 2)는 0으로 바꿔 이진 분류용 정답을 만듭니다.
model_sigmoid = LogisticRegression(max_iter=10000).fit(X_sub, y_binary)  # 이진 분류 문제로 로지스틱 회귀 모델을 학습시킵니다.
# 첫 번째 샘플에 대한 확률 ([0이 아닐 확률, 0일 확률])
prob_sigmoid = model_sigmoid.predict_proba(X_sub[:1])[0]  # 첫 번째 샘플이 각 클래스일 확률을 계산하고 첫 결과만 꺼냅니다.


# --- [실습 2] 소프트맥스 (Softmax): "0, 1, 2 중 누구입니까?" ---
model_softmax = LogisticRegression(max_iter=10000).fit(X_sub, y_sub)  # 이번에는 0, 1, 2를 그대로 사용해 다중분류 모델을 학습시킵니다.
# 첫 번째 샘플에 대한 확률 ([0일 확률, 1일 확률, 2일 확률])
prob_softmax = model_softmax.predict_proba(X_sub[:1])[0]  # 첫 번째 샘플이 0, 1, 2일 확률을 각각 계산합니다.


# 2. 결과 출력 및 시각화
print("=== 확률 결과 비교 ===")  # 출력 구간 제목을 보여줍니다.
print(f"Sigmoid Output (Is it 0?): {prob_sigmoid}")  # 이진 분류 모델의 확률 결과를 출력합니다.
print(f"Softmax Output (0, 1, or 2?): {prob_softmax}")  # 다중분류 모델의 확률 결과를 출력합니다.

plt.figure(figsize=(12, 5))  # 가로 12, 세로 5 크기의 그래프 창을 만듭니다.

# 왼쪽: 시그모이드 결과 (이진 분류 - 선택지가 2개)
plt.subplot(1, 2, 1)  # 1행 2열 그래프 중 첫 번째 칸을 선택합니다.
plt.bar(["Not 0", "Is 0"], prob_sigmoid, color=["lightgray", "royalblue"], alpha=0.8)  # 두 확률을 막대그래프로 그립니다.
plt.title("Sigmoid (Binary Classification)", fontsize=14)  # 왼쪽 그래프 제목을 설정합니다.
plt.ylabel("Probability")  # y축 이름을 Probability로 설정합니다.
plt.ylim(0, 1.1)  # y축 범위를 0~1.1로 잡아 막대 위 숫자가 잘 보이게 합니다.
for i, v in enumerate(prob_sigmoid):  # 각 막대의 위치(i)와 확률값(v)을 하나씩 꺼냅니다.
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")  # 각 막대 위에 확률값을 소수 둘째 자리까지 적어 넣습니다.


plt.subplot(1, 2, 2)  # 1행 2열 그래프 중 두 번째 칸을 선택합니다.
plt.bar(["Digit 0", "Digit 1", "Digit 2"], prob_softmax, color=["tomato", "mediumseagreen", "orange"], alpha=0.8)  # 세 클래스 확률을 막대그래프로 그립니다.
plt.title("Softmax (Multiclass Classification)", fontsize=14)  # 오른쪽 그래프 제목을 설정합니다.
plt.ylabel("Probability")  # y축 이름을 Probability로 설정합니다.
plt.ylim(0, 1.1)  # y축 범위를 0~1.1로 잡아 막대 위 숫자가 잘 보이게 합니다.
for i, v in enumerate(prob_softmax):  # 각 클래스의 위치(i)와 확률값(v)을 하나씩 꺼냅니다.
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")  # 각 막대 위에 확률값을 소수 둘째 자리까지 적어 넣습니다.

plt.tight_layout()  # 제목, 축, 그래프 사이 간격이 겹치지 않도록 자동 정리합니다.
plt.show()  # 완성된 그래프를 화면에 표시합니다.
