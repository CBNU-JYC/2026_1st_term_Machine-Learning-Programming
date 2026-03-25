import matplotlib.pyplot as plt  # 그래프를 그릴 때 사용하는 라이브러리입니다.
import numpy as np  # 숫자 배열과 수학 계산을 쉽게 해주는 라이브러리입니다.
from sklearn.metrics import roc_curve  # ROC 곡선 계산에 필요한 함수를 불러옵니다.


# [1단계] 실험할 데이터 만들기
y_true = np.array([0, 0, 1, 1])  # 실제 정답입니다. 0은 음성, 1은 양성이라고 생각하면 됩니다.
y_scores = np.array([0.1, 0.4, 0.35, 0.8])  # 모델이 예측한 점수입니다. 값이 클수록 1일 가능성이 높다고 본 것입니다.


# [2단계] roc_curve 함수를 사용해 여러 임곗값에서의 FPR, TPR 계산하기
fpr, tpr, thresholds = roc_curve(y_true, y_scores)  # 기준점을 여러 개 바꿔가며 FPR, TPR, threshold 값을 한 번에 계산합니다.


# [3단계] 계산된 결과 확인하기
print("=== ROC 곡선 계산 결과 ===")  # 출력 결과의 제목을 보여줍니다.
print(f"기준점(Thresholds) : {thresholds}")  # 어떤 기준점들로 계산했는지 출력합니다.
print(f"X축 = FPR (오경보 비율) : {fpr}")  # FPR은 실제 0인데 1이라고 잘못 판단한 비율입니다.
print(f"Y축 = TPR (민감도)     : {tpr}")  # TPR은 실제 1인 데이터를 얼마나 잘 찾아냈는지 보여주는 비율입니다.
print()  # 보기 좋게 한 줄 띄웁니다.


# [4단계] matplotlib를 이용해 ROC 곡선 그리기
plt.figure(figsize=(6, 6))  # 가로 6, 세로 6 크기의 그래프 창을 만듭니다.

# 1. 우리의 AI 모델 성능 선 그리기
plt.plot(fpr, tpr, marker="o", color="blue", linewidth=2, label="Our AI Model")  # 계산된 FPR, TPR 값을 이어서 ROC 곡선을 그립니다.

# 2. 랜덤 분류기의 기준선 그리기
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (50%)")  # 무작위로 예측하는 모델의 기준선을 그립니다.

# 3. 그래프 꾸미기
plt.title("ROC Curve Analysis")  # 그래프 제목을 설정합니다.
plt.xlabel("False Positive Rate (FPR)")  # x축 이름을 설정합니다.
plt.ylabel("True Positive Rate (TPR)")  # y축 이름을 설정합니다.
plt.grid(True)  # 그래프의 격자선을 켭니다.
plt.legend()  # 그래프에 표시된 선들의 설명 상자를 보여줍니다.

# 4. 완성된 그래프 창 띄우기
plt.show()  # 최종 그래프를 화면에 표시합니다. 환경에 따라 창 대신 저장만 할 수도 있습니다.
