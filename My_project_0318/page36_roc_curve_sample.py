import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


# [1단계] 실험할 데이터 만들기
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])


# [2단계] roc_curve 함수를 사용해 여러 임곗값에서의 FPR, TPR 계산하기
fpr, tpr, thresholds = roc_curve(y_true, y_scores)


# [3단계] 계산된 결과 확인하기
print("=== ROC 곡선 계산 결과 ===")
print(f"기준점(Thresholds) : {thresholds}")
print(f"X축 = FPR (오경보 비율) : {fpr}")
print(f"Y축 = TPR (민감도)     : {tpr}")
print()


# [4단계] matplotlib를 이용해 ROC 곡선 그리기
plt.figure(figsize=(6, 6))  # 그래프의 가로세로 크기를 지정합니다.

# 1. 우리의 AI 모델 성능 선 그리기
plt.plot(fpr, tpr, marker="o", color="blue", linewidth=2, label="Our AI Model")

# 2. 랜덤 분류기의 기준선 그리기
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (50%)")

# 3. 그래프 꾸미기
plt.title("ROC Curve Analysis")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.grid(True)
plt.legend()

# 4. 완성된 그래프 창 띄우기
plt.show()
