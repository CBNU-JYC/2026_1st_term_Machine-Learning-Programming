import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


# [1단계] 시그모이드 함수 정의 만들기
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print("[PART 1] 시그모이드 함수 수치 테스트")

# 1. 입력값이 0일 때 (정확히 절반의 확률)
s_0 = sigmoid(0)
print(f"입력값 0 => 압축된 확률: {s_0:.5f} (50%)")

# 2. 아주 큰 양수일 때 (100% 확신에 수렴)
s_100 = sigmoid(10000)
print(f"입력값 100 => 압축된 확률: {s_100:.5f} (약 100%)")

# 3. 아주 작은 음수일 때 (0% 확신에 수렴)
s_m100 = sigmoid(-10000)
print(f"입력값 -100 => 압축된 확률: {s_m100:.5f} (약 0%)")


print("[PART 2] matplotlib으로 시그모이드 곡선 시각화하기")

# 1. 그래프를 그릴 x축 데이터 만들기
z_values = np.linspace(-10, 10, num=200)

# 2. 만든 x축 데이터로 시그모이드 y값 계산
probabilities = sigmoid(z_values)

# 3. 그래프 크기 설정
plt.figure(figsize=(10, 6))

# 4. 시그모이드 S자 곡선 그리기
plt.plot(z_values, probabilities, color="red", linewidth=3, label="Sigmoid Curve")

# y축 기준선 (상한선 1.0, 하한선 0.0)
plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
plt.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)

# 50% 기준선 그리기
plt.axhline(y=0.5, color="black", linestyle=":", label="Threshold (0.5)")

# x축 기준선 (원점값 0)
plt.axvline(x=0.0, color="gray", linestyle="--", alpha=0.5)

# 예시 포인트 표시
plt.scatter(x=0, y=0.5, color="blue", s=100, zorder=5, label="sigmoid(0) = 0.5")

# 5. 그래프 꾸미기
plt.title("Sigmoid Function (Magic Compressor)")
plt.xlabel("Raw Score (z) - from AI model")
plt.ylabel("Probability (0.0 ~ 1.0) - output")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="upper left")

# 6. 그래프 파일로 저장하기
output_path = Path(__file__).with_name("page14_sigmoid_sample.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"그래프 저장 완료: {output_path}")

# 7. 완성된 그래프 보여주기
plt.show()
