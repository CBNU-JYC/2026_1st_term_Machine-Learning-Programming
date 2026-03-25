import numpy as np
import matplotlib.pyplot as plt


def softmax(values):
    array_values = np.exp(values)
    return array_values / np.sum(array_values)


# --- 입력 데이터 ---
values = [-2, -1, -5, 0.5]
labels = [f"Class {i}\n({v})" for i, v in enumerate(values)]

# 소프트맥스 계산
y = softmax(values)

# --- 그래프 그리기 ---
plt.figure(figsize=(10, 6))

# 막대 그래프 생성
bars = plt.bar(labels, y, color=["#4f46e5", "#818cf8", "#c7d2fe", "#f43f5e"])

# 그래프 위에 확률 값 표시 (%)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height * 100:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# 그래프 꾸미기
plt.title("Softmax Output: Probability Distribution", fontsize=15, pad=20)
plt.xlabel("Classes (Input Score)", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 텍스트 결과 출력
print("입력 점수:", values)
print("확률 분포:", [f"{prob * 100:.2f}%" for prob in y])
