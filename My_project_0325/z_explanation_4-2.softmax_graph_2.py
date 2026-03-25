import numpy as np  # 숫자 계산을 쉽게 하기 위해 numpy를 np라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt  # 그래프를 그리기 위해 pyplot을 plt라는 이름으로 불러옵니다.


def softmax(values):  # 여러 점수를 입력받아 확률처럼 보이는 값으로 바꾸는 함수를 만듭니다.
    array_values = np.exp(values)  # 각 점수에 e^x를 적용해서 큰 값이 더 크게 강조되도록 만듭니다.
    return array_values / np.sum(array_values)  # 전체 합으로 나누어 결과들의 총합이 1이 되게 만듭니다.


# --- 데이터 준비 ---
values = [-2, -1, -5, 0.5]  # 예제로 사용할 원래 점수 4개를 리스트로 저장합니다.
y = softmax(values)  # 위 점수들을 softmax 함수에 넣어서 각 클래스의 확률을 계산합니다.
labels = [f"Class {i} ({v})" for i, v in enumerate(values)]  # 범례에 표시할 클래스 이름과 원래 점수를 문자열로 만듭니다.
colors = ["#4f46e5", "#818cf8", "#c7d2fe", "#f43f5e"]  # 각 구간에 칠할 색상을 순서대로 준비합니다.


# --- 그래프 그리기 ---
fig, ax = plt.subplots(figsize=(12, 4))  # 하나의 그래프(fig)와 축(ax)을 만들고, 크기를 가로 12 세로 4로 정합니다.

# 단일 누적 가로 막대 그래프 생성
left_pos = 0  # 첫 번째 막대 조각이 시작할 가로 위치를 0으로 둡니다.
for i in range(len(y)):  # 확률값 y를 처음부터 끝까지 하나씩 순서대로 반복합니다.
    ax.barh(  # 가로 막대 그래프를 그리는 함수입니다.
        "Softmax Total (1.0)",  # y축에 표시될 막대의 이름입니다. 여기서는 전체 확률 1.0을 뜻합니다.
        y[i],  # 현재 클래스의 확률만큼 막대 길이를 그립니다.
        left=left_pos,  # 현재 막대 조각이 시작할 왼쪽 위치를 지정합니다. 누적 막대의 핵심입니다.
        color=colors[i],  # 현재 클래스에 대응되는 색상을 사용합니다.
        label=labels[i],  # 범례에는 현재 클래스 이름을 표시합니다.
        edgecolor="white",  # 각 조각 경계선을 흰색으로 그려 구분하기 쉽게 합니다.
        height=0.6,  # 막대의 두께를 정합니다.
    )

    # 막대 중앙에 확률 값(%) 표시
    if y[i] > 0.03:  # 너무 작은 조각은 글씨가 겹칠 수 있으므로 3%보다 큰 경우에만 글자를 씁니다.
        ax.text(  # 그래프 안의 특정 위치에 글자를 씁니다.
            left_pos + y[i] / 2,  # 현재 막대 조각의 중앙 x 위치를 계산합니다.
            0,  # 가로 막대가 놓인 세로 위치입니다.
            f"{y[i] * 100:.1f}%",  # 확률을 퍼센트로 바꾸어 소수 첫째 자리까지 표시합니다.
            va="center",  # 세로 방향으로 글자를 가운데 정렬합니다.
            ha="center",  # 가로 방향으로 글자를 가운데 정렬합니다.
            color="white" if i != 2 else "black",  # 배경색이 밝은 3번째 조각은 검정, 나머지는 흰색 글씨를 씁니다.
            fontweight="bold",  # 글씨를 굵게 표시합니다.
            fontsize=11,  # 글자 크기를 11로 설정합니다.
        )

    left_pos += y[i]  # 다음 막대 조각은 현재 조각 끝부터 시작해야 하므로 위치를 누적해서 더합니다.


# 그래프 꾸미기
ax.set_title("Softmax Output: 100% Stacked Visualization", fontsize=16, pad=20)  # 그래프 제목을 설정합니다.
ax.set_xlim(0, 1)  # x축 범위를 0부터 1까지로 고정해서 전체 확률 구간을 정확히 보여줍니다.
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])  # x축 눈금을 0, 0.25, 0.5, 0.75, 1.0 위치에 찍습니다.
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0 (Total)"])  # 각 눈금에 보일 글자를 지정합니다.
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)  # 범례를 그래프 아래 중앙에 가로 4칸으로 배치합니다.


# 불필요한 테두리 제거
ax.spines["top"].set_visible(False)  # 위쪽 테두리 선을 숨깁니다.
ax.spines["right"].set_visible(False)  # 오른쪽 테두리 선을 숨깁니다.
ax.spines["left"].set_visible(False)  # 왼쪽 테두리 선을 숨깁니다.

plt.tight_layout()  # 제목, 축, 범례가 겹치지 않도록 전체 여백을 자동으로 정리합니다.
plt.show()  # 완성된 그래프를 화면에 표시합니다.


# 검증용 출력
print("각 클래스 확률:", [f"{prob:.4f}" for prob in y])  # 각 클래스의 확률을 소수 넷째 자리까지 문자열로 출력합니다.
print("확률 총합:", np.sum(y))  # 모든 확률을 더한 값이 1인지 확인하기 위해 출력합니다.
