import numpy as np  # 숫자 계산을 쉽게 하기 위해 numpy를 np라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt  # 그래프를 그리기 위해 pyplot을 plt라는 이름으로 불러옵니다.


def softmax(values):  # 여러 점수를 받아서 확률처럼 보이는 값으로 바꾸는 함수를 정의합니다.
    array_values = np.exp(values)  # 각 점수에 e^x를 적용해서 큰 값은 더 크게 강조합니다.
    return array_values / np.sum(array_values)  # 전체 합으로 나누어 결과들의 총합이 1이 되게 만듭니다.


# --- 입력 데이터 ---
values = [-2, -1, -5, 0.5]  # 예제로 사용할 원래 점수 4개를 리스트로 저장합니다.
labels = [f"Class {i}\n({v})" for i, v in enumerate(values)]  # 각 막대 아래에 표시할 이름표를 만듭니다.

# 소프트맥스 계산
y = softmax(values)  # 입력 점수들을 softmax에 넣어서 각 클래스의 확률값을 계산합니다.

# --- 그래프 그리기 ---
plt.figure(figsize=(10, 6))  # 가로 10, 세로 6 크기의 그래프 창을 만듭니다.

# 막대 그래프 생성
bars = plt.bar(labels, y, color=["#4f46e5", "#818cf8", "#c7d2fe", "#f43f5e"])  # 각 클래스의 확률을 막대로 그립니다.

# 그래프 위에 확률 값 표시 (%)
for bar in bars:  # 만들어진 막대를 하나씩 꺼내면서 반복합니다.
    height = bar.get_height()  # 현재 막대의 높이, 즉 그 클래스의 확률값을 가져옵니다.
    plt.text(  # 그래프 위 특정 위치에 글자를 써서 확률값을 표시합니다.
        bar.get_x() + bar.get_width() / 2.0,  # 막대의 가로 중앙 위치를 계산합니다.
        height + 0.01,  # 막대보다 아주 조금 위쪽에 글자가 보이도록 세로 위치를 잡습니다.
        f"{height * 100:.1f}%",  # 확률을 퍼센트로 바꾸어 소수 첫째 자리까지 표시합니다.
        ha="center",  # 글자를 가로 방향으로 가운데 정렬합니다.
        va="bottom",  # 글자의 아래쪽을 기준으로 위치를 맞춥니다.
        fontweight="bold",  # 글씨를 굵게 표시합니다.
    )

# 그래프 꾸미기
plt.title("Softmax Output: Probability Distribution", fontsize=15, pad=20)  # 그래프 제목을 설정합니다.
plt.xlabel("Classes (Input Score)", fontsize=12)  # x축 이름을 설정합니다.
plt.ylabel("Probability", fontsize=12)  # y축 이름을 설정합니다.
plt.ylim(0, 1.1)  # y축 범위를 0부터 1.1까지로 정해서 위쪽 글자가 잘 보이게 합니다.
plt.grid(axis="y", linestyle="--", alpha=0.7)  # y축 방향의 점선 격자를 추가합니다.
plt.show()  # 완성된 그래프를 화면에 표시합니다.

# 텍스트 결과 출력
print("입력 점수:", values)  # 원래 입력 점수를 터미널에 출력합니다.
print("확률 분포:", [f"{prob * 100:.2f}%" for prob in y])  # 계산된 확률들을 퍼센트 문자열로 바꿔 출력합니다.
