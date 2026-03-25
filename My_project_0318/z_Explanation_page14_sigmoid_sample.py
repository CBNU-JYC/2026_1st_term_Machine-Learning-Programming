import numpy as np  # 수치 계산을 쉽게 해주는 라이브러리입니다. 배열, exp 같은 수학 함수를 제공합니다.
from pathlib import Path  # 현재 파일 기준으로 저장 경로를 만들 때 사용합니다
import matplotlib.pyplot as plt  # 그래프를 그릴 때 사용하는 시각화 라이브러리입니다.


# [1단계] 시그모이드 함수 정의 만들기
def sigmoid(z):  # z라는 입력값을 받아서 0~1 사이 값으로 바꾸는 함수를 정의합니다.
    return 1 / (1 + np.exp(-z))  # 시그모이드 공식입니다. 큰 양수는 1에, 큰 음수는 0에 가까워집니다.


print("[PART 1] 시그모이드 함수 수치 테스트")  # 지금부터 간단한 값 테스트를 시작한다는 안내 문장입니다.

# 1. 입력값이 0일 때 (정확히 절반의 확률)
s_0 = sigmoid(0)  # 입력값 0을 시그모이드 함수에 넣어 결과를 계산합니다.
print(f"입력값 0 => 압축된 확률: {s_0:.5f} (50%)")  # 계산 결과를 소수점 5자리까지 보기 좋게 출력합니다.

# 2. 아주 큰 양수일 때 (100% 확신에 수렴)
s_100 = sigmoid(1000)  # 아주 큰 양수를 넣으면 결과가 1에 가까워지는지 확인합니다.
print(f"입력값 100 => 압축된 확률: {s_100:.5f} (약 100%)")  # 양수 입력의 결과를 출력합니다.

# 3. 아주 작은 음수일 때 (0% 확신에 수렴)
s_m100 = sigmoid(-10000)  # 아주 작은 음수를 넣으면 결과가 0에 가까워지는지 확인합니다.
print(f"입력값 -100 => 압축된 확률: {s_m100:.5f} (약 0%)")  # 음수 입력의 결과를 출력합니다.


print("[PART 2] matplotlib으로 시그모이드 곡선 시각화하기")  # 이제 수치 결과를 그래프로 표현하는 단계라는 뜻입니다.

# 1. 그래프를 그릴 x축 데이터 만들기
z_values = np.linspace(-10, 10, num=200)  # -10부터 10까지를 200칸으로 나눠 x축 값들을 만듭니다.

# 2. 만든 x축 데이터로 시그모이드 y값 계산
probabilities = sigmoid(z_values)  # 방금 만든 모든 x값에 시그모이드 함수를 적용해 y값들을 계산합니다.

# 3. 그래프 크기 설정
plt.figure(figsize=(10, 6))  # 가로 10, 세로 6 크기의 새 그래프 창을 준비합니다.

# 4. 시그모이드 S자 곡선 그리기
plt.plot(z_values, probabilities, color="red", linewidth=3, label="Sigmoid Curve")  # x값과 y값을 이용해 빨간색 곡선을 그립니다.

# y축 기준선 (상한선 1.0, 하한선 0.0)
plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)  # y=1 위치에 회색 점선을 그어 위쪽 기준을 보여줍니다.
plt.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)  # y=0 위치에 회색 점선을 그어 아래쪽 기준을 보여줍니다.

# 50% 기준선 그리기
plt.axhline(y=0.5, color="black", linestyle=":", label="Threshold (0.5)")  # 확률 0.5를 나타내는 기준선을 검은 점선으로 그립니다.

# x축 기준선 (원점값 0)
plt.axvline(x=0.0, color="gray", linestyle="--", alpha=0.5)  # x=0 위치에 세로 기준선을 그립니다.

# 예시 포인트 표시
plt.scatter(x=0, y=0.5, color="blue", s=100, zorder=5, label="sigmoid(0) = 0.5")  # (0, 0.5) 지점을 파란 점으로 강조합니다.

# 5. 그래프 꾸미기
plt.title("Sigmoid Function (Magic Compressor)")  # 그래프의 제목을 설정합니다.
plt.xlabel("Raw Score (z) - from AI model")  # x축 이름을 설정합니다.
plt.ylabel("Probability (0.0 ~ 1.0) - output")  # y축 이름을 설정합니다.
plt.grid(True, linestyle="--", alpha=0.3)  # 배경 격자선을 넣어 그래프를 읽기 쉽게 만듭니다.
plt.legend(loc="upper left")  # label로 등록한 항목들의 설명 상자를 왼쪽 위에 표시합니다.

# 6. 그래프 파일로 저장하기
output_path = Path(__file__).with_name("page14_sigmoid_sample.png")  # 현재 파이썬 파일과 같은 폴더에 저장될 png 경로를 만듭니다.
plt.savefig(output_path, dpi=150, bbox_inches="tight")  # 그래프를 이미지 파일로 저장합니다. dpi는 해상도, bbox_inches는 여백 정리 옵션입니다.
print(f"그래프 저장 완료: {output_path}")  # 저장된 파일 위치를 화면에 알려줍니다.

# 7. 완성된 그래프 보여주기
plt.show()  # 그래프 창을 화면에 띄웁니다. 실행 환경에 따라 창 대신 저장만 될 수도 있습니다.
