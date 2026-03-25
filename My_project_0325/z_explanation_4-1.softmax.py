import numpy as np  # 숫자 계산을 쉽게 하기 위해 numpy를 np라는 이름으로 불러옵니다.


def softmax(values):  # softmax라는 함수를 만듭니다. 입력은 values라는 숫자 목록입니다.
    # np.exp(values)는 values 안의 각 숫자에 e^x를 적용합니다.
    # 예를 들어 값이 클수록 결과가 더 크게 강조됩니다.
    array_values = np.exp(values)

    # np.sum(array_values)는 위에서 만든 값들을 모두 더합니다.
    # 각 값을 전체 합으로 나누면 결과들의 총합이 1이 됩니다.
    # 그래서 이 결과를 "확률처럼" 해석할 수 있습니다.
    return array_values / np.sum(array_values)


values = [-2, -1, -5, 0.5]  # 예제로 사용할 입력값 4개를 리스트로 저장합니다.

# softmax(values)를 실행해서 각 값이 차지하는 상대적 비율을 구합니다.
y = softmax(values)

print("1. 입력값 (Values):", values)  # 원래 입력값이 무엇인지 먼저 출력합니다.
print("2. 소프트맥스 결과 (y):")  # 이제 softmax를 통과한 결과를 보여주겠다는 안내문입니다.

for i, prob in enumerate(y):  # y를 하나씩 꺼내면서 몇 번째 값인지(i)도 함께 확인합니다.
    # prob:.8f 는 소수점 아래 8자리까지 출력하라는 뜻입니다.
    # prob * 100 은 확률을 퍼센트(%)로 보기 쉽게 바꿔줍니다.
    print(f"   - 클래스 {i}의 확률: {prob:.8f} ({prob * 100:.2f}%)")


# y.sum()은 y 안의 모든 값을 더합니다.
# softmax 결과는 원래 합이 1이 되어야 하므로 그것이 맞는지 확인합니다.
total_sum = y.sum()

print("\n3. 결과값의 총합 (y.sum()):", total_sum)  # 결과값 전체를 더한 값이 1에 가까운지 출력합니다.


# np.argmax(y)는 y 안에서 가장 큰 값의 위치(인덱스)를 찾아줍니다.
# 즉, 가장 높은 확률을 가진 항목이 몇 번째인지 알려줍니다.
max_index = np.argmax(y)

# values[max_index]는 가장 높은 확률을 만든 원래 입력값이 무엇인지 보여줍니다.
print(f"\n가장 높은 확률을 가진 인덱스는 {max_index}번이며, 값은 {values[max_index]}입니다.")
