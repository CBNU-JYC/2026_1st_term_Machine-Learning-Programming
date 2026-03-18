from sklearn.metrics import confusion_matrix  # 혼동행렬을 계산해주는 함수를 불러옵니다.


# [1단계] 실험할 데이터 만들기
y_true = [1, 0, 1, 1, 0, 1]  # 실제 정답입니다. 1은 양성, 0은 음성이라고 생각하면 됩니다.
y_pred = [0, 0, 1, 1, 0, 1]  # 모델이 예측한 결과입니다.


# [2단계] 혼동 행렬(Confusion Matrix) 계산 및 출력
cm = confusion_matrix(y_true, y_pred)  # 실제값과 예측값을 비교해서 혼동행렬을 계산합니다.

print("1. 혼동 행렬 전체 모습")  # 첫 번째 출력 제목입니다.
print(cm)  # 2x2 형태의 혼동행렬 전체를 출력합니다.
print()  # 보기 좋게 한 줄 띄웁니다.


# [3단계] ravel() 함수로 2차원 행렬을 1차원으로 펼쳐 각 값을 변수에 담기
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # 혼동행렬의 4개 값을 차례대로 꺼내 각 변수에 저장합니다.

print("2. 세부 지표 값 확인")  # 두 번째 출력 제목입니다.
print(f"TN (True Negative) : {tn}")  # 실제 0이고 예측도 0인 개수입니다.
print(f"FP (False Positive): {fp}")  # 실제 0인데 예측을 1로 잘못한 개수입니다.
print(f"FN (False Negative): {fn}")  # 실제 1인데 예측을 0으로 잘못한 개수입니다.
print(f"TP (True Positive) : {tp}")  # 실제 1이고 예측도 1인 개수입니다.
