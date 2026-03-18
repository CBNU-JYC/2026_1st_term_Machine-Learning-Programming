from sklearn.metrics import confusion_matrix


# [1단계] 실험할 데이터 만들기
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1]


# [2단계] 혼동 행렬(Confusion Matrix) 계산 및 출력
cm = confusion_matrix(y_true, y_pred)

print("1. 혼동 행렬 전체 모습")
print(cm)
print()


# [3단계] ravel() 함수로 2차원 행렬을 1차원으로 펼쳐 각 값을 변수에 담기
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("2. 세부 지표 값 확인")
print(f"TN (True Negative) : {tn}")
print(f"FP (False Positive): {fp}")
print(f"FN (False Negative): {fn}")
print(f"TP (True Positive) : {tp}")
