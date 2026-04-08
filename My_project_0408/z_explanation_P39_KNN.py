"""
이 프로그램은 붓꽃(iris) 데이터를 사용해서 KNN 분류를 해보는 예제입니다.

전체 흐름은 아주 간단합니다.
1. CSV 파일에서 데이터를 읽어옵니다.
2. 입력값(X)과 정답(y)으로 나눕니다.
3. 학습용 데이터와 시험용 데이터로 나눕니다.
4. 숫자 크기를 비슷하게 맞추기 위해 스케일링을 합니다.
5. 여러 k 값으로 KNN 모델을 시험해 봅니다.
6. 가장 성능이 좋은 k 값을 고릅니다.
7. 그 k 값으로 다시 예측합니다.
8. 정확도, 분류 리포트, 혼동 행렬을 출력합니다.

이 예제에서 중요한 점은 KNN이 "거리"를 사용한다는 것입니다.
그래서 숫자 크기가 너무 다르면 큰 숫자가 더 큰 영향을 줄 수 있습니다.
이를 막기 위해 StandardScaler를 사용해 각 특성의 크기를 비슷하게 맞춥니다.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 현재 파이썬 파일과 같은 폴더에 있는 iris.csv 파일 경로를 찾습니다.
# 이렇게 쓰면 윈도우, 맥, 리눅스에서 모두 더 안전하게 동작합니다.
file_path = Path(__file__).with_name("iris.csv")

# CSV 파일을 표 형태의 데이터프레임으로 읽어옵니다.
df = pd.read_csv(file_path)

# 'variety' 열은 우리가 맞히고 싶은 정답이므로 X에서 뺍니다.
# X는 문제지의 입력값이라고 생각하면 됩니다.
X = df.drop(columns=["variety"])

# 'variety' 열만 따로 꺼내서 정답 y로 저장합니다.
# y는 답안지라고 생각하면 이해하기 쉽습니다.
y = df["variety"]

# 데이터를 학습용과 테스트용으로 나눕니다.
# test_size=0.2 는 전체의 20%를 테스트용으로 쓰겠다는 뜻입니다.
# random_state=42 는 실행할 때마다 같은 방식으로 나누기 위해 넣습니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# StandardScaler 객체를 만듭니다.
# 이 도구는 각 열의 숫자 크기를 비슷하게 맞춰 줍니다.
scaler = StandardScaler()

# 학습용 데이터의 기준으로 평균과 표준편차를 배웁니다.
# 그리고 그 기준으로 학습용 데이터를 변환합니다.
X_train_scaled = scaler.fit_transform(X_train)

# 테스트용 데이터도 같은 기준으로만 변환합니다.
# 여기서 fit을 다시 하면 시험 문제를 미리 보고 공부하는 것과 비슷해져서 안 됩니다.
X_test_scaled = scaler.transform(X_test)

# 시험해 볼 k 값 후보들을 리스트로 준비합니다.
# range(1, 34, 2)는 1부터 시작해서 2씩 커지는 숫자를 34 전까지 만든다는 뜻입니다.
# 그래서 1, 3, 5, 7, ..., 33이 만들어집니다.
# 홀수를 자주 쓰는 이유는 동점이 나는 일을 줄이는 데 도움이 되기 때문입니다.
k_candidates = list(range(1, 34, 2))

# 각 k 값의 결과를 저장할 빈 리스트를 만듭니다.
k_scores = []

# 여러 k 값을 비교해 보기 위한 제목과 구분선을 출력합니다.
print("\n" + "=" * 50)
print("[k 값별 정확도 비교]")

# k 후보를 하나씩 꺼내서 반복합니다.
for k in k_candidates:
    # 현재 k 값으로 KNN 모델을 하나 만듭니다.
    candidate_model = KNeighborsClassifier(n_neighbors=k)

    # 학습용 데이터로 모델을 학습시킵니다.
    candidate_model.fit(X_train_scaled, y_train)

    # 테스트용 데이터를 넣어서 예측 결과를 만듭니다.
    candidate_pred = candidate_model.predict(X_test_scaled)

    # 예측이 얼마나 잘 맞았는지 정확도를 계산합니다.
    candidate_acc = accuracy_score(y_test, candidate_pred)

    # 나중에 가장 좋은 k를 찾기 위해 (k, 정확도)를 함께 저장합니다.
    k_scores.append((k, candidate_acc))

    # 현재 k 값의 결과를 한 줄씩 출력합니다.
    print(f"k={k}: 정확도 {candidate_acc * 100:.2f}%")

# max 함수로 가장 정확도가 높은 k 값을 찾습니다.
# key=lambda item: item[1] 은 (k, 정확도) 중에서 두 번째 값인 정확도를 기준으로 비교하라는 뜻입니다.
best_k, best_acc = max(k_scores, key=lambda item: item[1])

# 가장 좋은 결과를 출력합니다.
print("=" * 50)
print(f"가장 좋은 k 값: {best_k} (정확도 {best_acc * 100:.2f}%)")
print("=" * 50 + "\n")

# 이제 가장 좋은 k 값으로 최종 KNN 모델을 만듭니다.
knn = KNeighborsClassifier(n_neighbors=best_k)

# 스케일링된 학습용 데이터를 이용해 모델을 학습합니다.
knn.fit(X_train_scaled, y_train)

# 테스트용 데이터를 보고 품종을 예측합니다.
y_pred = knn.predict(X_test_scaled)

# 보기 좋게 구분선을 출력합니다.
print("\n" + "=" * 50)

# 어떤 k 값이 최종 선택되었는지 먼저 보여 줍니다.
print(f"최종 선택된 k 값: {best_k}")

# 실제 정답과 예측 정답을 비교해서 전체 정확도를 계산합니다.
print(f"붓꽃 품종 분류 정확도: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 다시 구분선을 출력합니다.
print("=" * 50 + "\n")

# classification_report는 품종별 정밀도, 재현율, F1-score를 계산해 줍니다.
# output_dict=True 로 두면 표처럼 다루기 쉬운 사전(dictionary) 형태로 받습니다.
report_dict = classification_report(y_test, y_pred, output_dict=True)

# 사전 형태 결과를 데이터프레임으로 바꿔서 표처럼 깔끔하게 출력합니다.
report_df = pd.DataFrame(report_dict).transpose()

# 표 제목을 출력합니다.
print("[표 1: 분류 성능 상세 분석 지표]")

# round(3)은 소수점 셋째 자리까지 보이도록 정리해 줍니다.
print(report_df.round(3))

# 다음 표와 구분하기 위한 선입니다.
print("=" * 50 + "\n")

# 혼동 행렬을 계산합니다.
# 이 표는 "실제 정답"과 "예측 결과"를 한눈에 비교할 수 있게 도와줍니다.
cm = confusion_matrix(y_test, y_pred)

# 모델이 알고 있는 품종 이름 목록을 가져옵니다.
labels = knn.classes_

# 혼동 행렬도 데이터프레임으로 바꿔서 더 읽기 쉽게 만듭니다.
# 행은 실제 정답, 열은 예측한 정답이라는 뜻을 이름에 붙였습니다.
cm_df = pd.DataFrame(
    cm,
    index=[f"실제_{label}" for label in labels],
    columns=[f"예측_{label}" for label in labels],
)

# 표 제목을 출력합니다.
print("[표 2: 혼동 행렬 (Confusion Matrix Table)]")

# 혼동 행렬 표를 출력합니다.
print(cm_df)

# 마지막 구분선을 출력합니다.
print("=" * 50)
