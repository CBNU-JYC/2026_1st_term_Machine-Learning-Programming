import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. 데이터 로드 및 전처리 (인코딩 포함)
file_path = Path(__file__).with_name("iris.csv")
df = pd.read_csv(file_path)
X = df.drop(columns=['variety'])
y = df['variety']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 여러 k 값 비교
k_candidates = list(range(1, 34, 2))
k_scores = []

print("\n" + "=" * 50)
print("[k 값별 정확도 비교]")
for k in k_candidates:
    candidate_model = KNeighborsClassifier(n_neighbors=k)
    candidate_model.fit(X_train_scaled, y_train)
    candidate_pred = candidate_model.predict(X_test_scaled)
    candidate_acc = accuracy_score(y_test, candidate_pred)
    k_scores.append((k, candidate_acc))
    print(f"k={k}: 정확도 {candidate_acc * 100:.2f}%")

best_k, best_acc = max(k_scores, key=lambda item: item[1])
print("=" * 50)
print(f"가장 좋은 k 값: {best_k} (정확도 {best_acc * 100:.2f}%)")
print("=" * 50 + "\n")

# 3. 가장 좋은 k 값으로 최종 모델 학습 및 예측
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# === 결과 표 출력 모듈 ===

print("\n" + "=" * 50)
print(f"최종 선택된 k 값: {best_k}")
print(f"붓꽃 품종 분류 정확도: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("=" * 50 + "\n")

# 표 1: 분류 리포트 (Precision, Recall, F1-score)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print("[표 1: 분류 성능 상세 분석 지표]")
print(report_df.round(3))  # 소수점 3자리까지 출력
print("=" * 50 + "\n")

# 표 2: 혼동 행렬 (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
labels = knn.classes_
cm_df = pd.DataFrame(
    cm,
    index=[f"실제_{l}" for l in labels],
    columns=[f"예측_{l}" for l in labels]
)

print("[표 2: 혼동 행렬 (Confusion Matrix Table)]")
print(cm_df)
print("=" * 50)
