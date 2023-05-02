import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 데이터셋 로딩
wine = load_wine()
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# 데이터 전처리
X = X / 255.
y = y.astype(int)

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 입력값 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP 모델 생성 및 학습
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10)
mlp.fit(X_train, y_train)

# 테스트 데이터로 예측 수행 및 정확도 출력
accuracy = mlp.score(X_test, y_test)
print("Accuracy:", accuracy)