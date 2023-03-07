import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('c:/Users/Slim5/Documents/GitHub/machine_learning/코딩애플/5/gpascore.csv')
# 왜 데이터를 전처리하느냐? 빈 데이터도 분명 있을 거다. 그런거를 판다스로 처리해야한다.
# print(data.isnull().sum())
data = data.dropna() # 빈칸있는 행 제거

# .values는 리스트로 바꿔줍니다.
y_data = data['admit'].values


x_data = []

# 판다스로 연 dataframe 에 itterrows()를 쓰면 한 행씩 출력 해 볼 수 있습니다.
for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])


# 케라스를 통한 쉬운 딥러닝 모델 만들기
model = tf.keras.models.Sequential([
    # 레이어 만들기 / 히든레이어에 들어갈 노드 개수, 활성함수
    # 노드개수는 맘대로 써도 되나 관습적으로 2의 제곱수로 보통 하는 것 같더라고요.
    # 활성함수는 sigmoid / tanh / relu / leakyrelu / elu / linear / softmax 등 여러가지가 있습니다.
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid'), # 마지막레이어는 항상 예측 결과를 뱉어야함.
    # sigmoid 는 모든 것을 0과 1사이의 확률로 예측 결과를 뱉어냅니다.
])

# 우리가 기울기를 뺄 때 러닝메이트를 이용해서 뺀다고 했는데, 그 빼는 값을 조정해주는 게 옵티마이저입니다.
# adam, adagrad, adadelta, rmsprop, sgd 등의 옵티마이저가 있는데, 기본적으로 그냥 adam을 쓰면 됩니다.
# 로스함수중에 bianry_crossentropy는 보통 0과 1사이의 분류/확률 문제에서 씁니다.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 에포크는 학습시키고 싶은 수다.
# x에는 학습할 데이터를 행을 리스트로 바꿔서 저장하면된다.
# 근데, 여기에 데이터 넣을 때 리스트를 넣는게 아니라, npmpy array나 tensor를 집어넣어야 합니다.
model.fit(np.array(x_data), np.array(y_data), epochs=2000)

# 근데 매번 정확도는 운빨입니다.
# 실무에서는 여러번 돌리고, 정확도가 좋은 경우를 저장해놓고 쓰면 됩니다.

########################################################
# 예측
예측값 = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(예측값)


# 지금까지 딥러닝을 봤는데요.
# 실제로 딥러닝학습을 하려면 여러분들이 연구를 하셔야합니다.
# 어떤 활성함수를 넣고 손실함수를 넣고.. 레이어 개수.. 노드 개수.. 등등
# 정해진 게 없고, 실험적으로 때려 맞쳐서 놀아보시면 됩니다.
# 이 데이터는 조금 중구난방 해서, 결과값이 잘 나오지는 않지만, 그래도 한 번 해보십시오.