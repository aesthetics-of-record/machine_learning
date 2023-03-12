import tensorflow as tf
import matplotlib.pyplot as plt

# ( ( trainX, trainY ), ( testX, testY ) ) 가 튜플형식으로 들어있다.
( trainX, trainY ), ( testX, testY ) = tf.keras.datasets.fashion_mnist.load_data()

# print(trainX)
# print(trainX.shape) # trainX는 이미지 6만개입니다.
#
# print(trainY) # 정답(종류)이 들어있는 리스트 글자를 정수로 치환해서 저장해놓은 겁니다.

# plt.imshow( trainX[0] )
# plt.gray() # 원래는 흑백인데, 이걸 설정안하면 보기 편하라고 컬러로 나온다.
# plt.colorbar()
# plt.show()

## 딥러닝 순서 : 1. 모델만들기 2. compile 하기 3. fit 하기

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 이렇게 0~255를 나눠서 0~1로 압축시켜도 된다. 정확도가 더 잘 나오는 쪽으로 가면 된다.
trainX = trainX / 255.0
testX = testX / 255.0

# 1을추가해서 한차원을 더 추가해 리쉐잎 해줘야한다.
# 이게 흑백데이터라 그렇다. 컬러데이터는 [0, 0, 0]인데, 흑백이라 0이된다. 그래서 한 차원이 부족하다.
trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1) )
testX = testX.reshape( (testX.shape[0], 28, 28, 1) )

model = tf.keras.models.Sequential([
	# Conv2D는 4차원데이터 입력이 필요하기 때문에, input_shape를 3차원으로 설정해줘야하고,
	# 데이터에 괄호를 한 번 더 쳐줘야합니다. [[]] => [[[]]]
	tf.keras.layers.Conv2D( 32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
	tf.keras.layers.MaxPooling2D( (2, 2) ),
	# relu : 음수는 다 0으로 만들어주셈 (이미지는 0~255로 음수가 나오면 안 되니까)
	# tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu'),
	tf.keras.layers.Flatten(), # 행렬을 1차원으로 압축해주는 레이어
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax'),
	# 결과를 0에서 1사이로 확률에 쓰는 softmax, 주로 카테고리 예측같은 것에 사용. 노드갯수는 카테고리갯수

	# sigmoid도 0과 1로 압축해주나, 대학합격 같은 binary예측 문제에 사용히고, 마지막 노드 갯수는 1개쓰면 됩니다.
	# 원하는 아웃풋에 따른 마지막레이어가 중요합니다.
])

# input_shape=(,)를 써줘야 summary를 쓸 수 있다.
model.summary()


# 'sparse_categorical_crossentropy' 카테고리 예측에 쓰이는 손실함수 (trainY가 정수형 0, 1, 2, ... 일 때)
# 'categorical_crossentropy'는 trainY가 원핫인코딩이 되어있을 때 사용.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20)

# 모델평가 (여기에는 trainX를 넣으면 안 됩니다. 너무 많이 봐서 답안을 외울수도 있어요.)
# 그래서 생전 처음보은 testX, Y 를 넣어줘야한다.
# 얘도 근데, 기존 답안을 많이보고 그것만 외워서 정확도를 높인거라, 새로운 데이터를 넣으면 상대적으로 낮게 나온다.
# 그래서 epoch 1회 끝날 때 마다 채점하는 방법도 있습니다. validation_data=(testX, testY)
# score = model.evaluate(testX, testY)
# print(score) # score => [ loss, accuracy ]

# 이렇게 중간중간 테스트하면 장점이, 오버피팅이 일어날 때 멈추고, 그 때의 모델을 뽑을 수 있습니다.



