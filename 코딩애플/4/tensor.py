import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

# 보통 이 초깃값은 랜덤으로 집어넣거나 하긴 합니다.
a = tf.Variable(0.1)
b = tf.Variable(0.1)

# 항상 모델을 먼저 만들어야한다.
# 예측_y = train_x * a + b

# 손실함수
# 데이터가 여러개 일 때는? 오차1^2 + 오차2^2 + ... / n 을 하면된다.
# 이게 meansquarederror 이다. 근데, 케라스에서는 이것도 구현 해 놨다.
def loss_function(a, b):
    예측_y = train_x * a + b # a가 텐서플로우라서 리스트 덧셈/곱셈이 가능하다.
    # 순서 조심하고 실제값, 예측값을 적어주자. (리스트를 넣으면 한 번에 계산 알아서해준다.)
    return tf.keras.losses.mse(train_y, 예측_y)
    



# 러닝레이트도 여러분들이 결과가 잘 나올 때 까지 값을 수정하면서 찾으시면 됩니다.
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(1000):
    # (a,b)파라미터 없이도 가능하지만, 이렇게 파라미터 써 주는게 더 직관적이고 나은 방식이다.
    # 근데 그렇게 파라미터 쓰려면, 함수로 또 감싸줘야한다. (lambda를 쓰면 된다.)
    opt.minimize(lambda:loss_function(a,b), var_list=[a,b])
    print(a.numpy(), b.numpy())
