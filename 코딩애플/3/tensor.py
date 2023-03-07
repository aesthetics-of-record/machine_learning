import tensorflow as tf

키 = 170
신발 = 260
# 신발 = 키 * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)


def 손실함수():
  # 보통은 오차(실제-예측)를 뱉어주면 된다.
  # 근데 그냥 하기보다는, 제곱을 해 주는 게 좋다.
  예측값 = (키 * a + b)
  return tf.square(260 - 예측값)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

print(170 * 1.52 + 1.62)

# 이 경사하강을 반복하면 된다.
# for i in range(300):
#   opt.minimize(손실함수, var_list=[a, b])
#   print(a.numpy(), b.numpy())
  