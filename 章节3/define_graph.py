import tensorflow.compat.v1 as tf

# 为输入变量创建占位符，并为每一个变量命名
X = tf.placeholder(dtype=tf.float32, name='X')
Y = tf.placeholder(dtype=tf.float32, name='Y')
Z = tf.placeholder(dtype=tf.float32, name='Z')

# 结果一为X与Y相加
result1 = X + Y
# 结果二为(X + Y) * Z
result2 = result1 * Z
