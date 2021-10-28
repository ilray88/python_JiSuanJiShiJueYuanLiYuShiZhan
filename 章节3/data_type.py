import tensorflow.compat.v1 as tf

var1 = tf.Variable(1.5, name='var1')
# 将浮点数转为int32
re1 = tf.cast(var1, dtype=tf.int32, name='var2')

const1 = tf.constant(False, name='const1')
# 将布尔值转为float32
re2 = tf.cast(const1, dtype=tf.float32, name='const2')

plh1 = tf.placeholder(dtype=tf.string, name='plh1')
# 将string转为bool（报错）
re3 = tf.cast(plh1, dtype=tf.bool)

with tf.Session() as sess:
    # 初始化所有的变量
    sess.run(tf.global_variables_initializer())
    print(sess.run(var1))
    print(sess.run(re1))

    print(sess.run(const1))
    print(sess.run(re2))

    print(sess.run(plh1, feed_dict={plh1: 'TensorFlow is awesome'}))

    # 报错，不允许从string转换为bool
    # print(sess.run(re3, feed_dict={plh1: 'TensorFlow is great'}))
    