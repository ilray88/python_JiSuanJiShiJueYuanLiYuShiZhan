import tensorflow.compat.v1 as tf

# 建立4个含有常量值的节点
# const1传入整型值
const1 = tf.constant(0)
# const2传入浮点数
const2 = tf.constant(0.0)
# const3传入含有整型值的list，tf会将其自动转换为const张量
const3 = tf.constant([0, 1])
# const4传入含有整型与浮点数的list，tf会将其自动转换为相应数据类型的const张量
const4 = tf.constant([0, 1.0])

# 初始化会话以运行节点
with tf.Session() as sess:
    # 分别运行4个常量值节点以及直接打印节点
    print(sess.run(const1), const1)
    print(sess.run(const2), const2)
    print(sess.run(const3), const3)
    print(sess.run(const4), const4)

# ===============================================================
# 第三种与第四种特殊传参情况
# 指定的形状shape与传入的常量形状不一致，用常量中最后一个值进行填充
# 以0填充为形状为(2, 3)的数组
const5 = tf.constant(0, shape=[2, 3])

# 以1填充为形状为(2, 3)的数组
const6 = tf.constant([0, 1], shape=[2, 3])

# 以1填充为形状为(2, 3)的数组
const7 = tf.constant([[0], [1]], shape=[2, 3])

# 指定的常量形状大于shape参数，报错
# const8 = tf.constant([0, 1], shape=[1, 1])

# 指定verify_shape参数
# 指定verify_shape为True并且常量值形状与给定的shape相同
const9 = tf.constant([[0, 1]], shape=[1, 2], verify_shape=True)

# 指定verify_shape为True并且常量值形状与给定的shape不同，报错
# const10 = tf.constant([[0, 1]], shape=[2, 1], verify_shape=True)

# 指定verify_shape为False并且常量值形状与给定的shape相同
const11 = tf.constant([[0, 1]], shape=[2, 1], verify_shape=False)

with tf.Session() as sess:
    print(sess.run(const5), const5)
    print(sess.run(const6), const6)
    print(sess.run(const7), const7)
    # print(sess.run(const8), const8)
    print(sess.run(const9), const9)
    # print(sess.run(const10), const10)
    print(sess.run(const11), const11)
    print(sess.run(const11, feed_dict={const11: [[4],[5]]}))

# ===============================================================
# 定义初始化值为0的变量，其类型为整型，并定义变量名为var1
var1 = tf.Variable(initial_value=0, name='var1')

# 定义初始化值为[0., 1]的变量，其类型为浮点型，并定义变量名为var2
var2 = tf.Variable(initial_value=[0., 1], name='var2')

# 使用随机正态分布值（均值1.0，标准差0.2）初始化变量，形状为(1, 2)，并定义变量名为var3
var3 = tf.Variable(
        initial_value=tf.random_normal(shape=[1, 2], mean=1.0, stddev=0.2),
        name='var3'
    )

# 使用整型值10初始化变量，并指定该变量不可训练，并定义变量名为var4
var4 = tf.Variable(initial_value=10, trainable=False, name='var4')

with tf.Session() as sess:
    # 初始化所有定义的变量
    sess.run(tf.global_variables_initializer())
    print(sess.run(var1), var1)
    print(sess.run(var2), var2)
    print(sess.run(var3), var3)
    print(sess.run(var4), var4)
    
# 使用tf.trainable_variables()打印所有可训练变量
for v in tf.trainable_variables():
    print(v)

# ====================================================================
# 定义一个数据类型为float32的占位符，形状任意
plh1 = tf.placeholder(dtype=tf.float32)
plh2 = tf.placeholder(dtype=tf.float32, name='plh2')

# 定义一个数据类型为float32的占位符，形状为(2, 2)
plh3 = tf.placeholder(dtype=tf.float32, shape=[2, 2], name='plh3')

# 定义一个数据类型为float32的占位符，形状第一维任意，第二维为2
plh4 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='plh4')

# 定义一个数据类型为float32的占位符，形状第一维任意，第二维也任意
plh5 = tf.placeholder(dtype=tf.float32, shape=[None, None], name='plh5')

with tf.Session() as sess:
    print(sess.run(plh1, feed_dict={plh1: 1}))
    print(sess.run(plh1, feed_dict={plh1: [1, 2]}))
    print(sess.run(plh1, feed_dict={plh1: [1, 2, 3]}))
    print(sess.run(plh2, feed_dict={plh2: 2}))
    print(sess.run(plh3, feed_dict={plh3: [[1, 2], [3, 4]]}))

    # 报错，因为feed的数据形状与placeholder定义的形状不一致
    # print(sess.run(plh3, feed_dict={plh3: [[1, 2]]}))
    
    print(sess.run(plh4, feed_dict={plh4: [[1, 2]]}))
    print(sess.run(plh4, feed_dict={plh4: [[1, 2], [3, 4]]}))
    print(sess.run(plh5, feed_dict={plh5: [[1, 2]]}))
    print(sess.run(plh5, feed_dict={plh5: [[1, 2], [3, 4]]}))
    print(sess.run(plh5, feed_dict={plh5: [[1], [2]]}))
