import tensorflow.compat.v1 as tf


def relu(inp, name):
    with tf.variable_scope(name) as scope:
        output = tf.nn.relu(inp, name='relu')
        return output


def sigmoid(inp, name):
    with tf.variable_scope(name) as scope:
        output = tf.nn.sigmoid(inp, name='sigmoid')
        return output


def softmax(inp, axis, name):
    with tf.variable_scope(name) as scope:
        output = tf.nn.softmax(inp, axis=axis, name='softmax')
        return output


def tanh(inp, name):
    with tf.variable_scope(name) as scope:
        output = tf.nn.tanh(inp, name='tanh')
        return output

def leaky_relu(inp, a, name):
    with tf.variable_scope(name) as scope:
        output = tf.nn.leaky_relu(inp, a)
        return output

def prelu(inp, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # 根据输入数据的最后一个维度来定义参数形状
        # 对于卷积即通道数，对于全连接即特征数
        alpha = tf.get_variable('alpha', inp.get_shape()[-1],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)

        # 得到负半轴为0，正半轴不变的激活结果
        pos = tf.nn.relu(inp)

        # 得到正半轴为0，负半轴为ax的激活结果
        neg = alpha * (inp - abs(inp)) * 0.5
    
        # 将两部分激活结果相加
        return pos + neg

def rrelu(inp, is_training, name):
    with tf.variable_scope(name) as scope:
        # 定义a值取值范围
        u = 1
        l = 0

        # 从均匀分布中随机选取a值
        rand_a = tf.Variable(tf.random_uniform(tf.shape(inp), minval=l, maxval=u))

        # 若is_training=True，则使用随机生成的rand_a
        # 若is_training=False，则使用(l + u) / 2作为负半轴斜率
        alpha = tf.cond(tf.cast(is_training, tf.bool), lambda: rand_a,
                        lambda: tf.Variable((u + l) / 2.0, dtype=tf.float32))

        pos = tf.nn.relu(inp)
        neg = alpha * (inp - abs(inp)) * 0.5

        return pos + neg

def relu6(inp, name):
    with tf.variable_scope(name) as scope:
        output = tf.nn.relu6(inp)
        return output

def elu(inp, name):
    with tf.variable_scope(name) as scope:
        output = tf.nn.elu(inp)
        return output

def swish(inp, name):
    with tf.variable_scope(name) as scope:
        output = tf.nn.swish(inp)
        return output

def mish(inp, name):
    with tf.variable_scope(name) as scope:
        output = inp * tf.nn.tanh(tf.nn.softplus(inp))
        return output