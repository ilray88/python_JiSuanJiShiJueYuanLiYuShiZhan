import tensorflow.compat.v1 as tf


def batch_normalization(inp, 
                        name, 
                        weight1=0.99, 
                        weight2=0.99, 
                        is_training=True):
    with tf.variable_scope(name):
        # 获取输入张量的形状
        inp_shape = inp.get_shape().as_list()

        # 定义不可训练变量hist_mean记录均值的移动平均值
        # 形状与输入张量最后一个维度相同
        hist_mean = tf.get_variable('hist_mean', 
                                    shape=inp_shape[-1:], 
                                    initializer=tf.zeros_initializer(), 
                                    trainable=False)

        # 定义不可训练变量hist_var记录方差的移动平均值
        # 形状与输入张量最后一个维度相同
        hist_var = tf.get_variable('hist_var', 
                                   shape=inp_shape[-1:], 
                                   initializer=tf.ones_initializer(), 
                                   trainable=False)

        # 定义可训练变量gamma和beta，形状与输入张量最后一个维度相同
        gamma = tf.Variable(tf.ones(inp_shape[-1:]), name='gamma')
        beta = tf.Variable(tf.zeros(inp_shape[-1:]), name='beta')

        # 计算输入张量除了最后一个维度外上面的均值与方差
        batch_mean, batch_var = tf.nn.moments(inp, 
                                    axes=[i for i in range(len(inp_shape) - 1)], 
                                    name='moments')

        # 计算均值的移动平均值，并将计算结果赋予hist_mean/running_mean
        running_mean = tf.assign(hist_mean, 
                                 weight1 * hist_mean + (1 - weight1) * batch_mean)

        # 计算方差的移动平均值，并将计算结果赋予hist_var/running_var
        running_var = tf.assign(hist_var, 
                                weight2 * hist_var + (1 - weight2) * batch_var)

        # 使用control_dependencies限制先计算移动平均值
        with tf.control_dependencies([running_mean, running_var]):
            # 根据当前状态是训练或是测试选取不同的值进行标准化
            # is_training=True，使用batch_mean & batch_var
            # is_training=False，使用running_mean & running_var
            output = tf.cond(tf.cast(is_training, tf.bool),
                             lambda: tf.nn.batch_normalization(inp, 
                                                mean=batch_mean, 
                                                variance=batch_var, 
                                                scale=gamma, 
                                                offset=beta, 
                                                variance_epsilon=1e-5, 
                                                name='bn'),
                             lambda: tf.nn.batch_normalization(inp, 
                                                mean=running_mean, 
                                                variance=running_var, 
                                                scale=gamma, 
                                                offset=beta, 
                                                variance_epsilon=1e-5, 
                                                name='bn')
                             )
        return output

def _batch_normalization(inp, name, weight1=0.99, weight2=0.99, is_training=True):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(
            inp,
            training=is_training
        )