import tensorflow.compat.v1 as tf


def layer_normalization(inp, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # 获取输入张量的形状
        inp_shape = inp.get_shape().as_list()

        # 定义可训练变量gamma和beta，batch维度与输入张量第一个维度相同
        para_shape = [inp_shape[0]] + [1] * (len(inp_shape) - 1)
        # gamma = tf.Variable(tf.ones(para_shape, name='gamma'))

        gamma = tf.get_variable(name='gamma',
                                shape=para_shape,
                                dtype=tf.float32,
                                intializer=tf.ones_initializer)

        # beta = tf.Variable(tf.zeros(para_shape, name='beta'))

        beta = tf.get_variable(name='beta',
                               shape=para_shape,
                               dtype=tf.float32,
                               intializer=tf.zeros_initializer)
        # 计算输入张量除了第一个维度外上面的均值与方差
        layer_mean, layer_var = tf.nn.moments(inp, 
                                    axes=[i for i in range(1, len(inp_shape))],
                                    name='moments', keep_dims=True)

        output = gamma * (inp - layer_mean) / tf.sqrt(layer_var + 1e-5) + beta
        return output

if __name__ == "__main__":
    a = tf.ones([128, 10, 10, 3])
    b = layer_normalization(a, name='ln')
    print(b.shape)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(b))