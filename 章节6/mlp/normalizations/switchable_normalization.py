import tensorflow.compat.v1 as tf

def switchable_normalization(inp, name):
    with tf.variable_scope(name):
        # 获取输入张量的形状
        inp_shape = inp.get_shape().as_list()

        # 定义可训练变量gamma和beta，形状为[n,1,1,c]方便直接线性变换
        para_shape = [inp_shape[0], 1, 1, inp_shape[-1]]
        gamma = tf.Variable(tf.ones(para_shape, name='gamma'))
        beta = tf.Variable(tf.zeros(para_shape, name='beta'))

        # 计算输入张量第一（H)和第二（W）维度外上面的均值与方差
        insta_mean, insta_var = tf.nn.moments(inp, 
                                        axes=[1,2],
                                        name='moments', keep_dims=True)

        output = gamma * (inp - insta_mean) / tf.sqrt(insta_var + 1e-5) + beta
        return output

if __name__ == "__main__":
    a = tf.ones([128, 10, 10, 3])
    b = instance_normalization(a, name='in')
    print(b.shape)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(b))