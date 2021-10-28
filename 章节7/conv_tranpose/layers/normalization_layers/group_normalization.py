import tensorflow.compat.v1 as tf


def group_normalization(inp, name, G=32):
    with tf.variable_scope(name):
        # 获取输入张量的形状
        insp = inp.get_shape().as_list()

        # 将输入的NHWC格式转换为NCHW方便进行分组
        inp = tf.transpose(inp, [0, 3, 1, 2])
        
        # 将输入张量进行分组，得到新张量形状为[n,G,c//G,h,w]
        inp = tf.reshape(inp, 
                [insp[0], G, insp[-1] // G, insp[1], insp[2]])

        # 定义可训练变量gamma和beta，形状为[1,1,1,c]方便直接线性变换
        para_shape = [1, 1, 1, insp[-1]]
        gamma = tf.Variable(tf.ones(para_shape, name='gamma'))
        beta = tf.Variable(tf.zeros(para_shape, name='beta'))

        # 计算输入张量第二、三和四（c//G，h，w）维度外上面的均值与方差
        group_mean, group_var = tf.nn.moments(inp, 
                                        axes=[2, 3, 4],
                                        name='moments', keep_dims=True)
        
        inp = (inp - group_mean) / tf.sqrt(group_var + 1e-5)

        # 将张量形状还原为原始形状[n,h,w,c]
        # 先将标准化之后的分组结果重新组合为[n,c,w,h]
        inp = tf.reshape(inp, 
                [insp[0], insp[-1], insp[1], insp[2]])

        # 通过转置操作将NCHW格式转换为NHWC
        inp = tf.transpose(inp, [0, 2, 3, 1])

        output = gamma * inp + beta
        return output

if __name__ == "__main__":
    import numpy as np
    a = tf.constant(np.random.randn(1,2,2,64), dtype=tf.float32)
    b = group_normalization(a, name='gn1')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        rb = sess.run(b)
        print(rb)