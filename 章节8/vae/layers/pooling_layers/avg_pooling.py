import tensorflow.compat.v1 as tf

def avg_pooling(inp, kernel, stride, padding='SAME', name='avg_pool'):
    with tf.variable_scope(name):
        output = tf.nn.avg_pool(inp,
                                ksize=[1, kernel, kernel, 1],
                                strides=[1, stride, stride, 1],
                                padding=padding, name='avg_pool')

        return output