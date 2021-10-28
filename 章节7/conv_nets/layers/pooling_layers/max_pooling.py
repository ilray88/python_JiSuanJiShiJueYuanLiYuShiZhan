import tensorflow.compat.v1 as tf

def max_pooling(inp, kernel, stride, padding='SAME', name='avg_pool'):
    with tf.variable_scope(name):
        output = tf.nn.max_pool(inp,
                                ksize=[1, kernel, kernel, 1],
                                strides=[1, stride, stride, 1],
                                padding=padding, name='max_pool')

        return output