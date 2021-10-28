import tensorflow.compat.v1 as tf

def global_avg_pooling(inp, name='global_avg_pool'):
    with tf.variable_scope(name):
        output = tf.math.reduce_mean(inp,
                                     axis=[1, 2],
                                     keepdims=True, name='global_avg_pool')

        return output