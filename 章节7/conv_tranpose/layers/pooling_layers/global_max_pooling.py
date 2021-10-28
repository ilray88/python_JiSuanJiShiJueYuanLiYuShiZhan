import tensorflow.compat.v1 as tf

def global_max_pooling(inp, name='global_max_pool'):
    with tf.variable_scope(name):
        output = tf.math.reduce_max(inp,
                                    axis=[1, 2],
                                    keepdims=True, name='global_max_pool')

        return output