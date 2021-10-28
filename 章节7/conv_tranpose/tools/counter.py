import numpy as np
import tensorflow.compat.v1 as tf


class Counter:
    def __init__(self):
        pass

    def __call__(self, vars=None):
        if not vars:
            vars = tf.trainable_variables()
        
        amount = np.sum([np.prod(v.get_shape().as_list()) for v in vars])
        print('Total parameters: {}'.format(amount))