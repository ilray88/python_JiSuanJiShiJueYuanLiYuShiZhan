import tensorflow.compat.v1 as tf
import numpy as np


def fully_connected(inp, out_num, name):
    with tf.variable_scope(name) as scope:
        # 确保全连接层的输入张量是二维的，形状为[batch_size, n]
        if len(inp.get_shape().as_list()) != 2:
        	inp = tf.reshape(inp, shape=[inp.shape[0], -1])
        
        n = inp.get_shape().as_list()[-1]

        # 使用Kaiming初始化对参数进行初始化
        # 为了保证取值尽可能大的随机性，我们使用较大的方差进行初始化
        w = tf.Variable(
                tf.random_normal(
                    [n, out_num], 
                    mean=0.0, 
                    stddev=np.sqrt(2 / min(n, out_num))
                ), dtype=tf.float32, name='w')
        
        b = tf.Variable(
                tf.random_normal(
                    [out_num], 
                    mean=0.0, 
                    stddev=np.sqrt(2 / min(n, out_num))
                ), dtype=tf.float32, name='b')
        
        output = tf.add(tf.matmul(inp, w), b)

        return output