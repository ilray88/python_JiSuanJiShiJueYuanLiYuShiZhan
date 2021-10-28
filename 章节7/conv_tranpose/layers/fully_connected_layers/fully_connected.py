import tensorflow.compat.v1 as tf
from functools import reduce
import math

# 实现方式1，使用定义计算
def fully_connected(inp, out_num, name='fully_connected'):
	with tf.variable_scope(name):
		shape = inp.get_shape().as_list()
		
		# 当输入不是二维张量时进行重整
		if len(shape) != 2:
			# 新形状指定为[-1, n]，其中n为除第一个维度外的所有维度乘积
			inp = tf.reshape(inp, shape=[-1, reduce(lambda x, y: x * y, shape[1: ])])

		n = inp.get_shape().as_list()[-1]
		
		w = tf.Variable(
				tf.truncated_normal(
					[n, out_num], 
					mean=0.0, 
					stddev=math.sqrt(2 / min(n, out_num))
				), dtype=tf.float32, name='w')
		
		b = tf.Variable(
			tf.zeros(
				[out_num]
			), dtype=tf.float32, name='b')
		
		output = tf.add(tf.matmul(inp, w), b)

		return output


# 实现方式2，使用tf.layers.dense函数
def fully_connected_(inp, out_num, name='fully_connected'):
	with tf.variable_scope(name):
		shape = inp.get_shape().as_list()
		
		if len(shape) != 2:
			inp = tf.reshape(inp, shape=[-1, reduce(lambda x, y: x * y, shape[1: ])])
		
		return tf.layers.dense(
			inp,
			units=out_num,
			name='dense'
		)