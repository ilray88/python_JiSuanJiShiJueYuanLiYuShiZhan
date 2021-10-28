import tensorflow.compat.v1 as tf
import numpy as np

from tools.printer import VarsPrinter
from models.fullyConnected import FullyConnected
from activations.activations import sigmoid

# 定义异或数据及其标签
data = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
label = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

# 隐层包含2个节点，输出层包含1个节点
structure = [2, 1]

# 搭建全连接神经网络
model = FullyConnected(structure)

# 调用模型得到其输出
output = model.build(data)

# 最后需要输出0-1值，所以使用Sigmoid作为其激活函数
output = sigmoid(output, name='sigmoid')

loss = tf.reduce_mean(
            -(label * tf.log(output) + (1 - label) * tf.log(1 - output))
       )

# 学习率可以自行调节
op = tf.train.MomentumOptimizer(1e-1, 0.9, use_nesterov=True).minimize(loss)

# 用于打印模型中所有参数的工具
VarsPrinter()()

with tf.Session() as sess:
    # 初始化全连接网络中的随机变量
    tf.global_variables_initializer().run()

    # 训练1000个周期
    for i in range(1000):
        # 拿出每个周期的loss，方便观察其变化情况
        loss_i, _ = sess.run([loss, op])

        # 打印每一个周期的loss信息
        print('Epoch {}: loss -> {}'.format(i, loss_i))
