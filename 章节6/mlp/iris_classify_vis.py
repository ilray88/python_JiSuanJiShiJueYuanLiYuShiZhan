import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tools.printer import VarsPrinter, AccPrinter

# Loss类含有各种损失函数，根据传入的损失名称执行不同计算
from losses.loss import Loss

# Optimizer类含有各种优化器
# 接受学习率、是否warmup、学习率如何递减以及使用哪一种优化器作为参数
from optimizers.optimizer import Optimizer
from data_utils.iris import Iris

class FullyConnected_IRIS:
    def build(self, x):
        from layers.fc import fully_connected
        from activations.activations import sigmoid

        with tf.variable_scope('fc') as scope:
            # 定义输入层到第一层隐含层的全连接结构与激活函数
            output1 = fully_connected(x, 20, name='fc0')
            output1 = sigmoid(output1, name='sigmoid0')

            output2 = fully_connected(output1, 2, name='fc1')
            output2 = sigmoid(output2, name='sigmoid1')

            output3 = fully_connected(output2, 3, name='fc2')

            return output3, output2

def scatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers

    if not ax: 
        ax=plt.gca()
    
    sc = ax.scatter(x,y,**kw)

    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

# 设定batch大小为30（为了保持和测试集大小兼容）
batch_size = 30

# 数据存储路径
data_path = r'data path'

# 初始化数据集对象，默认进行shuffle与标准化
iris = Iris(data_path=data_path, 
            batch_size=batch_size)

# 数据样本由4个分量组成，标签由3个分量组成
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 3])

# 搭建全连接神经网络
model = FullyConnected_IRIS()
output, output2 = model.build(x)

# 使用交叉熵计算损失值
loss = Loss('ce').get_loss(y, output)

# 默认选用NAG，学习率在第40、60、80周期时减小为1/5
op = Optimizer(1e-1, [40, 60, 80], 0.2, False, 0, 'nestrov').minimize(loss)

# 用于打印模型中所有参数的工具
VarsPrinter()()

plt.ion()

# 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2
marker = ['^', '*', 'o']
color = ['red', 'green', 'blue']

with tf.Session() as sess:
    # 初始化全连接网络中的随机变量
    tf.global_variables_initializer().run()

    # 训练100个周期
    for i in range(100):
        plt.cla()

        # 每个周期包括iris.num_examples('train') // batch_size次迭代
        for _ in range(iris.num_examples('train') // batch_size):
            _x, _y = iris.next_batch('train')

            output2_i, _ = sess.run([output2, op], feed_dict={x: _x, y: _y})

            m = [marker[np.argmax(i)] for i in _y]
            c = [color[np.argmax(i)] for i in _y]
            scatter(output2_i[:, 0], output2_i[:, 1], m=m, c=c)

        plt.pause(0.001)

plt.ioff()
plt.show()
