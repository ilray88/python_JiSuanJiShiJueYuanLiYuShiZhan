import tensorflow.compat.v1 as tf
import numpy as np
import os

from tools.printer import VarsPrinter, AccPrinter
from models.fullyConnected import FullyConnected

# Loss类含有各种损失函数，根据传入的损失名称执行不同计算
from losses.loss import Loss

# Optimizer类含有各种优化器
# 接受学习率、是否warmup、学习率如何递减以及使用哪一种优化器作为参数
from optimizers.optimizer import Optimizer
from data_utils.iris import Iris

# 设定batch大小为30（为了保持和测试集大小兼容）
batch_size = 30

# 数据存储路径
data_path = r'data path\iris\iris.data'

# 初始化数据集对象，默认进行shuffle与标准化
iris = Iris(data_path=data_path, 
            batch_size=batch_size)

# 数据样本由4个分量组成，标签由3个分量组成
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 3])

# 隐层包含20个节点，输出层包含3个节点
structure = [20, 3]

# 搭建全连接神经网络
model = FullyConnected(structure)
output = model.build(x)

# 使用交叉熵计算损失值
loss = Loss('ce').get_loss(y, output)

# 通过计算标签与模型输出最大分量位置是否相同计算模型是否预测正确
acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.math.argmax(output, axis=1), 
                tf.math.argmax(y, axis=1)
            ), tf.float32
        )
      )

# 默认选用NAG，学习率在第40、60、80周期时减小为1/5
op = Optimizer(1e-1, [40, 60, 80], 0.2, False, 0, 'nestrov').minimize(loss)

# 用于打印模型中所有参数的工具
VarsPrinter()()
acc_printer = AccPrinter()

with tf.Session() as sess:
    # 初始化全连接网络中的随机变量
    tf.global_variables_initializer().run()

    # 训练100个周期
    for i in range(100):
        # 拿出每个周期的loss，方便观察其变化情况
        loss_e = 0
        
        # 每个周期包括iris.num_examples('train') // batch_size次迭代
        for _ in range(iris.num_examples('train') // batch_size):
            _x, _y = iris.next_batch('train')

            loss_i, _ = sess.run([loss, op], feed_dict={x: _x, y: _y})
            loss_e += loss_i
        
        test_x, test_y = iris.next_batch('test')
        
        # 计算每个周期模型在测试集上的表现
        acc_e = sess.run(acc, feed_dict={x: test_x, y: test_y})
        # 打印每一个周期的信息
        acc_printer(i, loss_e, acc_e)
