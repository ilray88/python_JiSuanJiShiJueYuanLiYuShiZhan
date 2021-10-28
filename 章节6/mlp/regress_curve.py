import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np

from tools.printer import VarsPrinter
from models.fullyConnected import FullyConnected

# Loss类含有各种损失函数，根据传入的损失名称执行不同计算
from losses.loss import Loss

# Optimizer类含有各种优化器
# 接受学习率、是否warmup、学习率如何递减以及使用哪一种优化器作为参数
from optimizers.optimizer import Optimizer

# activation.py文件中含有各种激活函数，直接import需要的激活函数即可
from activations.activations import leaky_relu

# 函数的定义域
x = np.float32(
        np.linspace(-10, 1, 1000)
    ).reshape(-1, 1)

# 根据定义域计算函数值
y = np.float32(
        np.exp(x) - x * np.sin(x) * np.cos(x) + np.log(x ** 2 + 1) * np.sin(x)
    ).reshape(-1, 1)

# 为数据添加高斯噪声
# y = np.float32(y + np.random.randn(1000, 1))

# 定义需要拟合的数据
data = tf.constant(x, dtype=tf.float32)
label = tf.constant(y, dtype=tf.float32)

# 隐层包含1个节点，输出层包含1个节点
# 隐含层个数可以任意更改，也可以增加隐含层数量
structure = [8, 1]

# 搭建全连接神经网络
model = FullyConnected(structure)

# 调用模型得到其输出，并使用Leaky ReLU完成第二次非线性转换
output = model.build(data)
output = leaky_relu(output, a=0.5, name='lrelu')

# 使用最小均方差作为损失函数，还可以选用mae、huber等损失
loss = Loss('mse').get_loss(y, output)

# 因为我们只需要看模型拟合函数的能力，不涉及模型在测试数据上的表现
# 所以选用在训练集能收敛更快的adam优化器
op = Optimizer(1e-1, None, 1.0, False, 0, 'adam').minimize(loss)

# 用于打印模型中所有参数的工具
VarsPrinter()()

# 打开交互作图模式，方便查看拟合
plt.ion()

with tf.Session() as sess:
    # 初始化全连接网络中的随机变量
    tf.global_variables_initializer().run()

    # 训练2000个周期
    for i in range(2000):
        # 拿出每个周期的loss，方便观察其变化情况
        loss_i, output_i, _ = sess.run([loss, output, op])

        # 绘图部分
        plt.cla()
        
        # 绘制原始数据
        plt.plot(x, y)
        # plt.scatter(x, y, s=1)

        # 绘制模型拟合结果
        plt.plot(x, output_i)

        plt.pause(0.001)

        # 打印每一个周期的loss信息
        print('Epoch {}: loss -> {}'.format(i, loss_i))

plt.ioff()
plt.show()