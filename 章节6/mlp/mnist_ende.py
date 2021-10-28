import tensorflow.compat.v1 as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from tools.printer import VarsPrinter
from models.fullyConnected import FullyConnected

from losses.loss import Loss

from optimizers.optimizer import Optimizer
from data_utils.mnist import Mnist

# 设定batch大小为32
batch_size = 32

# 数据与标签文件
files = ['train-images.idx3-ubyte', 't10k-images.idx3-ubyte',
         'train-labels.idx1-ubyte', 't10k-labels.idx1-ubyte']
# 数据存储路径
data_path = r'data path'

# 初始化数据集对象，不执行标准化
mnist = Mnist(data_path=[os.path.join(data_path, _p) for _p in files], 
              batch_size=batch_size,
              normalize=False)

# 数据样本由784个分量组成，使用输入作为标签
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784])

# 隐层包含256、10、256个节点，输出层包含784个节点
structure = [256, 10, 256, 784]

# 搭建全连接神经网络
model = FullyConnected(structure)
output = model.build(x)

# 最后一层使用Sigmoid激活函数将输出压缩至0-1（与归一化的图像取值范围保持一致）
output = tf.nn.sigmoid(output)

# 使用均方差计算重构损失
loss = Loss('mse').get_loss(y, output)

# 选用adam，学习率在第40、60、80周期时减小为1/5
op = Optimizer(1e-3, [40, 60, 80], 0.2, False, 0, 'adam').minimize(loss)

# 用于打印模型中所有参数的工具
VarsPrinter()()

plt.ion()
fig, axes = plt.subplots(1, 2)

with tf.Session() as sess:
    # 初始化全连接网络中的随机变量
    tf.global_variables_initializer().run()

    # 训练100个周期
    for i in range(100):
        # 拿出每个周期的loss，方便观察其变化情况
        loss_e = 0
        plt.cla()

        # 每个周期包括mnist.num_examples('train') // batch_size次迭代
        for _ in range(mnist.num_examples('train') // batch_size):
            # 不需要使用数字标签，以_替代
            _x, _ = mnist.next_batch('train')
            _x = _x / 255.0

            # 监督标签也是输入数据
            loss_i, _ = sess.run([loss, op], feed_dict={x: _x, y: _x})
            loss_e += loss_i
        
        output_i = sess.run(output, feed_dict={x: _x, y: _x})

        axes[0].imshow(np.reshape(_x[0], (28, 28)), cmap='gray')
        axes[1].imshow(np.reshape(output_i[0], (28, 28)), cmap='gray')
        plt.pause(0.001)

        print('Epoch {}: {}'.format(i, loss_e))

plt.ioff()
plt.show()