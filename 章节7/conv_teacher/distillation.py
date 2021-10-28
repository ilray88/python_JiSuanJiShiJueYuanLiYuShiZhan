import os
import numpy as np
import tensorflow.compat.v1 as tf
import random

from tools.printer import VarsPrinter, AccPrinter
from models.fullyConnected import FullyConnected

from optimizers.optimizer import Optimizer
from data_utils.mnist import Mnist

# 设定batch大小为32
batch_size = 32

# 数据与标签文件
files = ['train-images.idx3-ubyte', 't10k-images.idx3-ubyte',
         'train-labels.idx1-ubyte', 't10k-labels.idx1-ubyte']
# 数据存储路径
data_path = r'data path'

# 初始化数据集对象并执行标准化
mnist = Mnist(data_path=[os.path.join(data_path, _p) for _p in files], 
              batch_size=batch_size)
train_iter = mnist.num_examples('train') // batch_size
output_cls = [np.zeros([10]) for _ in range(10)]

ckpt_folder = '../conv_nets/checkpoint/resnet18_mnist'
ckpt_path = os.path.join(ckpt_folder, "checkpoint-288")

T = 10

resnet_x = list()
resnet_y = list()
ys = list()

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(os.path.join(ckpt_folder, 'checkpoint-288.meta'))
    new_saver.restore(sess, ckpt_path)

    graph = tf.get_default_graph()

    # 获取模型中的占位符
    X = graph.get_operation_by_name('X').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]

    # 获取模型输出张量节点
    model_output = graph.get_tensor_by_name('resnet_18/classifier/fully_connected/Add:0')
    # 计算ResNet输出蒸馏后的结果
    model_softmax_output = tf.nn.softmax(model_output / T)

    for _ in range(train_iter):
        batch_x, batch_y = mnist.next_batch('train')
        # 得到最终的卷积输出与全连接层的权重
        outputs = sess.run(model_softmax_output, 
                           feed_dict={X: np.reshape(batch_x, [-1, 28, 28, 1]), is_training: False})
        
        for i in range(batch_size):
            resnet_x.append(batch_x[i])
            resnet_y.append(outputs[i])
            ys.append(batch_y[i])


resnet_x = np.stack(resnet_x, axis=0)
resnet_y = np.stack(resnet_y, axis=0)
ys = np.stack(ys, axis=0)

# 数据样本以及标签的占位符
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784])
y_soft = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10])
y_hard = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10])

# 隐层包含256个节点，输出层包含10个节点
structure = [256, 10]

# 搭建全连接神经网络
model = FullyConnected(structure)
output = model.build(x)

alpha = 0.8

# 
loss = alpha * tf.losses.softmax_cross_entropy(y_soft, output / T) + (1 - alpha) * tf.losses.softmax_cross_entropy(y_hard, output)

# 选用NAG，学习率在第40、60、80周期时减小为1/5
op = Optimizer(1e-2, [200, 300, 400], 0.2, False, 0, 'nestrov').minimize(loss)

# 通过计算标签与模型输出最大分量位置是否相同计算模型是否预测正确
acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.math.argmax(output, axis=1), 
                tf.math.argmax(y_hard, axis=1)
            ), tf.float32
        )
      )

acc_printer = AccPrinter()
# 记录最大的准确率
acc_max = -1

with tf.Session() as sess:
    # 初始化全连接网络中的随机变量
    tf.global_variables_initializer().run()

    # 训练500个周期
    for i in range(500):
        # 拿出每个周期的loss，方便观察其变化情况
        loss_e = 0
        
        # 取出所有ResNet的预测结果
        for idx in range(resnet_x.shape[0] // batch_size):
            index = random.choices(range(resnet_x.shape[0]), k=batch_size)
            _x = resnet_x[index]
            _y_soft = resnet_y[index]
            _y_hard = ys[index]

            loss_i, _ = sess.run([loss, op], feed_dict={x: _x, y_soft: _y_soft, y_hard: _y_hard})
            loss_e += loss_i
        
        test_iter = mnist.num_examples('test') // batch_size
        acc_e = 0

        for _ in range(test_iter):
            _x, _y = mnist.next_batch('test')
            acc_e += sess.run(acc, feed_dict={x: _x, y_hard:_y}) / test_iter
        
        if acc_e > acc_max:
            # 保存当前表现最佳的模型
            # 更新最佳准确率
            acc_max = acc_e
            print('Saving...')

        acc_printer(i, loss_e, acc_e)
