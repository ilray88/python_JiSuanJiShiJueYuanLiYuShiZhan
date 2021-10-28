import tensorflow.compat.v1 as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from tools.printer import VarsPrinter, AccPrinter
from models.fullyConnected import FullyConnected

from losses.loss import Loss

from optimizers.optimizer import Optimizer
from data_utils.mnist import Mnist

# ================================================================
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

# 数据样本由784个分量组成，输出是10维的向量
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10])

# 隐层包含256个节点，输出层包含10个节点
structure = [256, 10]

# 搭建全连接神经网络
model = FullyConnected(structure)
output = model.build(x)

# 使用均方差计算重构损失
loss = Loss('ce').get_loss(y, output)

# 选用NAG，学习率在第40、60、80周期时减小为1/5
op = Optimizer(1e-2, [40, 60, 80], 0.2, False, 0, 'nestrov').minimize(loss)

# 通过计算标签与模型输出最大分量位置是否相同计算模型是否预测正确
acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.math.argmax(output, axis=1), 
                tf.math.argmax(y, axis=1)
            ), tf.float32
        )
      )

VarsPrinter()()
acc_printer = AccPrinter()

# 定义saver
saver = tf.train.Saver()
ckpt_folder = 'ckpt'
ckpt_name = 'handwritten.ckpt'

# 记录最大的准确率
acc_max = -1

with tf.Session() as sess:
    # 初始化全连接网络中的随机变量
    tf.global_variables_initializer().run()

    # # 训练100个周期
    # for i in range(100):
    #     # 拿出每个周期的loss，方便观察其变化情况
    #     loss_e = 0
        
    #     # 每个周期包括mnist.num_examples('train') // batch_size次迭代
    #     for _ in range(mnist.num_examples('train') // batch_size):
    #         # 不需要使用数字标签，以_替代
    #         _x, _y = mnist.next_batch('train')

    #         loss_i, _ = sess.run([loss, op], feed_dict={x: _x, y: _y})
    #         loss_e += loss_i
        
    #     test_iter = mnist.num_examples('test') // batch_size
    #     acc_e = 0

    #     for _ in range(test_iter):
    #         _x, _y = mnist.next_batch('test')
    #         acc_e += sess.run(acc, feed_dict={x: _x, y:_y}) / test_iter
        
    #     if acc_e > acc_max:
    #         # 保存当前表现最佳的模型
    #         saver.save(sess, os.path.join(ckpt_folder, ckpt_name))
    #         # 更新最佳准确率
    #         acc_max = acc_e
    #         print('Saving...')

    #     acc_printer(i, loss_e, acc_e)

# ================================================================
# 纯黑的背景画布
img = np.zeros((128, 128), np.uint8)

# 定义一个全局布尔值表示当前是否需要执行绘制
drawing = False

# 鼠标绘制回调函数
def draw(event, x, y, flags, param):
    global drawing
    # 左键按下时表示开始绘制
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    # 移动鼠标并且左键按下时进行绘制
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img[y: y + 10, x: x + 10] = 255
    # 左键释放时停止绘制
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# 创建一个名为“Write number”的窗口
cv2.namedWindow('Write number')
# 为该窗口绑定回调函数
cv2.setMouseCallback('Write number', draw)

while True:
    cv2.imshow('Write number', img)
    # 当按下q键时退出绘制
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# ================================================================
# 将图像缩放为28*28
img = cv2.resize(img, (28, 28))

# 将二维图像重整为一维向量
img = np.reshape(img, [784])

# 归一化后的MNIST数据集上的均值与方差
mean = 0.13092535192648502
std = 0.3084485240270358

# 将输入数据归一化
img = img / 255.0

# 将输入数据标准化
img = (img - mean) / std

# 将输入数据堆叠起来以满足模型的输入要求
img = np.stack([img] * batch_size, axis=0)
# ================================================================
with tf.Session() as sess:
    # 取得保存的最新的权值文件
    last_ckpt = tf.train.latest_checkpoint(ckpt_folder)
    print(last_ckpt)

    # 从权值文件中读取对应权重
    saver.restore(sess, last_ckpt)

    # 得到以用户数据为输入的模型输出
    r = sess.run(output, feed_dict={x: img})

    # 由于batch中的数据都一样,因此取batch结果中的第一个结果即可
    # 使用np.argmax得到最大分量的位置即为预测结果
    print('Result: {}'.format(np.argmax(r[0])))
