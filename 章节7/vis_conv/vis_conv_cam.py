import os
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import cv2

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


# MNIST数据的均值与标准差，方便进行展示
mean = 0.13092535192648502
std = 0.3084485240270358

# 每个类别图像的占位符
img_data = [None] * 10

# 若有某一类别的图像未被取到则重新获取
while any([x is None for x in img_data]):
    batch_x, batch_y = mnist.next_batch('train')

    for idx, _y in enumerate(batch_y):
        # 将图像放入对应位置
        img_data[np.argmax(_y)] = batch_x[idx]


ckpt_folder = '../conv_nets/checkpoint/resnet18_mnist'
ckpt_path = os.path.join(ckpt_folder, "checkpoint-288")

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(os.path.join(ckpt_folder, 'checkpoint-288.meta'))
    new_saver.restore(sess, ckpt_path)

    graph = tf.get_default_graph()

    # 获取模型中的占位符
    X = graph.get_operation_by_name('X').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]

    # 获取目标输出张量节点
    feature_before_gap = graph.get_tensor_by_name('resnet_18/block_3_1/relu/relu:0')
    fc_w = graph.get_tensor_by_name('resnet_18/classifier/fully_connected/w:0')

    for idx, img in enumerate(img_data):
        # 将输入数据进行堆叠以满足BN的需求
        batch_img = np.stack([img] * batch_size, axis=0)
        # 得到最终的卷积输出与全连接层的权重
        features, weights = sess.run([feature_before_gap, fc_w], 
                                      feed_dict={X: batch_img, is_training: False})

        # 使用该类别相关的权重与每一个卷积输出的通道特征进行相乘并按位加
        mask = np.sum(features[0] * weights[:, idx], axis=-1)
        # 将得到的特征图缩放到输入图像的大小
        mask = cv2.resize(mask, (28, 28))
        
        img = np.squeeze(img)
        # 由于输入数据经过归一化，所以将其变换为原图进行显示
        plt.imshow((img * std + mean) * 255, cmap='gray')
        # 以热力图的形式将特征图覆盖到原图上进行显示
        plt.imshow(mask, alpha=0.4, cmap='jet')
        
        plt.show()
        