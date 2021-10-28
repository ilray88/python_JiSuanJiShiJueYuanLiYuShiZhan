import os
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

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

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(os.path.join(ckpt_folder, 'checkpoint-288.meta'))
    new_saver.restore(sess, ckpt_path)

    graph = tf.get_default_graph()

    # 获取模型中的占位符
    X = graph.get_operation_by_name('X').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]

    # 获取目标输出张量节点
    model_output = graph.get_tensor_by_name('resnet_18/classifier/fully_connected/Add:0')

    for _ in range(train_iter):
        batch_x, batch_y = mnist.next_batch('train')
        # 得到最终的卷积输出的结果
        outputs = sess.run(model_output, 
                           feed_dict={X: batch_x, is_training: False})

        for i in range(batch_size):
            idx = np.argmax(batch_y[i])
            output_cls[idx] += outputs[i]
        
H = 4
W = 3
fig, axes = plt.subplots(H, W)

for i in range(H):
    for j in range(W):
        if i * W + j < 10:
            # 以索引的形式取出每一个axes
            axes[i][j].bar(range(0, 10), output_cls[i * W + j])
            axes[i][j].set_title('[{}]'.format(i * W + j))
# 设置总图标题
plt.suptitle('Prediction Distribution')
plt.show()
        