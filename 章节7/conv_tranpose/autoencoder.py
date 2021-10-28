import tensorflow.compat.v1 as tf
import cv2
import numpy as np
import os

from tqdm import tqdm
from data_utils.oxford_flower import OxfordFlower102
from losses.loss import Loss
from optimizers.optimizer import Optimizer
from layers.conv_layers.conv import conv2d, conv2d_transpose
from layers.pooling_layers import max_pooling
from layers.normalization_layers import batch_normalization
from activations.activations import leaky_relu, tanh, relu


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


img_size = (256, 256)

data_folder = r'data path'
folders = ['jpg', 'imagelabels.mat', 'setid.mat']
batch_size = 1
epoch = 100

# 获取数据的获取器
data = OxfordFlower102([os.path.join(data_folder, x) for x in folders], 
                       batch_size, 
                       normalize=False,
                       resize=img_size)

train_iter = int(data.num_examples('train') / batch_size)

# 定义模型输入输出的占位符
X = tf.placeholder(dtype=tf.float32, shape=[batch_size, *img_size, 3], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, *img_size, 3], name='Y')

# 定义神经网络模型
def autoEncoder(x):
    with tf.variable_scope('autoEncoder'):
        out_channel = 16

        output = x

        with tf.variable_scope('encoder'):
            # 编码部分
            for i in range(1):
                conv_out_channel = out_channel * 2 ** i

                output = conv2d(output, conv_out_channel, 3, 2, name='conv1{}'.format(i))
                output = batch_normalization(output, name='convbn1{}'.format(i))
                output = leaky_relu(output, 0.2, name='convrelu1{}'.format(i))

                output = conv2d(output, conv_out_channel, 3, 1, name='conv2{}'.format(i))
                output = batch_normalization(output, name='convbn2{}'.format(i))
                output = leaky_relu(output, 0.2, name='convrelu2{}'.format(i))


        with tf.variable_scope('decoder'):
            # 解码部分
            for i in reversed(range(1)):
                conv_trans_out_channel = out_channel * 2 ** i
                out_size = img_size[0] // (2 ** i)

                # 使用转置卷积层完成上采样
                output = conv2d_transpose(output,
                                        conv_trans_out_channel,
                                        out_size,
                                        3,
                                        2, 
                                        name='conv_transpose1{}'.format(i))

                output = batch_normalization(output, name='conv_trspbn1{}'.format(i))
                output = leaky_relu(output, 0.2, name='conv_trsrelu1{}'.format(i))

                output = conv2d(output, conv_out_channel, 3, 1, name='conv_2{}'.format(i))
                output = batch_normalization(output, name='convbn_2{}'.format(i))
                output = leaky_relu(output, 0.2, name='convrelu_2{}'.format(i))
            
        # 最后的卷积层使输出通道数为3
        output = conv2d(output, 3, 3, 1, name='convlast')

        # 由于输入的归一化图像取值范围是[-1,1]
        # 因此使用tanh将输出值压缩到[-1,1]
        output = tanh(output, name='tanh')
        return output

pred = autoEncoder(X)

# 定义损失函数类型
loss_type = 'mae'
loss = Loss(loss_type).get_loss(Y, pred)

optim = Optimizer(0.05, [50, 80, 90], 0.2, None, None, name='momentum').minimize(loss)
saver = tf.train.Saver(max_to_keep=5)


with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    
    for e in range(epoch):
        loss_e = 0

        for j in tqdm(range(train_iter), ncols=50):
            # 取出的图像为彩色图像,同时作为训练数据与标签使用
            img_data, _ = data.next_batch('train')
            img_data = img_data / 127.5 - 1

            loss_i, _ = sess.run([loss, optim], feed_dict={X: img_data, Y: img_data})
            loss_e += loss_i

        # 保存模型参数
        saver.save(sess, "checkpoint/autoenoder/checkpoint.ckpt", global_step=e)
        # 打印每个周期的损失值
        print(loss_e)

    test_X, _ = data.next_batch('test')
    test_X = test_X / 127.5 - 1
    restore_img = sess.run(pred, feed_dict={X: test_X})

    cv2.imshow("restore", np.uint8((restore_img[0] + 1) * 127.5))
    cv2.imshow("raw", np.uint8((test_X[0] + 1) * 127.5))
    cv2.waitKey(0)