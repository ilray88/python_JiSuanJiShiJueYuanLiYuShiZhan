import tensorflow.compat.v1 as tf
import cv2
import numpy as np
import os

from tqdm import tqdm
from data_utils.oxford_flower import OxfordFlower102
from losses.loss import Loss
from optimizers.optimizer import Optimizer
from layers.conv_layers.conv import conv2d, conv2d_transpose
from layers.normalization_layers import batch_normalization
from activations.activations import leaky_relu, tanh, relu


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# 大图像与小图像的大小
large_img_size = (256, 256)
small_img_size = (64, 64)

data_folder = r'data path'
folders = ['jpg', 'imagelabels.mat', 'setid.mat']
batch_size = 1
epoch = 100

# 获取数据的获取器
data = OxfordFlower102([os.path.join(data_folder, x) for x in folders], 
                       batch_size, 
                       normalize=False,
                       resize=large_img_size)

train_iter = int(data.num_examples('train') / batch_size)

# 定义模型输入输出的占位符
X = tf.placeholder(dtype=tf.float32, shape=[batch_size, *small_img_size, 3], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, *large_img_size, 3], name='Y')

# 定义神经网络模型
def defineNetwork(x):
    with tf.variable_scope('super_resolution'):
        # 连续使用卷积层获取图像的特征
        for i in range(5):
            output = conv2d(output, 64, 3, 1, name='conv{}'.format(i))
            output = batch_normalization(output, name='convbn{}'.format(i))
            output = leaky_relu(output, 0.2, name='convrelu{}'.format(i))

        # 通过两层转置卷积层将输入特征的尺寸进行放大
        out_channel = 16
        for i in reversed(range(2)):
            output = conv2d_transpose(output,
                                      out_channel * 2 ** i,
                                      large_img_size[0] // (2 ** i),
                                      3,
                                      2, 
                                      name='conv_transpose{}'.format(i))
            output = batch_normalization(output, name='conv_trspbn{}'.format(i))
            output = leaky_relu(output, 0.2, name='conv_trsrelu{}'.format(i))
        
        # 最后的卷积层使输出通道数为3
        output = conv2d(output, 3, 3, 1, name='convlast')

        # 由于输入的归一化图像取值范围是[-1,1]
        # 因此使用tanh将输出值压缩到[-1,1]
        output = tanh(output, name='tanh')
        return output

pred = defineNetwork(X)

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
            # 取出的图像为高分辨率图像,作为标签使用
            batch_Y, _ = data.next_batch('train')
            # 对图像进行归一化
            batch_Y = batch_Y / 127.5 - 1

            batch_X = np.empty([batch_size, *small_img_size, 3])

            for i in range(batch_size):
                # 将高分辨率图像缩小得到低分辨率图像
                # 作为模型的输入使用
                batch_X[i] = cv2.resize(batch_Y[i], small_img_size)

            loss_i, _ = sess.run([loss, optim], feed_dict={X: batch_X, Y: batch_Y})
            loss_e += loss_i
        # 保存模型参数
        saver.save(sess, "checkpoint/super_resolution/checkpoint.ckpt", global_step=e)
        # 打印每个周期的损失值
        print(loss_e)

    tmp_test_X, _ = data.next_batch('test')
    test_X = np.empty([batch_size, *small_img_size, 3])
    for i in range(batch_size):
        test_X[i] = cv2.resize(tmp_test_X[i], small_img_size)
    
    test_X = test_X / 127.5 - 1
    sr_img = sess.run(pred, feed_dict={X: test_X})
    
    cv2.imshow("super resolution", np.uint8((sr_img[0] + 1) * 127.5))
    cv2.imshow("high resolution", tmp_test_X[0])
    cv2.waitKey(0)