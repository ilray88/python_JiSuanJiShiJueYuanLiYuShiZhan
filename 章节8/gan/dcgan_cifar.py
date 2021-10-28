import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from layers.conv_layers import conv2d, conv2d_transpose
from layers.normalization_layers import batch_normalization
from activations.activations import leaky_relu, tanh, sigmoid
from data_utils.cifar import Cifar

from tools import print_net_info

from tqdm import tqdm

data_root = r'data path'
files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
batch_size = 16
generate_path = 'generated_img_cifar'

cifar = Cifar([os.path.join(data_root, x) for x in files], batch_size, normalize=False, augmentation=False)

z_dim = 128

epoch = 200

row = 4
col = 4

dis_learning_rate = 0.0002
gen_learning_rate = 0.0002

def plot(samples, iter):
    fig, axes = plt.subplots(row, col)
    for i in range(row):
        for j in range(col):
            axes[i][j].axis('off')
            axes[i][j].imshow(samples[i * row + j])

    plt.suptitle('Epoch {}'.format(iter))
    plt.savefig(os.path.join(generate_path, '{}.png'.format(iter)))


# 输入图像的占位符
X = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
# 从均匀分布采样数据的占位符
z = tf.placeholder(tf.float32, shape=[batch_size, 1, 1, z_dim])

def discriminator(X):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        # 16*16*64
        output = conv2d(X, 64, 3, 2, name='conv1')
        output = batch_normalization(output, name='bn1')
        output = leaky_relu(output, 0.2, name='lrelu1')

        # 8*8*128
        output = conv2d(output, 128, 3, 2, name='conv2')
        output = batch_normalization(output, name='bn2')
        output = leaky_relu(output, 0.2, name='lrelu2')

        # 4*4*256
        output = conv2d(output, 256, 3, 2, name='conv3')
        output = batch_normalization(output, name='bn3')
        output = leaky_relu(output, 0.2, name='lrelu3')

        # 2*2*512
        output = conv2d(output, 512, 3, 2, name='conv4')
        output = batch_normalization(output, name='bn4')
        output = leaky_relu(output, 0.2, name='lrelu4')

        # 2*2*1
        output = conv2d(output, 1, 1, 1, name='conv5')
        output = sigmoid(output, name='sigmoid')

        return output


def generator(Z):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        # 2*2*512
        output = conv2d_transpose(Z, 512, 2, 3, 2, name='conv_trans1')
        output = batch_normalization(output, name='bn1')
        output = leaky_relu(output, 0.2, name='lrelu1')

        # 4*4*256
        output = conv2d_transpose(output, 256, 4, 3, 2, name='conv_trans2')
        output = batch_normalization(output, name='bn2')
        output = leaky_relu(output, 0.2, name='lrelu2')

        # 8*8*64
        output = conv2d_transpose(output, 64, 8, 3, 2, name='conv_trans3')
        output = batch_normalization(output, name='bn3')
        output = leaky_relu(output, 0.2, name='lrelu3')

        # 16*16*32
        output = conv2d_transpose(output, 32, 16, 3, 2, name='conv_trans4')
        output = batch_normalization(output, name='bn4')
        output = leaky_relu(output, 0.2, name='lrelu4')

        # 32*32*3
        output = conv2d_transpose(output, 3, 32, 3, 2, name='conv_trans5')
        output = tanh(output, name='tanh')

        return output


# 由生成器将采样结果转换为图像数据
fake_img = generator(z)

real_prob = discriminator(X)
fake_prob = discriminator(fake_img)

# # 防止对0取对数造成Nan
# dis_loss = tf.reduce_mean(-tf.log(real_prob + 1e-6) - tf.log(1 - fake_prob + 1e-6))
# gen_loss = tf.reduce_mean(-tf.log(fake_prob + 1e-6))

# 使用LSGAN的损失形式
dis_loss = tf.reduce_mean(tf.squared_difference(real_prob, 1)) + tf.reduce_mean(tf.squared_difference(fake_prob, 0))
gen_loss = tf.reduce_mean(tf.squared_difference(fake_prob, 1))

dis_vars = list()
gen_vars = list()

# 根据变量命名空间分别取出判别器与生成器包含的变量
for v in tf.trainable_variables():
    if 'discriminator' in v.name:
        dis_vars.append(v)
    if 'generator' in v.name:
        gen_vars.append(v)

# 使用Adam优化判别器的变量
dis_step = tf.train.AdamOptimizer(dis_learning_rate).minimize(dis_loss, var_list=dis_vars)
# 使用Adam优化生成器的变量
gen_step = tf.train.AdamOptimizer(gen_learning_rate).minimize(gen_loss, var_list=gen_vars)

print_net_info()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    train_iter = int(cifar.num_examples('train') / batch_size)

    for e in range(epoch):
        dis_loss_e = 0
        gen_loss_e = 0

        for i in tqdm(range(train_iter), ncols=50):
            x_data, _ = cifar.next_batch('train')
            x_data = x_data / 127.5 - 1
            Z = np.random.uniform(-1, 1, size=[batch_size, 1, 1, z_dim])

            _, _, dis_loss_i, gen_loss_i = sess.run([dis_step, gen_step, dis_loss, gen_loss], feed_dict={X: x_data, z: Z})

            dis_loss_e += dis_loss_i
            gen_loss_e += gen_loss_i
        
        if e % 20 == 0:
            print('Epoch {}:\tdiscriminator: {}\tgenerator:{}'.format(e, dis_loss_e, gen_loss_e))

            samples = sess.run(fake_img, 
                        feed_dict={z: np.random.uniform(-1, 1, size=[row * col, 1, 1, z_dim])})

            samples = (samples + 1) / 2
            plot(samples, e)
