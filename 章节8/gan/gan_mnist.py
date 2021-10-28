import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from layers.fully_connected_layers import fully_connected
from activations.activations import leaky_relu, tanh, sigmoid
from data_utils.mnist import Mnist

from tools import print_net_info


data_root = r'data path'
files = ['t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 'train-images.idx3-ubyte', 'train-labels.idx1-ubyte']
batch_size = 16
generate_path = 'generated_img_mnist'

mnist = Mnist([os.path.join(data_root, x) for x in files], batch_size, normalize=False)

z_dim = 128
X_dim = 784
h_dim = 512

epoch = 200

row = 4
col = 4

dis_learning_rate = 0.0003
gen_learning_rate = 0.0001

def plot(samples, iter):
    fig, axes = plt.subplots(row, col)
    for i in range(row):
        for j in range(col):
            axes[i][j].axis('off')
            axes[i][j].imshow(samples[i * row + j].reshape(28, 28), cmap='gray')

    plt.suptitle('Epoch {}'.format(iter))
    plt.savefig(os.path.join(generate_path, '{}.png'.format(iter)))


# 输入图像的占位符
X = tf.placeholder(tf.float32, shape=[None, X_dim])
# 从均匀分布采样数据的占位符
z = tf.placeholder(tf.float32, shape=[None, z_dim])

def discriminator(X):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        # 使用全连接层提取图像特征
        output = fully_connected(X, h_dim, name='fc1')
        # 引入非线性
        output = leaky_relu(output, 0.2, name='lrelu1')

        output = fully_connected(output, z_dim, name='fc2')
        output = leaky_relu(output, 0.2, name='lrelu2')

        output = fully_connected(output, 1, name='fc3')
        # 将输出转换为一个0~1之间的概率值
        output = sigmoid(output, name='sigmoid')

        return output

def generator(Z):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        hidden = fully_connected(Z, h_dim, name='fc1')
        hidden = leaky_relu(hidden, 0.2, name='lrelu')
        output = fully_connected(hidden, X_dim, name='fc2')
        output = tanh(output, name='tanh')
        return output

# 由生成器将采样结果转换为图像数据
fake_img = generator(z)

real_prob = discriminator(X)
fake_prob = discriminator(fake_img)

# 防止对0取对数造成Nan
dis_loss = tf.reduce_mean(-tf.log(real_prob + 1e-6) - tf.log(1 - fake_prob + 1e-6))
gen_loss = tf.reduce_mean(-tf.log(fake_prob + 1e-6))

dis_vars = list()
gen_vars = list()

# 根据变量命名空间分别取出判别器与生成器包含的变量
for v in tf.trainable_variables():
    if 'discriminator' in v.name:
        dis_vars.append(v)
    if 'generator' in v.name:
        gen_vars.append(v)

# 使用不同的优化器优化两者的损失，以尽量平衡模型的性能
# 使用SGD优化判别器的变量
dis_step = tf.train.GradientDescentOptimizer(dis_learning_rate).minimize(dis_loss, var_list=dis_vars)
# 使用Adam优化生成器的变量
gen_step = tf.train.AdamOptimizer(gen_learning_rate).minimize(gen_loss, var_list=gen_vars)

print_net_info()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    train_iter = int(mnist.num_examples('train') / batch_size)

    for e in range(epoch):
        dis_loss_e = 0
        gen_loss_e = 0

        for i in range(train_iter):
            x_data, _ = mnist.next_batch('train', reshape=False)
            x_data = x_data / 127.5 - 1
            Z = np.random.uniform(-1, 1, size=[batch_size, z_dim])

            _, _, dis_loss_i, gen_loss_i = sess.run([dis_step, gen_step, dis_loss, gen_loss], feed_dict={X: x_data, z: Z})

            dis_loss_e += dis_loss_i
            gen_loss_e += gen_loss_i
        
        if e % 20 == 0:
            print('Epoch {}:\tdiscriminator: {}\tgenerator:{}'.format(e, dis_loss_e, gen_loss_e))

            samples = sess.run(fake_img, 
                        feed_dict={z: np.random.uniform(-1, 1, size=[row * col, z_dim])})

            samples = (samples + 1) / 2
            plot(samples, e)
