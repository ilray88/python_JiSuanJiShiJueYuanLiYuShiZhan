import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from layers.fully_connected_layers import fully_connected
from activations.activations import leaky_relu, sigmoid
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
# 从标准正态分布采样数据的占位符
z = tf.placeholder(tf.float32, shape=[None, z_dim])

def encode(X):
    with tf.variable_scope('encoder') as scope:
        # 使用全连接层提取图像特征
        hidden = fully_connected(X, h_dim, name='fc1')
        # 引入非线性
        hidden = leaky_relu(hidden, 0.2, name='lrelu')

        # 通过特征学习正态分布的参数
        mu = fully_connected(hidden, z_dim, name='fc2')
        log_var = fully_connected(hidden, z_dim, name='fc3')

        return mu, log_var

def sample(mu, log_var):
    # 从标准正态分布中采样
    eps = tf.random_normal(shape=tf.shape(mu))

    # 将标准正太分布中的采样结果进行变换得到目标分布中的采样
    return mu + tf.sqrt(tf.exp(log_var)) * eps


def decode(Z):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
        hidden = fully_connected(Z, h_dim, name='fc1')
        hidden = leaky_relu(hidden, 0.2, name='lrelu')
        output = fully_connected(hidden, X_dim, name='fc2')
        
        return output

# 从编码器得到正态分布的参数值
mu, log_var = encode(X)
# 从目标分布中进行采样
z_sample = sample(mu, log_var)
# 由解码器将采样结果转换为图像数据
logits = decode(z_sample)

# 使用交叉熵计算输出与输入数据之间的差异
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# 正态分布的参数损失
mu_sigma_loss = 0.5 * tf.reduce_sum(tf.exp(log_var) + mu ** 2 - 1. - log_var, 1)

# 变分自编码器的损失由两部分构成
vae_loss = tf.reduce_mean(recon_loss + mu_sigma_loss)
# 使用优化器优化这一损失
step = tf.train.AdamOptimizer().minimize(vae_loss)

test_prob = sigmoid(decode(z), name='sigmoid')

print_net_info()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    train_iter = int(mnist.num_examples('train') / batch_size)

    for e in range(epoch):
        loss_e = 0
        for i in range(train_iter):
            x_data, _ = mnist.next_batch('train', reshape=False)
            x_data = x_data / 255.0
            _, loss = sess.run([step, vae_loss], feed_dict={X: x_data})
            loss_e += loss
        
        if e % 20 == 0:
            print('Epoch {}:\t{}'.format(e, loss_e))
            samples = sess.run(test_prob, 
                        feed_dict={z: np.random.randn(row * col, z_dim)})
            plot(samples, e)
