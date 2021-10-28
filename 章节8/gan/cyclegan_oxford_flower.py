import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from layers.conv_layers import conv2d, conv2d_transpose
from layers.normalization_layers import batch_normalization
from activations.activations import leaky_relu, tanh, sigmoid
from data_utils.oxford_flower import OxfordFlower102

from tools import print_net_info

from tqdm import tqdm

batch_size = 16
generate_path = 'generated_img_cyclegan'

oxford_flower_root = r'data path'
oxford_flower_files = ['jpg','imagelabels.mat','setid.mat']

oxford_flower = OxfordFlower102([os.path.join(oxford_flower_root, x) for x in oxford_flower_files], batch_size, normalize=False, augmentation=False, resize=(128, 128))

epoch = 5000

row = 2
col = 2

dis_learning_rate = 0.0003
gen_learning_rate = 0.0001

def plot(samples, iter, name):
    fig, axes = plt.subplots(row, col)
    for i in range(row):
        for j in range(col):
            axes[i][j].axis('off')
            axes[i][j].imshow(samples[i * row + j])

    plt.suptitle('Epoch {}'.format(iter))
    plt.savefig(os.path.join(generate_path, '{}_{}.png'.format(name, iter)))


# 输入图像的占位符
oxford_flower_X = tf.placeholder(tf.float32, shape=[batch_size, 128, 128, 3])
oxford_flower_X_gray = tf.placeholder(tf.float32, shape=[batch_size, 128, 128, 3])

def discriminator_a2b(X):
    with tf.variable_scope('discriminator_a2b', reuse=tf.AUTO_REUSE) as scope:
        # 64*64*64
        output = conv2d(X, 64, 3, 2, name='conv1')
        output = batch_normalization(output, name='bn1')
        output = leaky_relu(output, 0.2, name='lrelu1')

        # 32*32*128
        output = conv2d(output, 128, 3, 2, name='conv2')
        output = batch_normalization(output, name='bn2')
        output = leaky_relu(output, 0.2, name='lrelu2')

        # 16*16*256
        output = conv2d(output, 256, 3, 2, name='conv3')
        output = batch_normalization(output, name='bn3')
        output = leaky_relu(output, 0.2, name='lrelu3')

        # 8*8*512
        output = conv2d(output, 512, 3, 2, name='conv4')
        output = batch_normalization(output, name='bn4')
        output = leaky_relu(output, 0.2, name='lrelu4')

        # 4*4*1
        output = conv2d(output, 1, 3, 2, name='conv5')
        output = sigmoid(output, name='sigmoid')

        return output

def discriminator_b2a(X):
    with tf.variable_scope('discriminator_b2a', reuse=tf.AUTO_REUSE) as scope:
        # 64*64*64
        output = conv2d(X, 64, 3, 2, name='conv1')
        output = batch_normalization(output, name='bn1')
        output = leaky_relu(output, 0.2, name='lrelu1')

        # 32*32*128
        output = conv2d(output, 128, 3, 2, name='conv2')
        output = batch_normalization(output, name='bn2')
        output = leaky_relu(output, 0.2, name='lrelu2')

        # 16*16*256
        output = conv2d(output, 256, 3, 2, name='conv3')
        output = batch_normalization(output, name='bn3')
        output = leaky_relu(output, 0.2, name='lrelu3')

        # 8*8*512
        output = conv2d(output, 512, 3, 2, name='conv4')
        output = batch_normalization(output, name='bn4')
        output = leaky_relu(output, 0.2, name='lrelu4')

        # 4*4*1
        output = conv2d(output, 1, 3, 2, name='conv5')
        output = sigmoid(output, name='sigmoid')

        return output


def generator_a2b(X):
    with tf.variable_scope('generator_a2b', reuse=tf.AUTO_REUSE) as scope:
        # 64*64*64
        output = conv2d(X, 64, 3, 2, name='conv1')
        output = batch_normalization(output, name='bn1')
        output = leaky_relu(output, 0.2, name='lrelu1')

        # 32*32*128
        output = conv2d(output, 128, 3, 2, name='conv2')
        output = batch_normalization(output, name='bn2')
        output = leaky_relu(output, 0.2, name='lrelu2')

        # 64*64*64
        output = conv2d_transpose(output, 64, 64, 3, 2, name='conv_trans3')
        output = batch_normalization(output, name='bn3')
        output = leaky_relu(output, 0.2, name='lrelu3')

        # 128*128*32
        output = conv2d_transpose(output, 32, 128, 3, 2, name='conv_trans4')
        output = batch_normalization(output, name='bn4')
        output = leaky_relu(output, 0.2, name='lrelu4')

        # 128*128*3
        output = conv2d(output, 3, 3, 1, name='conv_trans5')
        output = tanh(output, name='tanh')

        return output

def generator_b2a(X):
    with tf.variable_scope('generator_b2a', reuse=tf.AUTO_REUSE) as scope:
        # 64*64*64
        output = conv2d(X, 64, 3, 2, name='conv1')
        output = batch_normalization(output, name='bn1')
        output = leaky_relu(output, 0.2, name='lrelu1')

        # 32*32*128
        output = conv2d(output, 128, 3, 2, name='conv2')
        output = batch_normalization(output, name='bn2')
        output = leaky_relu(output, 0.2, name='lrelu2')

        # 64*64*64
        output = conv2d_transpose(output, 64, 64, 3, 2, name='conv_trans3')
        output = batch_normalization(output, name='bn3')
        output = leaky_relu(output, 0.2, name='lrelu3')

        # 128*128*32
        output = conv2d_transpose(output, 32, 128, 3, 2, name='conv_trans4')
        output = batch_normalization(output, name='bn4')
        output = leaky_relu(output, 0.2, name='lrelu4')

        # 128*128*3
        output = conv2d(output, 3, 3, 1, name='conv_trans5')
        output = tanh(output, name='tanh')

        return output


# 将彩色图像转换为黑白图像
img_gray = generator_a2b(oxford_flower_X)
# 将黑白图像转换为彩色图像
img_color = generator_b2a(oxford_flower_X_gray)

# 由生成的黑白图像重新转换为彩色图像
consist_color = generator_b2a(img_gray)
# 由生成的彩色图像重新转换为黑白图像
consist_gray = generator_a2b(img_color)

# 真实黑白图像的判别结果
real_prob_gray = discriminator_a2b(oxford_flower_X_gray)
# 生成黑白图像的判别结果
fake_prob_gray = discriminator_a2b(img_gray)

# 真实彩色图像的判别结果
real_prob_color = discriminator_b2a(oxford_flower_X)
# 虚假彩色图像的判别结果
fake_prob_color = discriminator_b2a(img_color)


# 连续经过两个生成器的结果应该相同
consist_loss = tf.reduce_mean(tf.squared_difference(consist_color, oxford_flower_X)) + \
                    tf.reduce_mean(tf.squared_difference(consist_gray, oxford_flower_X_gray))

# # 防止对0取对数造成Nan
# dis_loss = tf.reduce_mean(-tf.log(real_prob + 1e-6) - tf.log(1 - fake_prob + 1e-6))
# gen_loss = tf.reduce_mean(-tf.log(fake_prob + 1e-6))

# 使用LSGAN的损失形式
dis_a2b_loss = tf.reduce_mean(tf.squared_difference(real_prob_gray, 1)) + tf.reduce_mean(tf.squared_difference(fake_prob_gray, 0))
gen_a2b_loss = tf.reduce_mean(tf.squared_difference(fake_prob_gray, 1)) + 0.1 * consist_loss

dis_b2a_loss = tf.reduce_mean(tf.squared_difference(real_prob_color, 1)) + tf.reduce_mean(tf.squared_difference(fake_prob_color, 0))
gen_b2a_loss = tf.reduce_mean(tf.squared_difference(fake_prob_color, 1)) + 0.1 * consist_loss


dis_a2b_vars = list()
gen_a2b_vars = list()

dis_b2a_vars = list()
gen_b2a_vars = list()

# 根据变量命名空间分别取出判别器与生成器包含的变量
for v in tf.trainable_variables():
    if 'discriminator' in v.name and 'a2b' in v.name:
        dis_a2b_vars.append(v)
    if 'generator' in v.name and 'a2b' in v.name:
        gen_a2b_vars.append(v)

    if 'discriminator' in v.name and 'b2a' in v.name:
        dis_b2a_vars.append(v)
    if 'generator' in v.name and 'b2a' in v.name:
        gen_b2a_vars.append(v)

# 使用Adam优化判别器的变量
dis_a2b_step = tf.train.AdamOptimizer(dis_learning_rate).minimize(dis_a2b_loss, var_list=dis_a2b_vars)
# 使用Adam优化生成器的变量
gen_a2b_step = tf.train.AdamOptimizer(gen_learning_rate).minimize(gen_a2b_loss, var_list=gen_a2b_vars)

# 使用Adam优化判别器的变量
dis_b2a_step = tf.train.AdamOptimizer(dis_learning_rate).minimize(dis_b2a_loss, var_list=dis_b2a_vars)
# 使用Adam优化生成器的变量
gen_b2a_step = tf.train.AdamOptimizer(gen_learning_rate).minimize(gen_b2a_loss, var_list=gen_b2a_vars)

print_net_info()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    train_iter = int(oxford_flower.num_examples('train') / batch_size)

    for e in range(epoch):
        dis_a2b_loss_e = 0
        gen_a2b_loss_e = 0

        dis_b2a_loss_e = 0
        gen_b2a_loss_e = 0

        for i in tqdm(range(train_iter), ncols=50):
            # 取出一个batch的数据作为彩色图像
            oxford_flower_data_color, _ = oxford_flower.next_batch('train')
            # 取出另一个batch的数据，将其转换为黑白图像
            oxford_flower_data, _ = oxford_flower.next_batch('train')

            oxford_flower_data_gray = list()

            for i in range(batch_size):
                # 将彩色图像转换为黑白图像
                img = cv2.cvtColor(oxford_flower_data[i], cv2.COLOR_BGR2GRAY)
                # 将黑白图像转换为三通道的黑白图像
                img = np.stack([img] * 3, axis=-1)
                oxford_flower_data_gray.append(img)
            
            oxford_flower_data_gray = np.stack(oxford_flower_data_gray, axis=0)
            
            # 对彩色图像和黑白图像进行归一化
            oxford_flower_data_color = oxford_flower_data_color / 127.5 - 1
            oxford_flower_data_gray = oxford_flower_data_gray / 127.5 - 1

            # 先优化判别器
            _, _, dis_a2b_loss_i, dis_b2a_loss_i = \
                sess.run([dis_a2b_step, dis_b2a_step, dis_a2b_loss, dis_b2a_loss], feed_dict={oxford_flower_X: oxford_flower_data_color, oxford_flower_X_gray: oxford_flower_data_gray})
            
            # 再优化生成器
            _, _, gen_a2b_loss_i, gen_b2a_loss_i = \
                sess.run([gen_a2b_step, gen_b2a_step, gen_a2b_loss, gen_b2a_loss], feed_dict={oxford_flower_X: oxford_flower_data_color, oxford_flower_X_gray: oxford_flower_data_gray})

            dis_a2b_loss_e += dis_a2b_loss_i
            gen_a2b_loss_e += gen_a2b_loss_i

            dis_b2a_loss_e += dis_b2a_loss_i
            gen_b2a_loss_e += gen_b2a_loss_i

        if e % 20 == 0:
            print('Epoch {}:\tdiscriminator_a2b: {}\tgenerator_a2b:{}\tdiscriminator_b2a: {}\tgenerator_b2a:{}'.format(e, dis_a2b_loss_e, gen_a2b_loss_e, dis_b2a_loss_e, gen_b2a_loss_e))

            samples = sess.run([img_gray, img_color, consist_gray, consist_color], 
                        feed_dict={oxford_flower_X: oxford_flower_data_color, oxford_flower_X_gray: oxford_flower_data_gray})

            names = ['img_gray', 'img_color', 'consist_gray', 'consist_color']

            for name, sample in zip(names, samples):
                # 由于OpenCV读出的图像通道顺序为BGR，为显示正常将其转换为RGB
                sample = sample[:,:,::-1]
                sample = (sample + 1) / 2
                plot(sample, e, name)