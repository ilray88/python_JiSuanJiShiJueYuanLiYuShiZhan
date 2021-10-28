import sys
sys.path.append('..')

import tensorflow.compat.v1 as tf
from models.base_class import Model
from activations import relu, sigmoid
from layers import conv2d, batch_normalization, fully_connected, max_pooling, avg_pooling, global_avg_pooling


class SEResNet(Model):
    STRUCTURES = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    CONV_KERNEL = 3
    BASIC_OUT_CHANNEL = 64
    BOTTLENECK_CHANNEL_EXPANSION = 4
    REDUCTION = 16

    def __init__(self, depth, class_num, is_small):
        self.depth = depth
        self.structure = self.STRUCTURES[self.depth]
        self.class_num = class_num
        self.is_small = is_small

        if self.depth <= 34:
            self.block = self.building_block
        else:
            self.block = self.bottleneck

    def se_module(self, x, name):
        with tf.variable_scope(name):
            with tf.variable_scope('squeeze'):
                # 使用全局平均池化进行压缩操作
                se_output = global_avg_pooling(x, name='global_avg_pool')

            with tf.variable_scope('excitation'):
                # 原始通道数C
                ori_channel_num = se_output.get_shape().as_list()[-1]
                
                # 通过全连接层将其压缩到C/r个数
                se_output = fully_connected(se_output, 
                                            ori_channel_num // self.REDUCTION, 
                                            name='fc1')
                se_output = relu(se_output, name='relu')

                # 再次通过全连接层将其恢复至C个数
                se_output = fully_connected(se_output, 
                                            ori_channel_num, 
                                            name='fc2')
                # 使用Sigmoid函数将其值转换为0~1的数
                se_output = sigmoid(se_output, 'sigmoid')
                # 为了方便直接相乘，将激励值转换为4维张量
                se_output = tf.reshape(se_output, [-1, 1, 1, ori_channel_num])
            
            with tf.variable_scope('scale'):
                # 将激励值与输入张量相乘作为输出   
                x = se_output * x

            return x

    def building_block(self, x, out_channel, stride, name):
        with tf.variable_scope(name):
            with tf.variable_scope('sub_block1'):
                output = conv2d(x, 
                                out_channel, 
                                kernel=self.CONV_KERNEL, 
                                stride=stride, 
                                padding='SAME', name='conv')
                output = batch_normalization(output, 
                                             name='bn', 
                                             is_training=self.is_training)
                output = relu(output, name='relu')

            with tf.variable_scope('sub_block2'):
                output = conv2d(output, 
                                out_channel, 
                                kernel=self.CONV_KERNEL, 
                                stride=1, 
                                padding='SAME', name='conv')
                output = batch_normalization(output, 
                                             name='bn', 
                                             is_training=self.is_training)
            
            with tf.variable_scope('shortcut'):
                if stride != 1:
                    shortcut = conv2d(x, 
                                      out_channel, 
                                      kernel=1, 
                                      stride=stride, 
                                      padding='SAME', name='conv')
                    shortcut = batch_normalization(shortcut, 
                                                   name='bn', 
                                                   is_training=self.is_training)
                else:
                    shortcut = x
            
            output = self.se_module(output, 'se_module')

            output = output + shortcut
            output = relu(output, name='relu')
            
            return output

    def bottleneck(self, x, out_channel, stride, name):
        with tf.variable_scope(name):
            # 第1个1*1卷积
            with tf.variable_scope('sub_block1'):
                output = conv2d(x, 
                                out_channel, 
                                kernel=1, 
                                stride=1, 
                                padding='SAME', name='conv')
                output = batch_normalization(output, 
                                             name='bn', 
                                             is_training=self.is_training)
                output = relu(output, name='relu')

            # 第2个3*3卷积
            with tf.variable_scope('sub_block2'):
                output = conv2d(output, 
                                out_channel, 
                                kernel=self.CONV_KERNEL, 
                                stride=stride, 
                                padding='SAME', name='conv')
                output = batch_normalization(output, 
                                             name='bn', 
                                             is_training=self.is_training)
                output = relu(output, name='relu')
            
            # 第3个1*1卷积
            with tf.variable_scope('sub_block3'):
                output = conv2d(output, 
                                out_channel * self.BOTTLENECK_CHANNEL_EXPANSION, 
                                kernel=1, 
                                stride=1, 
                                padding='SAME', name='conv')
                output = batch_normalization(output, 
                                             name='bn', 
                                             is_training=self.is_training)

            c_in = x.get_shape().as_list()[-1]
            c_out = out_channel * self.BOTTLENECK_CHANNEL_EXPANSION

            # 短路通路
            with tf.variable_scope('shortcut'):
                if stride != 1 or c_in != c_out:
                    shortcut = conv2d(x,
                                      out_channel * self.BOTTLENECK_CHANNEL_EXPANSION, 
                                      kernel=1, 
                                      stride=stride, 
                                      padding='SAME', name='conv')
                    shortcut = batch_normalization(shortcut, 
                                                   name='bn', 
                                                   is_training=self.is_training)
                else:
                    shortcut = x
            
            # 对最终相加前的特征使用SE激励
            output = self.se_module(output, 'se_module')
            
            output = output + shortcut
            output = relu(output, name='relu')

            return output

    def build(self, x, is_training):
        self.is_training = is_training

        with tf.variable_scope('se-resnet_{}'.format(self.depth)):
            with tf.variable_scope('preprocess_layers'):
                if self.is_small:
                    x = conv2d(x, self.BASIC_OUT_CHANNEL, kernel=3, stride=1, name='conv')
                    x = batch_normalization(x, name='bn', is_training=self.is_training)
                    x = relu(x, name='relu')
                else:
                    x = conv2d(x, self.BASIC_OUT_CHANNEL, kernel=7, stride=2, name='conv')
                    x = batch_normalization(x, name='bn', is_training=self.is_training)
                    x = relu(x, name='relu')
                    x = max_pooling(x, kernel=3, stride=2, name='max_pool')
            
            for idx, st in enumerate(self.structure):
                out_channel = self.BASIC_OUT_CHANNEL * 2 ** idx
                
                if idx == 0:
                    first_stride = 1
                else:
                    first_stride = 2

                strides = [first_stride, *([1] * (st - 1))]

                for i, stride in zip(range(st), strides):
                    x = self.block(x, out_channel=out_channel, stride=stride, name='block_{}_{}'.format(idx, i))

            with tf.variable_scope('postprocess_layers'):
                if self.is_small:
                    x = global_avg_pooling(x, name='global_avg_pool')
                else:
                    x = avg_pooling(x, kernel=4, stride=4, name='avg_pool')
            
            with tf.variable_scope('classifier'):
                x = fully_connected(x, self.class_num, name='fully_connected')
            
            return x

if __name__ == "__main__":
    from tools import Counter

    class_num = 10
    batch_size = None
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])

    for d in (18, 34, 50, 101, 152):
        model = SEResNet(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=False)

        print((batch_size, class_num), output.shape)

        counter = Counter()
        counter()