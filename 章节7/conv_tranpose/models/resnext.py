import sys
sys.path.append('..')

import tensorflow.compat.v1 as tf
from models.base_class import Model
from activations import relu
from layers import conv2d, batch_normalization, fully_connected, max_pooling, avg_pooling, global_avg_pooling


class ResNeXt(Model):
    STRUCTURES = {
        29: {
            'block_nums': [3, 3, 3],
            'cardinality': 16
        }
    }

    FIRST_CONV_OUT_CHANNEL = 64
    BOTTLENECK_OUT_CHANNEL = 4
    BOTTLENECK_CHANNEL_EXPANSION = 4
    CONV_KERNEL = 3

    def __init__(self, depth, class_num, is_small):
        self.depth = depth
        self.class_num = class_num
        self.is_small = is_small

        self.structure = self.STRUCTURES[self.depth]['block_nums']
        self.cardinality = self.STRUCTURES[self.depth]['cardinality']


    def bottleneck(self, x, out_channel, stride, name):
        # 子模块2的实现函数
        def transform(x, out_channel, stride):
            # 第一步将输入的特征通过split分解成C组
            x_list = tf.split(x, 
                              num_or_size_splits=self.cardinality, 
                              axis=-1, name='split')
            # 计算每个分支卷积输出的通道数N/C
            out_channel = out_channel // self.cardinality

            # 对分解出的每一个子特征使用conv-BN-ReLU得到新特征
            # 并不断拼接得到通道数为N的特征
            for idx, x in enumerate(x_list):
                with tf.variable_scope('group_conv_{}'.format(idx)):
                    x = conv2d(x, 
                               out_channel, 
                               kernel=self.CONV_KERNEL, 
                               stride=stride, 
                               padding='SAME', name='conv')
                    x = batch_normalization(x, 
                                            name='bn', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu')

                    # 拼接子模块2的输出特征
                    out_x = x if idx == 0 else tf.concat([out_x, x], axis=-1)

            return out_x

        with tf.variable_scope(name):
            # 第一个子模块使用1*1卷积进行变换
            output = conv2d(x, 
                            out_channel, 
                            kernel=1, 
                            stride=1, 
                            padding='SAME', name='conv1')
            output = batch_normalization(output, 
                                         name='bn1', 
                                         is_training=self.is_training)
            output = relu(output, name='relu')

            # 第二个子模块使用split-transform-merge的方法进行变换
            with tf.variable_scope('transform'):
                output = transform(output, out_channel=out_channel, stride=stride)

            # 第三个子模块同样使用1*1的卷积进行变换
            output = conv2d(output, 
                            out_channel * self.BOTTLENECK_CHANNEL_EXPANSION, 
                            kernel=1, 
                            stride=1, 
                            padding='SAME', name='conv2')
            output = batch_normalization(output,
                                         name='bn2', 
                                         is_training=self.is_training)

            # 计算shortcut分支，与ResNet相同
            c_in = x.get_shape().as_list()[-1]
            c_out = out_channel * self.BOTTLENECK_CHANNEL_EXPANSION

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
            
            output = output + shortcut
            output = relu(output, name='relu')

            return output

    def build(self, x, is_training):
        self.is_training = is_training

        with tf.variable_scope(
                'resnext_{}_{}x{}d'.format(self.depth, 
                                        self.cardinality, 
                                        self.BOTTLENECK_OUT_CHANNEL)
            ):
            with tf.variable_scope('preprocess_layers'):
                x = conv2d(x, 
                            self.FIRST_CONV_OUT_CHANNEL, 
                            kernel=3, 
                            stride=1, 
                            padding='SAME', name='conv')
                x = batch_normalization(x, 
                                        name='bn', 
                                        is_training=self.is_training)
                x = relu(x, name='relu')
            
            for idx, st in enumerate(self.structure):
                out_channel = self.cardinality * self.BOTTLENECK_OUT_CHANNEL * 2 ** idx

                if idx == 0:
                    first_stride = 1
                else:
                    first_stride = 2

                strides = [first_stride, *([1] * (st - 1))]

                for i, stride in zip(range(st), strides):
                    x = self.bottleneck(x, 
                                        out_channel=out_channel, 
                                        stride=stride, 
                                        name='block_{}_{}'.format(idx, i))

            with tf.variable_scope('postprocess_layers'):
                if self.is_small:
                    x = global_avg_pooling(x, name='global_avg_pool')
                else:
                    x = avg_pooling(x, kernel=4, stride=4, name='avg_pool')
            
            with tf.variable_scope('classifier'):
                x = fully_connected(x, self.class_num, name='fully_connected')

            return x

if __name__ == "__main__":
    class_num = 10
    batch_size = None
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])

    for d in (29, ):
        model = ResNeXt(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=False)

        print((batch_size, class_num), output.shape)

    from tools import Counter
    counter = Counter()
    counter()
