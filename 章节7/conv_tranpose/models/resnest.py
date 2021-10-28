import sys
sys.path.append('..')

import tensorflow.compat.v1 as tf
from models.base_class import Model
from activations import relu
from layers import conv2d, batch_normalization, fully_connected, max_pooling, avg_pooling, global_avg_pooling


class ResNeSt(Model):
    STRUCTURES = {
        29: {
            'block_nums': [3, 3, 3],
            'cardinality': 1,
            'radix': 2
        },
        50: {
            'block_nums': [3, 4, 6, 3],
            'cardinality': 1,
            'radix': 2
        }
    }

    FIRST_CONV_OUT_CHANNEL = 64
    BOTTLENECK_OUT_CHANNEL = 64
    BOTTLENECK_CHANNEL_EXPANSION = 4
    CONV_KERNEL = 3
    REDUCTION = 16
    L = 32


    def __init__(self, depth, class_num, is_small):
        self.depth = depth
        self.class_num = class_num
        self.is_small = is_small

        self.structure = self.STRUCTURES[self.depth]
        self.cardinality = self.structure['cardinality']
        self.radix = self.structure['radix']

    def bottleneck(self, x, out_channel, stride, name):
        def split_attention(x, out_channel, stride):
            # 对分组后的特征进行3*3卷积（先对输出通道扩充）
            # 形状为[batch_size, H', W', C'*radix]
            x = conv2d(x, 
                       out_channel * self.radix, 
                       kernel=self.CONV_KERNEL, 
                       stride=stride, 
                       padding='SAME', name='conv')
            x = batch_normalization(x, 
                                    name='bn1', 
                                    is_training=self.is_training)
            x = relu(x, name='relu1')

            # 将处理后的特征进行再一次分组（Split-Attention的分组）
            x = tf.split(x, 
                         num_or_size_splits=self.radix, 
                         axis=-1, name='split')
            # 形状为[radix, batch_size, H', W', C']
            x = tf.stack(x, axis=0)

            # 对分完组的特征按位相加
            # 形状为[batch_size, H', W', C']
            x_sum = tf.reduce_sum(x, axis=0)
            # 使用全剧平均池化进行特征的压缩
            # 形状为[batch_size, C']
            x_gap = global_avg_pooling(x_sum, name='global_avg_pool')

            # 计算全连接层输出的特征数（可参见SKResNeXt中的实现）
            fc_out = max(out_channel // self.REDUCTION, self.L)

            # 将压缩后的特征使用类似SK模块中的方式减小维度
            x_gap = fully_connected(x_gap, 
                                    out_num=fc_out, 
                                    name='fc1')
            x_gap = batch_normalization(x_gap, 
                                        name='bn2', 
                                        is_training=self.is_training)
            x_gap = relu(x_gap, name='relu2')

            # 将压缩后的特征变换回原输出通道维度
            # 形状为[batch_size, C'*radix]
            attention = fully_connected(x_gap, out_num=out_channel * self.radix)
            
            # 将输出张量分为radix个张量
            attention = tf.split(attention, 
                                 num_or_size_splits=self.radix, 
                                 axis=-1, name='split_atten')
            # 形状为[radix, batch_size, C']
            attention = tf.stack(attention, axis=0)

            # 对attention张量按radix做softmax
            attention = tf.math.softmax(attention, axis=0)

            # 将attention形状变为[radix, batch_size, 1, 1, C']
            attention = tf.expand_dims(attention, axis=2)
            attention = tf.expand_dims(attention, axis=2)

            # 使用attention与对应的输入张量对应相乘
            x = attention * x
            x = tf.reduce_sum(x, axis=0)

            return x

        def transform(x, out_channel, stride):
            # 将输入张量先分为K组
            x_list = tf.split(x, 
                              num_or_size_splits=self.cardinality, 
                              axis=-1, name='split')
            # 每一组卷积输出的通道数
            out_channel = out_channel // self.cardinality

            for idx, x in enumerate(x_list):
                with tf.variable_scope('group_conv_{}'.format(idx)):
                    x = split_attention(x, out_channel, stride=stride)
                    # 拼接输出张量
                    out_x = x if idx == 0 else tf.concat([out_x, x], axis=-1)

            return out_x

        with tf.variable_scope(name):
            # 第1个1*1卷积
            output = conv2d(x, 
                            out_channel, 
                            kernel=1, 
                            stride=1, 
                            padding='SAME', name='conv1')
            output = batch_normalization(output, 
                                         name='bn1', 
                                         is_training=self.is_training)
            output = relu(output, name='relu')

            # 第2个3*3分组卷积 + Split-Attention
            with tf.variable_scope('split_attention'):
                output = transform(output, 
                                   out_channel=out_channel, 
                                   stride=stride)

            # 第3个1*1卷积
            output = conv2d(output, 
                            out_channel * self.BOTTLENECK_CHANNEL_EXPANSION, 
                            kernel=1, 
                            stride=1, 
                            padding='SAME', name='conv2')
            output = batch_normalization(output, 
                                         name='bn2', 
                                         is_training=self.is_training)

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

        with tf.variable_scope('resnest_{}_{}x{}d'.format(self.depth, self.cardinality, self.FIRST_CONV_OUT_CHANNEL)):
            with tf.variable_scope('preprocess_layers'):
                if self.is_small:
                    x = conv2d(x, 
                               self.FIRST_CONV_OUT_CHANNEL, 
                               kernel=self.CONV_KERNEL, 
                               stride=1, 
                               padding='SAME', name='conv')
                    x = batch_normalization(x, 
                                            name='bn', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu')
                else:
                    x = conv2d(x, 
                               self.FIRST_CONV_OUT_CHANNEL, 
                               kernel=self.CONV_KERNEL, 
                               stride=2, 
                               padding='SAME', name='conv1')
                    x = batch_normalization(x, 
                                            name='bn1', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu1')

                    x = conv2d(x, 
                               self.FIRST_CONV_OUT_CHANNEL, 
                               kernel=self.CONV_KERNEL, 
                               stride=1, 
                               padding='SAME', name='conv2')
                    x = batch_normalization(x, 
                                            name='bn2', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu2')

                    x = conv2d(x, 
                               self.FIRST_CONV_OUT_CHANNEL * 2, 
                               kernel=self.CONV_KERNEL, 
                               stride=1, 
                               padding='SAME', name='conv3')
                    x = batch_normalization(x, 
                                            name='bn3', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu3')
                    x = max_pooling(x, kernel=3, stride=2, name='avg_pool')
            
            for idx, st in enumerate(self.structure['block_nums']):
                out_channel = self.BOTTLENECK_OUT_CHANNEL * 2 ** idx

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
    from tools import Counter
    class_num = 10
    batch_size = None
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])

    for d in (29, 50):
        model = ResNeSt(d, class_num=class_num, is_small=True)
        output = model.build(image, is_training=False)

        print((batch_size, class_num), output.shape)

        counter = Counter()
        counter()
