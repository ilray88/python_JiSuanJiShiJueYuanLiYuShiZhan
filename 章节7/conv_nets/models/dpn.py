import sys
sys.path.append('..')

from models.base_class import Model
from activations import relu
from layers import conv2d, batch_normalization, fully_connected, max_pooling, avg_pooling, global_avg_pooling
import tensorflow.compat.v1 as tf


class DPN(Model):
    STRUCTURES = {
        92: {
            # 第一层卷积输出通道数
            'first_conv_channel': 64,
            # Bottleneck重复数
            'block_nums': [3, 4, 20, 3],
            # bottleneck中第1/2层输出通道数，第3层输出通道数
            'block_channels': [(96, 256), (192, 512), (384, 1024), (768, 2048)],
            # densenet分支输出通道数
            'dense_depth': [16, 32, 24, 128],
            # 分组卷积的组数G
            'cardinality': 32
        },
        98: {
            'first_conv_channel': 96,
            'block_nums': [3, 6, 20, 3],
            'block_channels': [(160, 256), (320, 512), (640, 1024), (1280, 2048)],
            'dense_depth': [16, 32, 32, 128],
            'cardinality': 40
        }
    }

    CONV_KERNEL = 3


    def __init__(self, depth, class_num, is_small):
        self.depth = depth
        self.st_dict = self.STRUCTURES[self.depth]
        self.cardinality = self.st_dict['cardinality']
        self.class_num = class_num
        self.is_small = is_small

    def bottleneck(self, 
                   x, 
                   channel_num12, 
                   channel_num3, 
                   stride, 
                   dense_depth, 
                   is_first, 
                   name):
        # 与ResNeXt的分组卷积写法一致
        def transform(x, out_channel, stride):
            x_list = tf.split(x, 
                              num_or_size_splits=self.cardinality, 
                              axis=-1, name='split')
            out_channel = out_channel // self.cardinality
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
                    out_x = x if idx == 0 else tf.concat([out_x, x], axis=-1)
            
            return out_x

        with tf.variable_scope(name):
            # 第1个1*1卷积层
            output = conv2d(x, 
                            channel_num12, 
                            kernel=1, 
                            stride=1, 
                            padding='SAME', name='conv1')
            output = batch_normalization(output,
                                         name='bn1', 
                                         is_training=self.is_training)
            output = relu(output, name='relu')

            # 第2个3*3卷积层
            with tf.variable_scope('transform'):
                output = transform(output, channel_num12, stride)
            
            # 第3个1*1卷积层
            output = conv2d(output, 
                            channel_num3 + dense_depth, 
                            kernel=1, 
                            stride=1, 
                            padding='SAME', name='conv2')
            output = batch_normalization(output, 
                                         name='bn2', 
                                         is_training=self.is_training)
            
            # 输出前部分为ResNeXt分支的特征
            # 后部分为DenseNet分支的特征
            res_out = output[..., :channel_num3]
            dense_out = output[..., channel_num3:]
            
            # 判断是不是每一个block中的第1个bottleneck
            with tf.variable_scope('shortcut'):
                # 若是第1个bottleneck则需要使用1*1卷积将形状重整
                if is_first:
                    shortcut = conv2d(x, 
                                      channel_num3 + dense_depth, 
                                      kernel=1, 
                                      stride=stride, 
                                      padding='SAME', name='conv')
                    shortcut = batch_normalization(shortcut, 
                                                   name='bn', 
                                                   is_training=self.is_training)
                else:
                    shortcut = x
            
            res_x = shortcut[..., :channel_num3]
            dense_x = shortcut[..., channel_num3:]
            
            # ResNeXt分支使用按位加 
            res_part = res_x + res_out
            # DenseNet分支使用通道拼接
            dense_part = tf.concat([dense_x, dense_out], axis=-1)

            # 最终的输出使用通道拼接进行连接
            output = tf.concat([res_part, dense_part], axis=-1)

            # 为输出特征进行非线性变换
            output = relu(output, name='relu')
            
        return output

    def build(self, x, is_training):
        self.is_training = is_training

        with tf.variable_scope('dpn_{}'.format(self.depth)):
            # 预处理层
            with tf.variable_scope('preprocess_layers'):
                # 对小图像的预处理
                if self.is_small:
                    x = conv2d(x, 
                               self.st_dict['first_conv_channel'], 
                               kernel=3, 
                               stride=1, name='conv')
                    x = batch_normalization(x, 
                                            name='bn', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu')
                else:
                    x = conv2d(x, 
                               self.st_dict['first_conv_channel'], 
                               kernel=7, 
                               stride=2, name='conv')
                    x = batch_normalization(x, 
                                            name='bn', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu')
                    x = max_pooling(x, 
                                    kernel=3, stride=2, name='max_pool')
            
            for idx, (block_nums, out_channels, dense_depth) in \
                             enumerate(zip(self.st_dict['block_nums'], 
                                           self.st_dict['block_channels'], 
                                           self.st_dict['dense_depth'])):
                out_channel12 = out_channels[0]
                out_channel3 = out_channels[1]
                
                if idx == 0:
                    first_stride = 1
                else:
                    first_stride = 2

                strides = [first_stride, *([1] * (block_nums - 1))]

                # 反复堆叠Bottleneck
                for i, stride in zip(range(block_nums), strides):
                    x = self.bottleneck(x, 
                                        out_channel12, 
                                        out_channel3, 
                                        stride, 
                                        dense_depth, 
                                        is_first=i == 0, 
                                        name='block_{}_{}'.format(idx, i))
                
            with tf.variable_scope('postprocess_layers'):
                x = global_avg_pooling(x, name='global_avg_pool')
            
            with tf.variable_scope('classifier'):
                x = fully_connected(x, self.class_num, name='fully_connected')
            
            return x

if __name__ == "__main__":
    from tools import Counter, VarsPrinter

    class_num = 10
    batch_size = None
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])

    counter = Counter()
    printer = VarsPrinter()

    for d in (92, 98):
        model = DPN(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=False)

        print((batch_size, class_num), output.shape)

        printer()
        counter()