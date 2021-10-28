import sys
sys.path.append('..')

from models.base_class import Model
from activations import relu
from layers import conv2d, batch_normalization, fully_connected, max_pooling, avg_pooling, global_avg_pooling
import tensorflow.compat.v1 as tf

class SKResNeXt(Model):
    STRUCTURES = {
        29: {
            'block_nums': [3, 3, 3],
            'cardinality': 16
        }
    }

    FIRST_CONV_OUT_CHANNEL =  64
    BOTTLENECK_OUT_CHANNEL = 4
    BOTTLENECK_CHANNEL_EXPANSION = 4
    CONV_KERNEL = 3
    REDUCTION = 16
    L = 32
    M = [3, 1]

    def __init__(self, depth, class_num, is_small):
        self.depth = depth
        self.class_num = class_num
        self.is_small = is_small

        self.structure = self.STRUCTURES[self.depth]['block_nums']
        self.cardinality = self.STRUCTURES[self.depth]['cardinality']


    def sk_module(self, x, out_channel, stride, name):
        with tf.variable_scope(name):
            u_list = list()
            with tf.variable_scope('split'):
                # 对同一输入使用不同大小的卷积核完成卷积
                for idx, k in enumerate(self.M):
                    u_list.append(
                        conv2d(x, 
                               out_channel, 
                               kernel=k, 
                               stride=stride, 
                               padding='SAME', 
                               name='conv_{}'.format(idx))
                    )
                # 将所有不同尺度的特征连接成一个张量
                # u_list形状为[num_fea, batch_size, H', W', C_out]
                u_list = tf.stack(u_list, axis=0)

            with tf.variable_scope('fuse'):
                # 把所有不同尺度的特征按位加得到融合特征
                # u的形状为[batch_size, H', W', C_out]
                u = tf.reduce_sum(u_list, axis=0, name='sum')

                # 将融合的特征进行压缩
                s = global_avg_pooling(u, name='global_avg_pool')
                
                # 计算全连接层输出的特征数
                fc_out = max(out_channel // self.REDUCTION, self.L)
                z = fully_connected(s, 
                                    fc_out, 
                                    name='fully_connected')
                z = relu(z, name='relu')

            attention_list = list()
            with tf.variable_scope('select'):
                # 对每一个不同尺度的特征计算相应的权重
                for idx in range(len(self.M)):
                    attention_list.append(
                        fully_connected(z, 
                                        out_channel, 
                                        name='fully_connected_{}'.format(idx))
                    )
                # 将所有不同尺度特征的权重连接成一个张量
                # attention_list形状为[num_fea, batch_size, C_out]
                attention_list = tf.stack(attention_list, axis=0)

                # 不同尺度特征的权重之间使用Softmax互相抑制
                attention_list = tf.math.softmax(attention_list, 
                                                 axis=0, 
                                                 name='softmax')

                # 为了方便按位乘，将其形状变为[num_fea, batch_size, 1, 1, C_out]
                attention_list = tf.expand_dims(attention_list, axis=2)
                attention_list = tf.expand_dims(attention_list, axis=2)

                # 使用特征与其对应的权重按位乘
                # output形状为[num_fea, batch_size, H', W', C_out]
                output = u_list * attention_list

                # 将按位乘的结果按位加进行融合
                # output形状为[batch_size, H', W', C_out]
                output = tf.reduce_sum(output, axis=0, name='merge')

            return output


    def bottleneck(self, x, out_channel, stride, name):
        def transform(x, out_channel, stride):
            x_list = tf.split(x, 
                              num_or_size_splits=self.cardinality, 
                              axis=-1, name='split')
            out_channel = out_channel // self.cardinality

            # 将ResNeXt中的Bottleneck的分组卷积操作全部替换为SK模块
            for idx, x in enumerate(x_list):
                with tf.variable_scope('group_conv_{}'.format(idx)):
                    x = self.sk_module(x, 
                                       out_channel, 
                                       stride=stride, 
                                       name='sk_module_{}'.format(idx))
                    x = batch_normalization(x, 
                                            name='bn', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu')
                    out_x = x if idx == 0 else tf.concat([out_x, x], axis=-1)
            
            return out_x

        with tf.variable_scope(name):
            # 第1个1*1卷积
            output = conv2d(x, 
                            out_channel, 
                            kernel=1, 
                            stride=1, name='conv1')
            output = batch_normalization(output, 
                                         name='bn1', 
                                         is_training=self.is_training)
            output = relu(output, name='relu')

            # SK模块
            with tf.variable_scope('transform'):
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

        with tf.variable_scope('skresnext_{}_{}x{}d'.format(self.depth, self.cardinality, self.BOTTLENECK_OUT_CHANNEL)):
            with tf.variable_scope('preprocess_layers'):
                x = conv2d(x, self.FIRST_CONV_OUT_CHANNEL, kernel=3, stride=1, padding='SAME', name='conv')
                x = batch_normalization(x, name='bn', is_training=self.is_training)
                x = relu(x, name='relu')
            
            for idx, st in enumerate(self.structure):
                out_channel = self.cardinality * self.BOTTLENECK_OUT_CHANNEL * 2 ** idx

                if idx == 0:
                    first_stride = 1
                else:
                    first_stride = 2

                strides = [first_stride, *([1] * (st - 1))]

                for i, stride in zip(range(st), strides):
                    x = self.bottleneck(x, out_channel=out_channel, stride=stride, name='block_{}_{}'.format(idx, i))

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
    
    # model = SKResNeXt(29, class_num=class_num, is_small=False)
        
    # sk_out = model.sk_module(image, 4096, name='test_sk')
    # print(sk_out)
    # exit(0)
    
    for d in (29, ):
        model = SKResNeXt(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=False)

        print((batch_size, class_num), output.shape)

    from tools import print_net_info
    print_net_info()
