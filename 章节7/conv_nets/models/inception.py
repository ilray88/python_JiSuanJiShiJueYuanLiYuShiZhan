import sys
sys.path.append('..')

from models.base_class import Model
from activations import relu
from layers import conv2d, batch_normalization, fully_connected, max_pooling, avg_pooling, global_avg_pooling
import tensorflow.compat.v1 as tf


class Inception(Model):
    def __init__(self, version, class_num, is_small):
        self.version = 1
        self.class_num = class_num
        self.is_small = is_small
    
    # Inception模块需要指定1*1卷积输出通道数
    # 3*3卷积中的1*1卷积输出通道数和最终输出通道数
    # 5*5卷积中的1*1卷积输出通道数和最终输出通道数
    # 最大池化输出通道数
    def inception_module(self, 
                         x, 
                         _1x1_oc, 
                         _3x3_roc, 
                         _3x3_oc,
                         _5x5_roc,
                         _5x5_oc,
                         pool_oc,
                         name):
        with tf.variable_scope(name):
            # 1*1卷积的分支
            with tf.variable_scope('1x1_conv'):
                _1x1_conv = conv2d(x, 
                                   _1x1_oc, 
                                   kernel=1, 
                                   stride=1, name='conv')
                _1x1_conv = batch_normalization(_1x1_conv, 
                                                 name='bn', 
                                                 is_training=self.is_training)
                _1x1_conv = relu(_1x1_conv, name='relu')

            # 3*3卷积的分支，包含1个1*1卷积和1个3*3卷积
            with tf.variable_scope('3x3_conv'):
                # 1*1卷积进行降维
                _3x3_conv = conv2d(x, 
                                   _3x3_roc, 
                                   kernel=1, 
                                   stride=1, name='1x1')
                _3x3_conv = batch_normalization(_3x3_conv, 
                                                name='bn1', 
                                                is_training=self.is_training)
                _3x3_conv = relu(_3x3_conv, name='relu1')

                # 为降维后的结果使用3*3卷积
                _3x3_conv = conv2d(_3x3_conv, 
                                   _3x3_oc, 
                                   kernel=3, 
                                   stride=1, name='3x3')
                _3x3_conv = batch_normalization(_3x3_conv, 
                                                name='bn2', 
                                                is_training=self.is_training)
                _3x3_conv = relu(_3x3_conv, name='relu2')

            # 5*5卷积的分支，包含1个1*1卷积和1个5*5卷积
            with tf.variable_scope('5x5_conv'):
                # 1*1卷积进行降维
                _5x5_conv = conv2d(x, 
                                   _5x5_roc, 
                                   kernel=1, 
                                   stride=1, name='1x1')
                _5x5_conv = batch_normalization(_5x5_conv, 
                                                name='bn1', 
                                                is_training=self.is_training)
                _5x5_conv = relu(_5x5_conv, name='relu1')

                # 为降维后的结果使用5*5卷积
                _5x5_conv = conv2d(x, 
                                   _5x5_oc, 
                                   kernel=5, 
                                   stride=1, name='5x5')
                _5x5_conv = batch_normalization(_5x5_conv, 
                                                name='bn2', 
                                                is_training=self.is_training)
                _5x5_conv = relu(_5x5_conv, name='relu2')
            
            # 最大池化分支，使用3*3的窗口，包含1个最大池化层和1个1*1卷积
            with tf.variable_scope('max_pool'):
                _mp = max_pooling(x, 3, stride=1, name='pool')
                _mp = conv2d(_mp, 
                             pool_oc, 
                             kernel=1, 
                             stride=1, name='1x1')
                _mp = batch_normalization(_mp, 
                                          name='bn', 
                                          is_training=self.is_training)
                _mp = relu(_mp, name='bn')
            
            # 将4个分支的结果在通道维度上进行拼接
            output = tf.concat([_1x1_conv, _3x3_conv, _5x5_conv, _mp], axis=-1)

            return output

    def build(self, x, is_training):
        self.is_training = is_training

        with tf.variable_scope('inception_{}'.format(self.version)):
            # 数据预处理层包含一个3*3卷积与可选的池化层
            with tf.variable_scope('preprocess_layers'):
                x = conv2d(x, 
                           192, 
                           kernel=3, 
                           stride=1, name='conv')
                x = batch_normalization(x, 
                                        name='bn', 
                                        is_training=self.is_training)
                x = relu(x, name='relu')

                if not self.is_small:
                    x = max_pooling(x, 
                                    3, 
                                    stride=2, name='max_pool')

            # 第一个大的Inception模块
            with tf.variable_scope('inception3'):
                x = self.inception_module(x, 
                                          64, 96, 128, 
                                          16, 32, 32, name='inception_3a')
                x = self.inception_module(x,
                                          128, 128, 192, 
                                          32, 96, 64, name='inception_3b')
                x = max_pooling(x, 3, stride=2, name='max_pool')

            # 第二个大的Inception模块
            with tf.variable_scope('inception4'):
                x = self.inception_module(x,
                                          192, 96, 208, 
                                          16, 48, 64, name='inception_4a')
                x = self.inception_module(x,
                                          160, 112, 224, 
                                          24, 64, 64, name='inception_4b')
                x = self.inception_module(x,
                                          128, 128, 256, 
                                          24, 64, 64, name='inception_4c')
                x = self.inception_module(x, 
                                          112, 144, 288, 
                                          32, 64, 64, name='inception_4d')
                x = self.inception_module(x,
                                          256, 160, 320, 
                                          32, 128, 128, name='inception_4e')
                x = max_pooling(x, 3, stride=2, name='max_pool')

            # 第三个大的Inception模块
            with tf.variable_scope('inception5'):
                x = self.inception_module(x, 
                                          256, 160, 320, 
                                          32, 128, 128, name='inception_5a')
                x = self.inception_module(x, 
                                          384, 192, 384,
                                           48, 128, 128, name='inception_5b')

                # 根据输入尺度使用不同的池化层
                if self.is_small:
                    x = global_avg_pooling(x, name='global_avg_pool')
                else:
                    x = avg_pooling(x, 3, stride=2, name='max_pool')

            # 使用Dropout随机丢弃神经元
            # 训练阶段随机保留40%的神经元，测试阶段使用100%的神经元
            with tf.variable_scope('dropout'):
                keep_prob = tf.cond(
                                tf.cast(is_training, tf.bool), 
                                lambda: 0.4, lambda: 1.0
                            )
                x = tf.nn.dropout(x, keep_prob)
            
            # 最后的全连接输出层
            with tf.variable_scope('classifier'):
                x = fully_connected(x, self.class_num)
            
            return x
        

if __name__ == "__main__":
    from tools import Counter, VarsPrinter
    
    counter = Counter()
    printer = VarsPrinter()

    class_num = 10
    batch_size = None
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])
    

    for d in (1, ):
        model = Inception(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=False)

        print((batch_size, class_num), output.shape)

        printer()
        counter()