import sys
sys.path.append('..')

import tensorflow.compat.v1 as tf

from activations import relu
from base_class import Model
from layers import conv2d, batch_normalization, depthwise_conv2d, global_avg_pooling, avg_pooling, fully_connected


class MobileNet(Model):
    # 每一个元组中表示（步长，输出通道数）
    STRUCTURES = {
        1: [
            (1, 64), (2, 128), (1, 128), (2, 256), 
            (1, 256), (2, 512), (1, 512), (1, 512), 
            (1, 512), (1, 512), (1, 512), (2, 1024), (2, 1024)
        ]
    }

    CONV_KERNEL = 3
    CHANNEL_MULTIPLIER = 1

    def __init__(self, version, class_num, is_small):
        self.version = 1
        self.class_num = class_num
        self.is_small = is_small
        self.structure = self.STRUCTURES[self.version]


    def build(self, x, is_training):
        self.is_training = is_training
        with tf.variable_scope('mobilenet_v{}'.format(self.version)):
            # 预处理层
            with tf.variable_scope('preprocess_layers'):
                x = conv2d(x, 
                           32, 
                           kernel=self.CONV_KERNEL, 
                           stride=2, name='conv')
                x = batch_normalization(x, 
                                        name='bn', 
                                        is_training=self.is_training)
                x = relu(x, name='relu')

            with tf.variable_scope('mobilenet_blocks'):
                for idx, (stride, out_channel) in enumerate(self.structure):
                    # 对每一对输入的步长与输出通道数组合分别输入到卷积中使用
                    with tf.variable_scope('block{}'.format(idx)):
                        # 深度卷积使用输入的步长参数
                        x = depthwise_conv2d(x, 
                                             self.CONV_KERNEL, 
                                             channel_multiplier=self.CHANNEL_MULTIPLIER, 
                                             stride=stride, name='depthwise_conv')

                        x = batch_normalization(x, 
                                                name='bn1', 
                                                is_training=self.is_training)
                        x = relu(x, name='relu1')
                        
                        # 使用普通1*1卷积实现逐点卷积
                        # 为逐点卷积输入输出通道数参数
                        x = conv2d(x, 
                                   out_channel, 
                                   kernel=1, 
                                   stride=1, name='pointwise_conv')
                        x = batch_normalization(x,
                                                name='bn2', 
                                                is_training=self.is_training)
                        x = relu(x, name='relu2')
            
            # 根据是否使小图像数据集选用不同的池化方法
            with tf.variable_scope('postprocess_layers'):
                if self.is_small:
                    x = global_avg_pooling(x, name='global_avg_pool')
                else:
                    x = avg_pooling(x, kernel=7, stride=1, name='avg_pool')
            
            # 最终的全连接分类函数
            with tf.variable_scope('classifier'):
                x = fully_connected(x, self.class_num, name='fully_connected')

            return x

if __name__ == "__main__":
    from tools import Counter

    class_num = 10
    batch_size = None
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])

    for d in (1, ):
        model = MobileNet(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=False)

        print((batch_size, class_num), output.shape)

        counter = Counter()
        counter()
    
