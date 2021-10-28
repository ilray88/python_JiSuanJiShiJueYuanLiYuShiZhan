import sys
sys.path.append('..')

from models.base_class import Model
from activations import relu
from layers import conv2d, batch_normalization, fully_connected, max_pooling

import tensorflow.compat.v1 as tf


class VGG(Model):
    MAX_POOL = 'mp'
    STRUCTURES = {
        11: [64, MAX_POOL, 128, MAX_POOL, 256, 256, 
             MAX_POOL, 512, 512, MAX_POOL, 512, 512, MAX_POOL],

        13: [64, 64, MAX_POOL, 128, 128, MAX_POOL, 
             256, 256, MAX_POOL, 512, 512, MAX_POOL, 512, 512, MAX_POOL],

        16: [64, 64, MAX_POOL, 128, 128, MAX_POOL, 
             256, 256, 256, MAX_POOL, 512, 512, 512, MAX_POOL, 512, 512, 512, MAX_POOL],

        19: [64, 64, MAX_POOL, 128, 128, MAX_POOL, 
             256, 256, 256, 256, MAX_POOL, 512, 512, 512, 512, MAX_POOL, 
             512, 512, 512, 512, MAX_POOL],
    }

    # 为编码灵活，将卷积核大小作为变量给出
    CONV_KERNEL = 3

    # 传入所需网络深度、数据类别数
    # 以及是否是小图数据集（保持模型类构造函数的参数一致性）
    def __init__(self, depth, class_num, is_small):
        self.depth = depth
        self.class_num = class_num

        # 我们可以对小图数据集做一些定制化操作
        self.is_small = is_small
        self.structure = self.STRUCTURES[self.depth]

    # 传入网络的输入x与当前网络的阶段（BN使用）
    def build(self, x, is_training):
        with tf.variable_scope('vgg_{}'.format(self.depth)):
            # 从网络结构中每一层参数进行构建
            for idx, st in enumerate(self.structure):
                # 当前为最大池化层
                if self.MAX_POOL == st:
                    x = max_pooling(x,
                                    kernel=2,
                                    stride=2,
                                    name='max_pooling_{}'.format(idx))
                else:
                    # 当前为卷积-BN-激活层
                    x = conv2d(x,
                               out_channel=st,
                               kernel=self.CONV_KERNEL,
                               stride=1,
                               name='conv2d_{}'.format(idx))

                    x = batch_normalization(x, 
                                            name='bn_{}'.format(idx), 
                                            is_training=is_training)
                    
                    x = relu(x, name='relu_{}'.format(idx))

            # 最后的全连接层输出与类别数相同维度的张量
            x = fully_connected(x, self.class_num,
                                name='fully_connected_{}'.format(idx + 1))

            return x


if __name__ == "__main__":
    from tools import Counter, VarsPrinter
    counter = Counter()
    printer = VarsPrinter()

    # 假设类别数为10，batch_size未知，输入图像为32*32*3
    class_num = 10
    batch_size = None
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])

    # 测试不同的深度
    for d in (11, 13, 16, 19):
        model = VGG(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=False)
        # 打印模型的输出形状
        print(output.shape)
        # 打印模型中的参数信息并计算参数总量
        printer()
        counter()
