import sys
sys.path.append('..')

import tensorflow.compat.v1 as tf
from models.base_class import Model
from activations import relu
from layers import conv2d, batch_normalization, fully_connected, max_pooling, avg_pooling, global_avg_pooling


class ResNet(Model):
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

    def __init__(self, depth, class_num, is_small):
        self.depth = depth
        
        # 图像类别数
        self.class_num = class_num

        # 输入是否为小图像
        self.is_small = is_small

        if self.depth <= 34:
            # 当模型深度小于34时使用Building Block
            self.block = self.building_block
        else:
            # 否则使用Bottleneck
            self.block = self.bottleneck
        
        # 根据传入的模型深度得到相应的结构
        self.structure = self.STRUCTURES[self.depth]

    # 使用的卷积核CONV_KERNEL大小默认为3，为编码灵活将其作为一个变量
    def building_block(self, x, out_channel, stride, name):
        with tf.variable_scope(name):
            with tf.variable_scope('sub_block1'):
                # 使用3*3的卷积
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
                # 使用3*3的卷积
                output = conv2d(output, 
                                out_channel, 
                                kernel=self.CONV_KERNEL, 
                                stride=1, 
                                padding='SAME', name='conv')
                output = batch_normalization(output, 
                                             name='bn', 
                                             is_training=self.is_training)
            
            with tf.variable_scope('shortcut'):
                # 判断是否需要重整形状（尺度减小，通道加倍）
                if stride != 1:
                    # 如果需要重整形状，使用1*1卷积得到期望形状
                    shortcut = conv2d(x, 
                                      out_channel, 
                                      kernel=1, 
                                      stride=stride, 
                                      padding='SAME', name='conv')
                    shortcut = batch_normalization(shortcut, 
                                                   name='bn', 
                                                   is_training=self.is_training)
                else:
                    # 不需要重整形状则直接使用输入张量x
                    shortcut = x
            
            # 将两路的输出按位加
            output = output + shortcut

            # 使用ReLU进行非线性激活
            output = relu(output, name='relu')
            
            return output

    # BOTTLENECK_CHANNEL_EXPANSION值为4
    def bottleneck(self, x, out_channel, stride, name):
        with tf.variable_scope(name):
            with tf.variable_scope('sub_block1'):
                # 第一个1*1的卷积层
                output = conv2d(x, 
                                out_channel, 
                                kernel=1, 
                                stride=1, 
                                padding='SAME', name='conv')
                output = batch_normalization(output,
                                                name='bn', 
                                                is_training=self.is_training)
                output = relu(output, name='relu')

            with tf.variable_scope('sub_block2'):
                # 第二个3*3的卷积层
                output = conv2d(output, 
                                out_channel, 
                                kernel=self.CONV_KERNEL, 
                                stride=stride, 
                                padding='SAME', name='conv')
                output = batch_normalization(output, 
                                                name='bn', 
                                                is_training=self.is_training)
                output = relu(output, name='relu')
            
            with tf.variable_scope('sub_block3'):
                # 第三个1*1的卷积层
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
            
            with tf.variable_scope('shortcut'):
                # 判断是否需要重整形状（尺度减小，通道加倍）
                if stride != 1 or c_in != c_out:
                    # 需要重整形状时，则使用1*1卷积改变其形状
                    shortcut = conv2d(x, 
                                    out_channel * \
                                        self.BOTTLENECK_CHANNEL_EXPANSION, 
                                    kernel=1, 
                                    stride=stride, 
                                    padding='SAME', name='conv')
                    shortcut = batch_normalization(shortcut, 
                                                    name='bn', 
                                                    is_training=self.is_training)
                else:
                    # 不需要重整形状则直接使用输入张量
                    shortcut = x
            
            # 将两路的输入按位相加
            output = output + shortcut

            # 使用ReLU进行非线性激活
            output = relu(output, name='relu')

            return output
    
    # BASIC_OUT_CHANNEL值为64
    def build(self, x, is_training):
        self.is_training = is_training
        with tf.variable_scope('resnet_{}'.format(self.depth)):
            with tf.variable_scope('preprocess_layers'):
                if self.is_small:
                    # 若输入为小图像,使用3*3卷积进行预处理
                    x = conv2d(x, 
                               self.BASIC_OUT_CHANNEL, 
                               kernel=3, 
                               stride=1, name='conv')
                    x = batch_normalization(x, 
                                            name='bn', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu')
                else:
                    # 若输入为大图像,使用7*7卷积+最大池化进行预处理
                    x = conv2d(x, 
                               self.BASIC_OUT_CHANNEL, 
                               kernel=7, 
                               stride=2, name='conv')
                    x = batch_normalization(x, 
                                            name='bn', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu')
                    x = max_pooling(x, 
                                    kernel=3,
                                    stride=2, 
                                    name='max_pool')
            
            # 每个ResNet分为4个阶段（stage），每个stage中包含不同数量的构块（block）
            for idx, st in enumerate(self.structure):
                # 当前阶段的构块输出通道数为64*2**i
                out_channel = self.BASIC_OUT_CHANNEL * 2 ** idx
                
                if idx == 0:
                    # 如果是第一个阶段，则不进行形状重整
                    first_stride = 1
                else:
                    # 否则阶段的第一个构块的卷积步长为2，进行形状重整
                    first_stride = 2

                # 除第一个构块步长需要单独考虑，剩下的步长都为1
                strides = [first_stride, *([1] * (st - 1))]

                for i, stride in zip(range(st), strides):
                    # 使用构块于其对应的步长和输出通道数处理输入数据x
                    x = self.block(x, 
                                   out_channel=out_channel, 
                                   stride=stride, 
                                   name='block_{}_{}'.format(idx, i))

            with tf.variable_scope('postprocess_layers'):
                # 使用全局池化层处理通过4个阶段提取到的特征
                x = global_avg_pooling(x, name='global_avg_pool')
            
            with tf.variable_scope('classifier'):
                # 最后分类器根据图像总类数使用全连接层
                x = fully_connected(x, self.class_num, name='fully_connected')
            
            return x

if __name__ == "__main__":
    from tools import Counter, VarsPrinter

    counter = Counter()
    printer = VarsPrinter()

    class_num = 10
    batch_size = None
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])

    for d in (18, 34, 50, 101, 152):
        model = ResNet(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=True)

        print((batch_size, class_num), output.shape)
        counter()
        printer()