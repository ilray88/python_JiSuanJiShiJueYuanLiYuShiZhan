import sys
sys.path.append('..')

from models.base_class import Model
from activations import relu
from layers import conv2d, batch_normalization, fully_connected, max_pooling, avg_pooling, global_avg_pooling
import tensorflow.compat.v1 as tf


class DenseNet(Model):
    STRUCTURES = {
        121: {
            'block_nums': [6, 12, 24, 16],
            'growth_rate': 32,
            'reduction': 0.5
        },
        169: {
            'block_nums': [6, 12, 32, 32],
            'growth_rate': 32,
            'reduction': 0.5
        },
        201: {
            'block_nums': [6, 12, 48, 32],
            'growth_rate': 32,
            'reduction': 0.5
        },
        264: {
            'block_nums': [6, 12, 64, 48],
            'growth_rate': 32,
            'reduction': 0.5
        }
    }

    DEFAULT_GROWTH_RATE = 12
    DEFAULT_REDUCTION = 0.5

    def __init__(self, depth, class_num, is_small):
        self.depth = depth
        self.class_num = class_num
        self.is_small = is_small

        # 若网络深度不为121/169/201/264，则采取默认值进行配置
        if self.depth not in self.STRUCTURES.keys():
            N = int((self.depth - 4) / 3)

            self.structure = [N] * 3

            # 默认值为12
            self.growth_rate = self.DEFAULT_GROWTH_RATE
            self.nChannels = 2 * self.growth_rate

            # 默认值为0.5
            self.reduction = self.DEFAULT_REDUCTION           
        else:
            self.structure = self.STRUCTURES[self.depth]['block_nums']
            self.growth_rate = self.STRUCTURES[self.depth]['growth_rate']
            self.nChannels = 2 * self.growth_rate
            self.reduction = self.STRUCTURES[self.depth]['reduction']

    def transition(self, x, out_channel, name):
        with tf.variable_scope(name):
            x = batch_normalization(x, name='bn', is_training=self.is_training)
            x = relu(x, name='relu')

            # 使用1*1卷积对通道进行减少
            x = conv2d(x, out_channel, kernel=1, stride=1, padding='SAME')
            # 使用卷积层对空间尺度进行减少
            x = avg_pooling(x, kernel=2, stride=2, name='avg_pool')

            return x
    
    def bottleneck(self, x, name):
        with tf.variable_scope(name):
            # 按照BN-ReLU-Conv的顺序进行处理
            x = batch_normalization(x, 
                                    name='bn1', 
                                    is_training=self.is_training)
            x = relu(x, name='relu2')
            # 使用1*1卷积，并且输出的通道数为4k
            x = conv2d(x, 
                    4 * self.growth_rate, 
                    kernel=1, 
                    stride=1, 
                    padding='SAME', name='conv1')

            x = batch_normalization(x, 
                                    name='bn2', 
                                    is_training=self.is_training)
            x = relu(x, name='relu2')
            # 使用3*3卷积，输出的通道数为k
            x = conv2d(x, 
                    self.growth_rate, 
                    kernel=3, 
                    stride=1, 
                    padding='SAME', name='conv2')

            return x
    
    def dense_block(self, x, N, name):
        with tf.variable_scope(name):
            for idx in range(N):
                output = self.bottleneck(x, name='bottleneck_{}'.format(idx))
                
                # 将每次的输出作为下一次的输入
                x = tf.concat([x, output], axis=-1, name='dense_connect')

            return x
    
    def build(self, x, is_training):
        self.is_training = is_training
        with tf.variable_scope('densenet_{}'.format(self.depth)):
            with tf.variable_scope('preprocess_layers'):
                if self.depth not in self.STRUCTURES.keys() or self.is_small:                    
                    # 对于小图像/默认的预处理层
                    x = conv2d(x, 
                            self.nChannels, 
                            kernel=3, 
                            stride=1, 
                            padding='SAME', name='conv')
                else:
                    # DenseNet原论文中使用的预处理层
                    x = conv2d(x, 
                            self.nChannels, 
                            kernel=7, 
                            stride=2, 
                            padding='SAME', name='conv')
                    x = batch_normalization(x, 
                                            name='bn', 
                                            is_training=self.is_training)
                    x = relu(x, name='relu')
                    x = max_pooling(x, kernel=3, stride=2, name='max_pool')

            
            for idx, st in enumerate(self.structure):
                x = self.dense_block(x, st, name='block_{}'.format(idx))

                # 计算过渡层的输出通道数
                out_channel = int(x.get_shape().as_list()[-1] * self.reduction)
                
                # 最后一个Dense Block后不使用过渡层
                if idx != len(self.structure):
                    x = self.transition(x, 
                                        out_channel, 
                                        name='transition_{}'.format(idx))
            
            with tf.variable_scope('postprocess_layers'):
                x = batch_normalization(x,
                                        name='bn', 
                                        is_training=self.is_training)
                x = relu(x, name='relu')
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

    for d in (11, 121, 169, 201, 264):
        model = DenseNet(d, class_num=class_num, is_small=False)
        output = model.build(image, is_training=False)

        print((batch_size, class_num), output.shape)
        
        printer()
        counter()
    
