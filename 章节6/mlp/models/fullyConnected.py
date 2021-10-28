import tensorflow.compat.v1 as tf

from models.base_class import Model
from layers.fc import fully_connected
from activations.activations import sigmoid, leaky_relu


class FullyConnected(Model):
    # 为构造函数传入含有整数的list表示每一层（除输入层）含有的神经元个数
    def __init__(self, structure):
        self.structure = structure

    # 定义具体模型结构
    # 模型中未使用BN，因此可以不使用is_training参数
    def build(self, x):
        with tf.variable_scope('fc') as scope:
            # 定义输入层到第一层隐含层的全连接结构与激活函数
            output = fully_connected(x, self.structure[0], name='fc0')
            # output = sigmoid(output, name='sigmoid0')

            output = leaky_relu(output, a=0.2, name='lrelu0')

            # 定义第二层隐含层到输入层的结构
            for i in range(1, len(self.structure)):
                output = fully_connected(
                            output, self.structure[i], name='fc{}'.format(i)
                         )
                # 为了编码灵活性，最后一层不使用激活函数，可以根据特定任务选用激活函数
                if i != len(self.structure) - 1:
                    # 选用sigmoid函数作为激活函数
                    # output = sigmoid(output, name='sigmoid{}'.format(i))
                    output = leaky_relu(output, a=0.2, name='lrelu{}'.format(i))

            return output
