import tensorflow.compat.v1 as tf

class Loss:
    @staticmethod
    def squared_difference(label, pred, name='mse'):
        with tf.variable_scope(name) as scope:
            output = tf.reduce_mean(tf.squared_difference(label, pred))
            return output

    @staticmethod
    def absolute_difference(label, pred, name='abs'):
        with tf.variable_scope(name) as scope:
            # 计算预测与真值差（pred-label）的绝对值，再计算绝对值的均值
            output = tf.reduce_mean(tf.abs(label - pred))
            return output

    @staticmethod
    def huber_loss(label, pred, name='huber'):
        with tf.variable_scope(name) as scope:
            output = tf.losses.huber_loss(label, pred)
            return output

    @staticmethod
    def softmax_ce(label, pred, name='softmax_ce'):
        with tf.variable_scope(name) as scope:
            # 将预测值通过softmax变换为0~1概率值
            pred = tf.nn.softmax(pred)
            # 计算预测值的以2为底的对数值
            pred = tf.math.log(pred) / tf.math.log(2.0)
            # 计算预测值与真值对应位置的熵
            output = -label * pred
            # 对每个样本而言，将每个位置上求得的熵进行求和
            # 得到的形状为[batch_size, 1]的张量
            output = tf.reduce_sum(output, axis=-1)
            # 计算整个batch上熵的均值
            output = tf.reduce_mean(output)
            
            return output

    @staticmethod
    def hinge_loss(label, pred, name='hinge'):
        with tf.variable_scope(name) as scope:
            output = tf.losses.hinge_loss(label, pred)
            return output

    @staticmethod
    def kl_div(label, pred, name='kl'):
        with tf.variable_scope(name) as scope:
            # 计算真值的熵与真值和预测值的交叉熵
            entro = label * tf.math.log(label + 1e-10) / tf.math.log(2.0)
            entro = tf.reduce_sum(entro, axis=-1)
            entro = tf.reduce_mean(entro)
            output =  entro + Loss.softmax_ce(label, pred, name='sm_ce')
            return output

    @staticmethod
    def js_div(label, pred, name='kl'):
        with tf.variable_scope(name) as scope:
            return 0.5 * Loss.kl_div(label, (label + pred) / 2, name='js1') \
                + 0.5 * Loss.kl_div(pred, (label + pred) / 2, name='js2')

    def __init__(self, name):
        if 'ce' in name:
            self.loss_func = tf.losses.softmax_cross_entropy
        elif 'mse' in name:
            self.loss_func = Loss.squared_difference
        elif 'mae' in name:
            self.loss_func = Loss.absolute_difference
        elif 'huber' in name:
            self.loss_func = Loss.huber_loss
        elif 'hinge' in name:
            self.loss_func = Loss.hinge_loss
        elif 'kl' in name:
            self.loss_func = Loss.kl_div
        elif 'js' in name:
            self.loss_func = Loss.js_div

    def get_loss(self, labels, logits):
        return tf.reduce_mean(self.loss_func(labels, logits))
