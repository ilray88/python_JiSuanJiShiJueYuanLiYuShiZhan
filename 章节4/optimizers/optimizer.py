import tensorflow.compat.v1 as tf


class Optimizer:
    def __init__(self, initial_lr, boundary, decay, warmup, warmup_iter, name):
        self.global_step = tf.Variable(dtype=tf.int32, initial_value=0, trainable=False, name='global_step')
        self.initial_lr = initial_lr
        self.boundary = boundary
        self.decay =  decay
        self.warmup = warmup
        self.warmup_iter = warmup_iter

        self.learning_rate = self.get_lr()

        tf.summary.scalar('learning rate', self.learning_rate)

        if 'adam' in name:
            self.optim = tf.train.AdamOptimizer(self.learning_rate)
        elif 'sgd' in name:
            self.optim = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif 'adadelta' in name:
            self.optim = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif 'adagrad' in name:
            self.optim = tf.train.AdagradOptimizer(self.learning_rate)
        elif 'momentum' in name:
            self.optim = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        elif 'nestrov' in name:
            self.optim = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True)
        elif 'ftrl' in name:
            self.optim = tf.train.FtrlOptimizer(self.learning_rate)
        elif 'rmsprop' in name:
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate)

    def get_lr(self):
        vals = [self.initial_lr, *[self.initial_lr * self.decay ** exp for exp in range(1, len(self.boundary) + 1)]]

        lr = tf.train.piecewise_constant(self.global_step, self.boundary, vals)
        
        
        if self.warmup:
            warmup_lr = self.initial_lr * tf.cast(self.global_step, tf.float32) / self.warmup_iter
            return tf.cond(
                        tf.math.less_equal(self.global_step, self.warmup_iter),
                        lambda: warmup_lr, 
                        lambda: lr
                    )
        return lr

    def minimize(self, loss):
        return self.optim.minimize(loss, global_step=self.global_step), self.learning_rate



if __name__ == '__main__':
    # ???????????????????????????
    lr = tf.placeholder(dtype=tf.float32, name='lr')
    
    # ????????????????????????
    op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    
    # ??????????????????100???epoch?????????????????????0.1
    epoch = 100
    base_lr = 0.1

    with tf.Session() as sess:
        for e in range(epoch):
            # ????????????????????????????????????????????????????????????????????????
            e_lr = base_lr * (1 - e / epoch)
            
            # ??????????????????????????????????????????????????????????????????
            sess.run(op, feed_dict={lr: e_lr})

            