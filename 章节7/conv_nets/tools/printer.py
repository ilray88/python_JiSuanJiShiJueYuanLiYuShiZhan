import tensorflow.compat.v1 as tf


class Printer:
    spliter1 = '=' * 100
    spliter2 = '-' * 100
    placeholder = '\t {:50} \t {}'
    
    def __init__(self, msg_or_placeholder=None):
        if msg_or_placeholder:
            self.placeholder = msg_or_placeholder

    def __call__(self, *msg):
        print(self.spliter1)
        print(self.placeholder.format(*msg))
        print(self.spliter1)


class VarsPrinter(Printer):
    def __call__(self, vars=None):
        if not vars:
            vars = tf.trainable_variables()

        print(self.spliter1)
        print(self.placeholder.format('Name', 'Shape'))
        print(self.spliter2)
        for v in vars:
            print(self.placeholder.format(v.name, v.shape))
        print(self.spliter1)


class ArgsPrinter(Printer):
    def __init__(self):
        self.placeholder = '\t {:20} \t {}'

    def __call__(self, args):
        print(self.spliter1)
        print(self.placeholder.format('Args', 'Values'))
        print(self.spliter2)

        for k, v in vars(args).items():
            if type(v) == list:
                for idx, _v in enumerate(v):
                    if idx == 0:
                        print(self.placeholder.format(k, _v))
                    else:
                        print(self.placeholder.format('', _v))
            else:
                print(self.placeholder.format(k, v))
        
        print(self.spliter1)

class AccPrinter(Printer):
    def __init__(self):
        self.placeholder = 'Epoch {:5}: \t{:.3}\t acc: {:.3}'
    
    def __call__(self, epoch, loss, acc):
        print(self.placeholder.format(epoch, loss, acc))