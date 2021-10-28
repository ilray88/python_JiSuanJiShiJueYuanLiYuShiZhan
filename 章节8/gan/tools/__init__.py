from .timer import *
from .counter import *
from .printer import *

counter = Counter()
vars_printer = VarsPrinter()
args_printer = ArgsPrinter()
acc_printer = AccPrinter()


def print_net_info(vars=None):
    counter(vars)
    vars_printer(vars)


def print_args(args):
    args_printer(args)


def print_training_info(epoch, loss, acc):
    acc_printer(epoch, loss, acc)
