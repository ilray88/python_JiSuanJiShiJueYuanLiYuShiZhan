import abc

class Model(metaclass=abc.ABCMeta):
    # 构造函数传入必要的结构定义参数
    @abc.abstractmethod
    def __init__(self, structure_param):
        pass
    
    # 模型搭建函数
    # 传入输入值x与布尔值is_training
    # （因为有一些操作在训练与测试阶段计算方式不同，如BN）
    @abc.abstractmethod
    def build(self, x, is_training):
        pass