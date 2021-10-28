import pandas as pd
import numpy as np
import random
from copy import deepcopy

try:
    from base_class import Dataset
except ImportError:
    from .base_class import Dataset

class Iris(Dataset):
    num_classes = 3

    # 类别名称与类标之间的映射关系
    flower_name_id_dic = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    # 每个特征上的最大值与最小值
    max_val = np.array([7.9, 4.4, 6.9, 2.5])
    min_val = np.array([4.3, 2.0, 1.0, 0.1])

    # 每个特征上的均值与标准差
    mean = np.array([0.4287037037037038, 0.4391666666666666, 0.4675706214689266, 0.4577777777777779])
    std = np.array([0.22925036, 0.18006108, 0.29805579, 0.31692192])

    def __init__(self, data_path, batch_size, shuffle=True, normalize=True, split_train_and_test=0.2, augmentation=None):
        if type(data_path) == str:
            data_path = [data_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize

        self.__train_pointer = [batch_size]

        self.__test_pointer = [batch_size]

        __data = list()
        __labels = list()

        for dp in data_path:
            data = self.__read(dp)
            __data.extend([_d[: 4] for _d in data])
            __labels.extend([_d[-1] for _d in data])

        # 计算出测试集应包含多少个样本
        num_test = int(split_train_and_test * len(__data))
        # 使用random随机选取测试集样本的下标
        test_ids = random.sample(list(range(len(__data))), k=num_test)

        # 将全部样本通过下标分割为训练集与测试集
        self.__train_data = [__data[idx] 
                                for idx in range(len(__data)) if idx not in test_ids]
        self.__train_labels = [__labels[idx] 
                                for idx in range(len(__data)) if idx not in test_ids]

        self.__test_data = [__data[idx] 
                                for idx in range(len(__data)) if idx in test_ids]
        self.__test_labels = [__labels[idx] 
                                for idx in range(len(__data)) if idx in test_ids]

        # 将训练和测试的标签由字符串转为标量值
        self.__train_labels = [self.flower_name_id_dic[n] 
                                    for n in self.__train_labels]
        self.__test_labels = [self.flower_name_id_dic[n] 
                                    for n in self.__test_labels]

        # 将数据和标签转为numpy.array类型
        self.__train_data = np.stack(self.__train_data, axis=0)
        self.__test_data = np.stack(self.__test_data, axis=0)

        # 将标签转换为one-hot向量
        self.__train_labels = np.eye(self.num_classes)[self.__train_labels]
        self.__test_labels = np.eye(self.num_classes)[self.__test_labels]

    
    @staticmethod
    def __read(file):
        return pd.read_csv(file, header=None, low_memory=False).values

    def __normalization(self, data):
        data = (data - self.min_val) / (self.max_val - self.min_val)
        data = (data - self.mean) / self.std

        return data

    def next_batch(self, which_set):
        # 判断是对哪一个集合做操作
        # 分别取出对应的数据、标签以及当前数据位置指针
        if 'train' in which_set:
            target_data = self.__train_data
            target_label = self.__train_labels
            target_pointer = self.__train_pointer
        elif 'test' in which_set:
            target_data = self.__test_data
            target_label = self.__test_labels
            target_pointer = self.__test_pointer

        # 如果需要batch内数据乱序（shuffle=True），直接随机选取样本即可
        if self.shuffle:
            indices = np.random.choice(
                self.num_examples(which_set), self.batch_size)
        else:
            # 否则使用指针顺序取出数据与标签，注意指针指到最后时需要将其重新打到数据开头
            indices = list(
                range(target_pointer[0] - self.batch_size, target_pointer[0]))

            target_pointer[0] = (target_pointer[0] + self.batch_size) % self.num_examples(which_set)
        
        # 取出batch数据后，使用深拷贝得到一个副本方便操作，防止篡改原始数据
        batch_data = deepcopy(target_data[indices])

        # 对batch里的数据做标准化
        if self.normalize:
            batch_data = self.__normalization(batch_data)

        return batch_data, target_label[indices]

    def num_examples(self, which_set):
        if 'train' in which_set:
            return len(self.__train_data)
        elif 'test' in which_set:
            return len(self.__test_data)


if __name__ == "__main__":
    import os

    files = ['iris.data']
    root = r'data path'
    iris = Iris(batch_size=16, 
                data_path=[os.path.join(root, x) for x in files], 
                shuffle=False)
    
    # while True:
    #     iris.next_batch('train')

    ims, labs = iris.next_batch('train')
    print(ims, labs.shape)
