import struct
import numpy as np
from copy import deepcopy

try:
    from base_class import Dataset
    from augmentation import random_crop_and_flip
except ImportError:
    from .base_class import Dataset
    from .augmentation import random_crop_and_flip


class Mnist(Dataset):
    mean = 0.13092535192648502
    std = 0.3084485240270358

    train_identifier = 'train'
    test_identifier = 't10k'
    num_classes = 10

    def __init__(self, data_path, batch_size, shuffle=True, normalize=True, augmentation=True):
        if type(data_path) == str:
            data_path = [data_path]

        image_path = [x for x in data_path if 'images' in x]
        label_path = [x for x in data_path if 'labels' in x]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.augmentation = augmentation

        self.__train_images = list()
        self.__train_labels = list()
        self.__train_pointer = [batch_size]

        self.__test_images = list()
        self.__test_labels = list()
        self.__test_pointer = [batch_size]

        for imp in image_path:
            if self.train_identifier in imp:
                self.__train_images.extend(self.__read(
                    self.__read_zip_file(imp), '>IIII', '>784B'))
            if self.test_identifier in imp:
                self.__test_images.extend(self.__read(
                    self.__read_zip_file(imp), '>IIII', '>784B'))

        for lp in label_path:
            if self.train_identifier in lp:
                self.__train_labels.extend(self.__read(
                    self.__read_zip_file(lp), '>II', '>1B'))
            if self.test_identifier in lp:
                self.__test_labels.extend(self.__read(
                    self.__read_zip_file(lp), '>II', '>1B'))

        self.__train_images = np.array(self.__train_images)
        self.__train_labels = np.array(self.__train_labels).squeeze()

        self.__test_images = np.array(self.__test_images)
        self.__test_labels = np.array(self.__test_labels).squeeze()

        self.__train_labels = np.eye(self.num_classes)[self.__train_labels]
        self.__test_labels = np.eye(self.num_classes)[self.__test_labels]


    def __normalization(self, imgs, epslion=1e-5):
        imgs = imgs / 255.0
        imgs = (imgs - self.mean) / self.std

        return imgs

    def __read(self, buffer, to_skip, each_size):
        objs = list()
        idx = struct.calcsize(to_skip)

        try:
            while True:
                o = struct.unpack_from(each_size, buffer, idx)
                objs.append(o)
                idx += struct.calcsize(each_size)
        except struct.error:
            return objs

    def __read_zip_file(self, file_path):
        with open(file_path, 'rb') as f:
            buffer = f.read()
        return buffer

    def next_batch(self, which_set, reshape=True):
        # 读取训练或测试数据
        if 'train' in which_set:
            target_image = self.__train_images
            target_label = self.__train_labels
            target_pointer = self.__train_pointer
        elif 'test' in which_set:
            target_image = self.__test_images
            target_label = self.__test_labels
            target_pointer = self.__test_pointer

        # 以随机或顺序的方式读取数据
        if self.shuffle:
            indices = np.random.choice(
                self.num_examples(which_set), self.batch_size)
        else:
            indices = list(
                range(target_pointer[0] - self.batch_size, target_pointer[0]))
            
            target_pointer[0] = (target_pointer[0] + self.batch_size) % self.num_examples(which_set)

        batch_data = deepcopy(target_image[indices])

        # 对输入数据进行标准化
        if self.normalize:
            batch_data = self.__normalization(batch_data)
        
        # 将数据重整为二维图像
        if reshape:
            batch_data = np.reshape(batch_data, [-1, 28, 28, 1])

        return batch_data, target_label[indices]

    def num_examples(self, which_set):
        if 'train' in which_set:
            return len(self.__train_images)
        elif 'test' in which_set:
            return len(self.__test_images)


if __name__ == "__main__":
    import os
    import random
    import matplotlib.pyplot as plt

    files = ['train-images.idx3-ubyte', 't10k-images.idx3-ubyte',
             'train-labels.idx1-ubyte', 't10k-labels.idx1-ubyte']
    root = r'data path'
    
    batch_size = 128
    mnist = Mnist(batch_size=batch_size, 
                  data_path=[os.path.join(root, x) for x in files], 
                  shuffle=False,
                  normalize=False)

    # 取出一个batch的数据
    ims, labs = mnist.next_batch('train')

    # 将数据重整成Matplotlib可以显示的形状
    ims = np.squeeze(ims)

    # 一共随机取出8*8张图像
    row = col = 8

    random_ids = random.sample(list(range(batch_size)), k=row * col)
    selected_ims = ims[random_ids]

    fig, axes = plt.subplots(row, col)
    for i in range(row):
        for j in range(col):
            # 取出每一个axes对图像进行显示
            axes[i][j].imshow(selected_ims[i * row + j], cmap='gray')
    # 总图标题
    plt.suptitle('MNIST samples')
    plt.show()
    
    print(ims[0], labs[0])
