import struct
import numpy as np
import random
from copy import deepcopy

try:
    from base_class import Dataset
    from augmentation import random_crop_and_flip
except ImportError:
    from .base_class import Dataset
    from .augmentation import random_crop_and_flip


class Cifar(Dataset):
    def __init__(self, 
                 data_path, 
                 batch_size, 
                 shuffle=True, 
                 normalize=True, 
                 c10=True, 
                 coarse_label=False, 
                 augmentation=True):
        if type(data_path) == str:
            data_path = [data_path]

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

        if c10:
            self.train_identifier = 'data_batch'
            self.test_identifier = 'test_batch'
            label_name = b'labels'
            self.num_classes = 10
            self.mean = [0.49186878, 0.48265391, 0.44717728] 
            self.std = [0.24697121, 0.24338894, 0.26159259]
        else:
            self.train_identifier = 'train'
            self.test_identifier = 'test'
            
            self.mean = [0.50736203, 0.48668956, 0.44108857] 
            self.std = [0.26748815, 0.2565931, 0.27630851]

            if coarse_label:
                label_name = b'coarse_labels'
                self.num_classes = 20
            else:
                label_name = b'fine_labels'
                self.num_classes = 100

        for dp in data_path:
            data = self.unpickle(dp)
            if self.train_identifier in dp:
                self.__train_images.append(data[b'data'])
                self.__train_labels.append(data[label_name])
            if self.test_identifier in dp:
                self.__test_images.append(data[b'data'])
                self.__test_labels.append(data[label_name])

        self.__train_images = np.concatenate(self.__train_images, axis=0)
        self.__train_images = np.transpose(np.reshape(
            self.__train_images, [-1, 3, 32, 32]), [0, 2, 3, 1])
        self.__train_labels = np.concatenate(self.__train_labels, axis=0)

        self.__test_images = np.concatenate(self.__test_images, axis=0)
        self.__test_images = np.transpose(np.reshape(
            self.__test_images, [-1, 3, 32, 32]), [0, 2, 3, 1])
        self.__test_labels = np.concatenate(self.__test_labels, axis=0)

        self.__train_labels = np.eye(self.num_classes)[self.__train_labels]
        self.__test_labels = np.eye(self.num_classes)[self.__test_labels]


    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __normalization(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - self.mean) / self.std

        return imgs

    def next_batch(self, which_set):
        if 'train' in which_set:
            target_image = self.__train_images
            target_label = self.__train_labels
            target_pointer = self.__train_pointer
            do_augment = self.augmentation
        elif 'test' in which_set:
            target_image = self.__test_images
            target_label = self.__test_labels
            target_pointer = self.__test_pointer
            do_augment = False

        if self.shuffle:
            indices = random.sample(
                range(self.num_examples(which_set)), k=self.batch_size)
        else:
            indices = list(
                range(target_pointer[0] - self.batch_size, target_pointer[0]))

            target_pointer[0] = (target_pointer[0] + self.batch_size) % self.num_examples(which_set)

        batch_data = deepcopy(target_image[indices])

        if do_augment:
            batch_data = random_crop_and_flip(batch_data, padding_size=4)

        if self.normalize:
            batch_data = self.__normalization(batch_data)

        return batch_data, target_label[indices]

    def num_examples(self, which_set):
        if 'train' in which_set:
            return len(self.__train_images)
        elif 'test' in which_set:
            return len(self.__test_images)


if __name__ == "__main__":
    import os
    # c10
    files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
    root = r'data path'

    # c100
    # files = ['train', 'test']
    # root = r'D:\cifar-100-python'
    cifar = Cifar(batch_size=1024, 
                  data_path=[os.path.join(root, x) for x in files], 
                  shuffle=False, 
                  c10=True, 
                  coarse_label=False,
                  normalize=True,
                  augmentation=True)
    # while True:
    #     cifar.next_batch('train')
    ims, labs = cifar.next_batch('test')
    print(ims[0], labs[0])
