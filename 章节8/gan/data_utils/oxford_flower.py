import cv2
import numpy as np
import os
import scipy.io as scio

try:
    from base_class import Dataset
    from augmentation import random_crop_and_flip
except ImportError:
    from .base_class import Dataset
    from .augmentation import random_crop_and_flip


class OxfordFlower102(Dataset):
    num_classes = 102
    mean = [0.28749102, 0.37729599, 0.43510646]
    std = [0.26957776, 0.24504408, 0.29615187]

    def __init__(self, data_path, batch_size, shuffle=True, normalize=True, resize=None, augmentation=True):
        if type(data_path) == str:
            data_path = [data_path]

        self.image_root = data_path[0]
        self.resize = resize

        image_split = [x for x in data_path[1:] if 'setid' in x][0]
        label = [x for x in data_path[1:] if 'imagelabels' in x][0]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation

        self.__train_pointer = [batch_size]
        self.__val_pointer = [batch_size]
        self.__test_pointer = [batch_size]

        readObjs = self.__read(image_split)
        self.__train_images = np.squeeze(readObjs['trnid'])
        self.__val_images = np.squeeze(readObjs['valid'])
        self.__test_images = np.squeeze(readObjs['tstid'])

        self.__all_labels = np.squeeze(self.__read(label)['labels']) - 1
        self.__all_labels = np.eye(self.num_classes)[self.__all_labels]

        self.__normalize = normalize

    def __read(self, file_path):
        return scio.loadmat(file_path)

    def next_batch(self, which_set):
        # 确定对某一个集合进行操作
        if 'train' in which_set:
            target_image = self.__train_images
            target_pointer = self.__train_pointer
            do_augment = self.augmentation
        elif 'val' in which_set:
            target_image = self.__val_images
            target_pointer = self.__val_pointer
            do_augment = False
        elif 'test' in which_set:
            target_image = self.__test_images
            target_pointer = self.__test_pointer
            do_augment = False

        # 以乱序或顺序的方式取一个batch的图像编号
        if self.shuffle:
            indexes = np.random.choice(
                self.num_examples(which_set), self.batch_size)
        else:
            indexes = list(
                range(target_pointer[0] - self.batch_size, target_pointer[0]))

            target_pointer[0] = (target_pointer[0] + self.batch_size) % self.num_examples(which_set)

        # 根据选出的batch图像编号读取batch图像
        imgs = np.stack([self.__read_images(target_image[idx])
                         for idx in indexes], axis=0)

        # 由于编号从1~8189，需要将其减1得到0~8188作为索引读取标签
        labels = np.stack([self.__all_labels[target_image[idx] - 1]
                           for idx in indexes], axis=0)
        
        # 是否进行数据增强
        if do_augment:
            imgs = random_crop_and_flip(imgs, padding_size=16)

        if self.__normalize:
            imgs = self.__normalization(imgs)

        return imgs, labels

    def __normalization(self, imgs, epslion=1e-5):
        imgs = imgs / 255.0
        imgs = (imgs - self.mean) / self.std

        return imgs

    def __read_images(self, img_id):
        # 将传入的image_id转为字符串方便进行拼接
        img_id = str(img_id)

        # 拼接出文件的完整路径
        img_path = os.path.join(self.image_root, 
                        'image_{}.jpg'.format('0' * (5 - len(img_id)) + img_id))

        # 使用OpenCV读取图像（numpy.array类型）
        img = cv2.imread(img_path)

        # 是否对图像进行缩放
        if self.resize:
            img = cv2.resize(img, self.resize)

        # 返回读取的图像
        return img

    def num_examples(self, which_set):
        if 'train' in which_set:
            return len(self.__train_images)
        elif 'val' in which_set:
            return len(self.__val_images)
        elif 'test' in which_set:
            return len(self.__test_images)


if __name__ == "__main__":
    import os
    from tqdm import tqdm

    files = ['jpg', 'setid.mat', 'imagelabels.mat']
    root = r'data path'
    of102 = OxfordFlower102(batch_size=512, data_path=[os.path.join(
        root, x) for x in files], shuffle=False, resize=(256, 256))

    # while True:
    #     of102.next_batch('train')
    im, label = of102.next_batch('train')
    print(im.shape, label.shape)
    print(of102.num_examples('train'), of102.num_examples('val'), of102.num_examples('test'))