import numpy as np
import cv2

def flip_one(image, axis):
    image = cv2.flip(image, axis)
    return image

def resize_one(image, size):
    return cv2.resize(image, size)


def horizontal_flip(batch_data, prob=0.5):
    # 获取batch的形状
    N, H, W, C = batch_data.shape

    # 为翻转后的数据创建一个形状与输入相同的占位符
    flipped_batch = np.zeros_like(batch_data)

    # 对batch中的每一张图像分别进行操作
    for i in range(N):
        # 随机生成执行概率
        flip_prob = np.random.rand()

        if flip_prob < prob:
            # 对batch中的第i张图像进行水平翻转操作
            flipped_batch[i] = flip_one(batch_data[i], axis=1)

    # 返回水平翻转后的batch
    return flipped_batch


def vertical_flip(batch_data, prob=0.5):
    # 获取batch的形状
    N, H, W, C = batch_data.shape

    # 为翻转后的数据创建一个形状与输入相同的占位符
    flipped_batch = np.zeros_like(batch_data)

    # 对batch中的每一张图像分别进行操作
    for i in range(N):
        # 随机生成执行概率
        flip_prob = np.random.rand()

        if flip_prob < prob:
            # 对batch中的第i张图像进行水平翻转操作
            flipped_batch[i] = flip_one(batch_data[i], axis=0)

    # 返回水平翻转后的batch
    return flipped_batch


def whitening_image(image_np, prob=0.5):
    N, H, W, C = image_np.shape

    for i in range(N):
        whiten_prob = np.random.rand()

        if whiten_prob <  prob:
            mean = np.mean(image_np[i])
            # Use adjusted standard deviation here, in case the std == 0.
            std = np.max([np.std(image_np[i]), 1.0 / np.sqrt(H * W * C)])
            image_np[i] = (image_np[i] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size, resize=False):
    # 获取batch的形状
    N, H, W, C = batch_data.shape

    # 为翻转后的数据创建一个形状与输入相同的占位符
    new_batch = np.zeros_like(batch_data)

    # 根据每个方向上填充的宽度计算新图像的大小
    new_H = H + 2 * padding_size
    new_W = W + 2 * padding_size

    for i in range(N):
        # 生成随机裁剪的左上角坐标
        y_offset = np.random.randint(low=0, high=2 * padding_size)
        x_offset = np.random.randint(low=0, high=2 * padding_size)
        
        # 使用缩放方式进行裁剪或填充方式进行裁剪
        if resize:
            image = resize_one(batch_data[i], (new_H, new_W))
        else:
            image = np.pad(batch_data[i], 
                           (
                               (padding_size, padding_size), 
                               (padding_size, padding_size), 
                               (0, 0)
                            ), 
                           mode='constant')
        
        # 完成图像的裁剪
        new_batch[i] = image[y_offset: y_offset + H, x_offset: x_offset + W, :]

    # 完成对batch的随机水平翻转
    new_batch = horizontal_flip(new_batch, prob=0.5)

    return new_batch