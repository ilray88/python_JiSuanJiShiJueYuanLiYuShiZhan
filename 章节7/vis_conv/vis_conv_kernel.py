import os
from tensorflow.python import pywrap_tensorflow

# 权值文件所在的文件夹
model_dir = "../conv_nets/checkpoint/resnet18_mnist"
# 权值文件名称
checkpoint_path = os.path.join(model_dir, "checkpoint-288")

# 创建权值文件的Reader
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# 取出权值文件中的所有变量名称
var_to_shape_map = reader.get_variable_to_shape_map()

kernel_names = list()

for key in var_to_shape_map:
    # 查询名称中含有conv/w的变量
    if key.endswith('conv/w'):
        kernel_names.append(key)

print('\n'.join(kernel_names))

import matplotlib.pyplot as plt

H = 4
W = 4
block_prefix_plh = 'block_{}_1/sub_block2'

for i in range(4):
    fig, axes = plt.subplots(H, W)
    block_prefix = block_prefix_plh.format(i)

    for name in kernel_names:
        if block_prefix in name:
            mat = reader.get_tensor(name)
            # 取出每一个卷积核 
            chosen_mat = mat[..., 0, :H*W]
            
            for i in range(H):
                for j in range(W):
                    # 以索引的形式取出每一个axes
                    axes[i][j].imshow(chosen_mat[..., i * W + j], cmap='gray')
                    axes[i][j].set_title('[{}]'.format(i * W + j))
            # 设置总图标题
            plt.suptitle('Conv Kernels of {}'.format(block_prefix))
            plt.show()
