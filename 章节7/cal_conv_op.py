import numpy as np

def conv_by_define(x, kernel, stride):
    # 获取输入的各维度大小
    b, h, w, c = x.shape

    # 获取卷积核的各维度大小
    k1, k2, cin, cout = kernel.shape
    # 获取各维度的步长
    sb, sh, sw, sc = stride
    
    # 计算输出的尺寸大小
    out_h = int((h - k1) / sh + 1)
    out_w = int((w - k2) / sw + 1)

    # 定义输出的占位符
    output = np.empty(shape=[b, out_h, out_w, cout])

    for _cout in range(cout):
        # 对每一个输出通道取出一个卷积核
        _kernel = kernel[...,_cout]

        # 计算每一个滑窗的位置
        for _out_h in range(out_h):
            for _out_w in range(out_w):
                h_start = sh * _out_h
                w_start = sw * _out_w
                # 使用点乘+求和的方式计算
                output[:, _out_h, _out_w, _cout] = \
                        np.sum(
                            _kernel * x[:, 
                                        h_start: h_start + k1,
                                         w_start: w_start + k2, 
                                        :]
                        )
    
    return output

def conv_by_img2col(x, kernel, stride):
    # 获取输入各维度大小
    b, h, w, c = x.shape

    # 获取卷积核各维度大小
    k1, k2, cin, cout = kernel.shape

    # 获取各维度上的步长
    sb, sh, sw, sc = stride

    # 计算输出的空间维度大小
    out_h = int((h - k1) / sh + 1)
    out_w = int((w - k2) / sw + 1)

    # 定义矩阵A的占位符
    col = np.empty((b * out_h * out_w, k1 * k2 * c))

    outsize = out_w * out_h

    for _out_h in range(out_h):
        # 原输入的卷积对应部分
        h_min = _out_h * sh
        h_max = h_min + k1

        h_start = _out_h * out_w

        for _out_w in range(out_w):
            w_min = _out_w * sw
            w_max = w_min + k2
            # 将原输入卷积操作对应部分重整放入矩阵A对应位置
            col[h_start + _out_w:: outsize, :] = \
                 x[:, h_min: h_max, w_min: w_max, :].reshape(b, -1)

    # 重整卷积核的形状
    kernel = np.reshape(kernel, [-1, cout])

    # 使用矩阵的乘法计算卷积
    z = np.dot(col, kernel)
    # 将乘法后的结果重整为输出的形状
    z = z.reshape(b, z.shape[0] // b, -1)
    
    return z.reshape(b, out_h, -1 , cout)


if __name__ == "__main__":
    import time
    # 运行100000次
    times = 100000

    # input的形状为[1,6,6,1]
    # 即batch中只有一个样本，高和宽都为6，通道数为1
    input = np.array(
        [[
            [[1],[1],[0],[1],[0],[1]],
            [[0],[1],[0],[0],[0],[1]],
            [[0],[0],[1],[1],[0],[0]],
            [[0],[1],[0],[1],[0],[1]],
            [[1],[0],[0],[0],[1],[1]],
            [[0],[0],[1],[1],[0],[0]]
        ]])

    # 卷积核的形状为[3,3,1,1]
    # 即卷积核尺寸为3*3，输入和输出通道数都为1
    kernel = np.array(
        [
            [[[1]],[[0]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[1]],[[0]],[[0]]]
        ])

    # 按定义计算的时间
    start = time.time()
    for _ in range(times):
        o1 = conv_by_define(input, kernel, stride=[1, 1, 1, 1])
    print(time.time() - start)

    # img2col的计算时间
    start = (time.time())
    for _ in range(times):
        o2 = conv_by_img2col(input, kernel, stride=[1, 1, 1, 1])
    print(time.time() - start)
