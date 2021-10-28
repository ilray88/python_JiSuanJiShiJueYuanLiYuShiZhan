import tensorflow.compat.v1 as tf

# input的形状为[1,6,6,1]
# 即batch中只有一个样本，高和宽都为6，通道数为1
input = tf.constant(
    [[
        [[1],[1],[0],[1],[0],[1]],
        [[0],[1],[0],[0],[0],[1]],
        [[0],[0],[1],[1],[0],[0]],
        [[0],[1],[0],[1],[0],[1]],
        [[1],[0],[0],[0],[1],[1]],
        [[0],[0],[1],[1],[0],[0]]
    ]], dtype=tf.float32)

# 卷积核的形状为[3,3,1,1]
# 即卷积核尺寸为3*3，输入和输出通道数都为1
kernel = tf.constant(
    [
        [[[1]],[[0]],[[0]]],
        [[[0]],[[1]],[[0]]],
        [[[1]],[[0]],[[0]]]
    ], dtype=tf.float32)

# 步长为1
# 使用SAME填充方式
output11 = tf.nn.conv2d(input, kernel, [1,1,1,1], 'SAME')
# 使用VALID填充方式
output12 = tf.nn.conv2d(input, kernel, [1,1,1,1], 'VALID')
# 使用EXPLICIT填充方式，在高和宽的维度上每侧都填充3个单位
output13 = tf.nn.conv2d(input, kernel, [1,1,1,1], [[0,0], [3,3], [3,3], [0,0]])

# 步长为2
# 使用SAME填充方式
output21 = tf.nn.conv2d(input, kernel, [1,2,2,1], 'SAME')
# 使用VALID填充方式
output22 = tf.nn.conv2d(input, kernel, [1,2,2,1], 'VALID')
# 使用EXPLICIT填充方式，在高和宽的维度上每侧都填充3个单位
output23 = tf.nn.conv2d(input, kernel, [1,2,2,1], [[0,0], [3,3], [3,3], [0,0]])

with tf.Session() as sess:
    o11 = sess.run(output11)
    o12 = sess.run(output12)
    o13 = sess.run(output13)

    o21 = sess.run(output21)
    o22 = sess.run(output22)
    o23 = sess.run(output23)

    print(o11, o11.shape)
    print(o12, o12.shape)
    print(o13, o13.shape)

    print(o21, o21.shape)
    print(o22, o22.shape)
    print(o23, o23.shape)
