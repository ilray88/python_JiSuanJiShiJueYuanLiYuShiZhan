import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np

data = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
label = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

def perceptron(inp, name):
    with tf.variable_scope(name) as scope:
        n = inp.get_shape().as_list()[-1]
        # 定义参数W形状为[n, 1]，以正态分布随机数进行初始化
        w = tf.Variable(
                tf.random_normal(
                    [n, 1], 
                    mean=0.0, 
                    stddev=0.02
                ), dtype=tf.float32, name='w')

        # 定义标量参数b，以正态分布随机数进行初始化
        b = tf.Variable(
                tf.random_normal(
                    [], 
                    mean=0.0, 
                    stddev=0.02
                ), dtype=tf.float32, name='b')
        
        # 以矩阵运算xW+b的方式计算感知机的输出结果
        output = tf.add(tf.matmul(inp, w), b)

        # 返回计算结果以及w参数值，便于作图
        return output, w, b

output, w, b = perceptron(data, name='perceptron')
output = tf.nn.sigmoid(output)

loss = tf.reduce_mean(
            -(label * tf.log(output) + (1 - label) * tf.log(1 - output))
       )
op = tf.train.MomentumOptimizer(1e-1, 0.9, use_nesterov=True).minimize(loss)

# 打开交互作图模式，方便查看分类边界变化情况
plt.ion()

# 生成x数据
x = np.linspace(-0.1, 1.5, 100)
plt.grid()

with tf.Session() as sess:
    # 初始化感知机中的随机变量
    tf.global_variables_initializer().run()

    # 训练100个周期
    for i in range(500):
        # 拿出每个周期内的W和b的参数值以及loss，方便观察其变化情况
        wi, bi, loss_i, _ = sess.run([w, b, loss, op])

        # 绘图部分
        plt.cla()
        
        # 绘制异或运算的结果
        plt.scatter(0, 0, s=150, c='red')
        plt.scatter(0, 1, s=200, c='blue', marker='*')
        plt.scatter(1, 0, s=200, c='blue', marker='*')
        plt.scatter(1, 1, s=150, c='red')

        # 计算当前得到的直线斜率与截距
        K = -wi[0] / wi[1]
        B = -bi / wi[1]

        # 绘制分类边界
        plt.plot(x,  K * x + B, label='$x_{2}$=' + '{}'.format(K) + '$x_{1}$+' + '{}'.format(B))
        plt.legend()
        plt.pause(0.001)

        # 打印每一个周期的参数与loss信息
        print('Epoch {}: K -> {}, B -> {}, loss -> {}'.format(i, K, B, loss_i))

# 关闭交互模式
plt.ioff()
# 显示最终结果
plt.show()

