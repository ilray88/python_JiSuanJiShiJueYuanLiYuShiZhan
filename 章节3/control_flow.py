import tensorflow.compat.v1 as tf

# 定义常量a与b，其值分别为1.0与2.0
a = tf.constant(1.0, name='a')
b = tf.constant(2.0, name='b')

# 定义一个判断条件的占位符，其类型为tf.bool
condition = tf.placeholder(dtype=tf.bool, name='condition')

# 当condition为True时，返回c=a+b，否则为c=a-b，此处使用匿名函数实现
c = tf.cond(condition, lambda: a + b, lambda: a - b)

# 当a<b时，d=a*b，否则d=a/b
d = tf.cond(a < b, lambda: a * b, lambda: a / b)

# 由于计算图中不存在变量，因此不需要使用variable_initializer
with tf.Session() as sess:
    # 根据传入的不同bool值得到不同的结果
    print(sess.run(c, feed_dict={condition: True}))
    print(sess.run(c, feed_dict={condition: False}))
    print(sess.run(d))
# ==================================================================
# 定义常量a与b，其值分别为1.0与2.0
a = tf.constant(1.0, name='a')
b = tf.constant(2.0, name='b')

# 定义一个判断条件的占位符，其类型为tf.int32
condition = tf.placeholder(dtype=tf.int32, name='condition')

# 使用键值对定义case，并指定exclusive为False
c = tf.case(
    {condition > 1: lambda: a + b, condition > 2: lambda: a + 2 * b}, default=lambda: a - b, exclusive=False
)

# c = tf.case(
#     [(condition > 1, lambda: a + b), (condition > 2, lambda: a + 2 * b)], default=lambda: a - b, exclusive=False
# )

# 使用键值对定义case，并指定exclusive为True，此时会报错，因为两个条件再condition>2时都为True
d = tf.case(
    {condition > 1: lambda: a + b, condition > 2: lambda: a + 2 * b}, default=lambda: a - b, exclusive=True
)

# 由于计算图中不存在变量，因此不需要使用variable_initializer
with tf.Session() as sess:
    # 根据传入的不同的condition值得到不同的结果
    print(sess.run(c, feed_dict={condition: 1}))
    print(sess.run(c, feed_dict={condition: 2}))
    print(sess.run(c, feed_dict={condition: 3}))

    print(sess.run(d, feed_dict={condition: 1}))
    print(sess.run(d, feed_dict={condition: 2}))
    # 报错
    # print(sess.run(d, feed_dict={condition: 3}))
# ==================================================================
# 定义循环中需要使用的变量i和n
i = 0
n = 10

# 循环条件函数
def judge(i, n):
    # 当i < sqrt(n)时才执行循环
    return i * i < n

# 循环体函数
def body(i, n):
    # 循环中使i增1
    i = i + 1

    # 返回的参数与输入的参数保持一致
    return i, n

# 为tf.while_loop传入条件函数、循环体函数以及参数
new_i, new_n = tf.while_loop(judge, body, [i, n])

with tf.Session() as sess:
    print(sess.run([new_i, new_n]))
# ==================================================================
# 定义一个变量x，其初始值为2
x = tf.Variable(2)
# 为x定义一个加一操作
x_assign = tf.assign(x, x + 1)
# y1值为x^2
y1 = x ** 2

with tf.control_dependencies([x_assign]):
    # y2值为x^2，其需要在执行x_assign之后执行
    y2 = x ** 2

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run([y1, y2]))
    