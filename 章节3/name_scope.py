import tensorflow.compat.v1 as tf

# 使用tf.Variable创建一个浮点型变量
var1 = tf.Variable(1.2, name='var1')

# 尝试使用tf.get_variable方法获取定义过的变量var1
var2 = tf.get_variable(name='var1', shape=[])

# 查看var2与var1是否指向同一变量
print(var1 == var2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1, sess.run(var1))
    print(var2, sess.run(var2))

# ===============================================
# 定义一个名为scope1的命名空间
with tf.name_scope('scope1'):
    # 使用tf.Variable定义变量var3
    var3_1 = tf.Variable(2.5, name='var3')
    
    # 使用tf.get_variable得到变量var4
    var4_1 = tf.get_variable(name='var4', initializer=0.0)
    
    # 定义加操作
    var5_1 = var3_1 + var4_1

# 打印空间内节点信息以查看其名称
print(var3_1)
print(var4_1)
print(var5_1)

# ===========================================================
# 使用tf.variable_scope创建命名空间
with tf.variable_scope('scope1'):
    # 使用tf.Variable创建名为var3的变量
    var3_2 = tf.Variable(3.5, name='var3')
    # 使用tf.get_variable得到名为var4的变量
    var4_2 = tf.get_variable(name='var4', initializer=1.0)

# 打印命名空间中的变量
print(var3_2)
print(var4_2)

# reuse=True/tf.AUTO_REUSE
with tf.variable_scope('scope1', reuse=tf.AUTO_REUSE):
    var4_3 = tf.get_variable(name='var4', initializer=10.0)
    var5_2 = tf.get_variable(name='var5', initializer=100)

print(var4_3 == var4_2)
print(var5_2)

with tf.variable_scope('scope_x'):
    with tf.variable_scope('scope_y'):
        reuse_var = tf.get_variable('reuse_var', initializer=1000.0)

with tf.variable_scope('scope_x', reuse=tf.AUTO_REUSE):
    reuse_var2 = tf.get_variable('scope_y/reuse_var', initializer=0.001)

print(reuse_var)
print(reuse_var2)