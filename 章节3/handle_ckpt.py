import tensorflow.compat.v1 as tf

X = tf.Variable(56.78, dtype=tf.float32, name='X')
Y = tf.Variable(12.34, dtype=tf.float32, name='Y')
Z = tf.placeholder(dtype=tf.float32, name='Z')

# 结果一为X与Y相加
result1 = X + Y
# 结果二为(X + Y) * Z
result2 = result1 * Z

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, 'graph.ckpt')
# ==================================================
new_ckpt = tf.train.NewCheckpointReader('graph.ckpt')
print(new_ckpt.debug_string().decode('utf8'))
print(new_ckpt.get_tensor('X'), new_ckpt.get_tensor('Y'))
# ==================================================
# 运行一下恢复程序，需要将上面定义的计算图注释掉，否则变量名会冲突
# 重新定义计算图，并改变变量的初始值
X = tf.Variable(11.1111, dtype=tf.float32, name='X')
Y = tf.Variable(22.2222, dtype=tf.float32, name='Y')
Z = tf.placeholder(dtype=tf.float32, name='Z')

# 结果一为X与Y相加
result1 = X + Y
# 结果二为(X + Y) * Z
result2 = result1 * Z

var_list = [X]
saver = tf.train.Saver(var_list=var_list)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    last_ckpt = tf.train.latest_checkpoint('.')
    print(last_ckpt)
    saver.restore(sess, last_ckpt)
    
    print(sess.run(X))
    print(sess.run(Y))
    
