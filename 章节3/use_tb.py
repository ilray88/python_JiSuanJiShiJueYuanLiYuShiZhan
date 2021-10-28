import tensorflow.compat.v1 as tf

# X = tf.Variable(56.78, dtype=tf.float32, name='X')
# Y = tf.Variable(12.34, dtype=tf.float32, name='Y')
# Z = tf.placeholder(dtype=tf.float32, name='Z')

# # 结果一为X与Y相加
# result1 = X + Y
# # 结果二为(X + Y) * Z
# result2 = result1 * Z

# # 创建一个summary的IO对象，并将计算图添加到summary中
# writer = tf.summary.FileWriter('summary', graph=tf.get_default_graph())

# with tf.Session() as sess:
#     tf.global_variables_initializer().run()

# # 关闭IO对象
# writer.close()
# ===========================================================
# i = tf.Variable(1)

# writer = tf.summary.FileWriter('summary_1', graph=tf.get_default_graph())

# assign_op = tf.assign(i, i + 1)

# tf.summary.scalar('i', i)
# merged_op = tf.summary.merge_all()

# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for e in range(100):
#         sess.run(assign_op)
#         summ = sess.run(merged_op)
#         writer.add_summary(summ, e)    
# writer.close()
# ===========================================================
with open('1.jpg', 'rb') as f:
    data = f.read()
 
# 图像解码节点，3通道jpg图像
image = tf.image.decode_jpeg(data, channels=3)

# 确保图像以4维张量的形式表示
image = tf.stack([image] * 3)

# 添加到日志中
tf.summary.image("image1", image)
merged_op = tf.summary.merge_all()

writer = tf.summary.FileWriter('summary_2', graph=tf.get_default_graph())

with tf.Session() as sess:
    # 运行并写入日志
    summ = sess.run(merged_op)
    writer.add_summary(summ)
writer.close()
