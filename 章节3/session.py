import tensorflow.compat.v1 as tensorflow
# 导入上一节所定义的计算图
from define_graph import *

# 方法1：手动开启/关闭会话
# sess = tf.Session()
# ... 运行计算图
# 关闭会话
# sess.close()

# 方法2：使用with语句让程序自动管理变量（推荐）
with tf.Session() as sess:
    # 运行result1（仅依赖X与Y变量），为X和Y变量分别赋值1和2
    r1 = sess.run(result1, feed_dict={X: 1, Y: 2})

    # 运行result2（依赖X、Y与Z变量），为X和Y变量分别赋值1、2和3
    r2 = sess.run(result2, feed_dict={X: 1, Y: 2, Z: 3})

    # 运行result1和result2，为X、Y、Z变量分别赋值4、5、6
    r3 = sess.run([result1, result2], feed_dict={X: 4, Y: 5, Z: 6})

    # 打印不同的运行结果
    print(r1, r2, r3)

    # 运行result1和result2，为X、Y变量分别赋值7、8
    r4 = sess.run([result1, result2], feed_dict={X: 7, Y: 8})
