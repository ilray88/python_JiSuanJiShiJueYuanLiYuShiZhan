import tensorflow as tf

sess = tf.InteractiveSession()

a = tf.constant(1.0)
b = tf.constant(2.5)

c = tf.add(a, b)

result = sess.run(c)
print(result)
