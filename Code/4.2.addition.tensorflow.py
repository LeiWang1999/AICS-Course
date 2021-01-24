import tensorflow as tf

print(tf.__version__) # tf 1.15.0

A = tf.constant(2, dtype=tf.float32)
B = tf.placeholder(tf.float32)
C = tf.add(A, B)
with tf.Session() as sess:
    print(sess.run(C, feed_dict={B: 1}))
