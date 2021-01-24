import tensorflow as tf

print(tf.__version__) # tf 1.15.0

_print_str = tf.constant("Hello, Tensorflow!")
_tf_print = tf.print(_print_str)
with tf.Session() as sess:
    sess.run(_tf_print)