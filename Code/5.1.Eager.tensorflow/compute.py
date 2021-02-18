import tensorflow as tf

print(tf.__version__)
tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())

A = [[2.]]
B = [[3.]]
m = tf.matmul(A, B)
print(m)

n = tf.add(A, B)
print(n)