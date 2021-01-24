import tensorflow as tf
import time

print(tf.__version__)  # tf 1.15.0

def _process(target):
    start = time.time()
    with tf.device(target):
        A = tf.Variable(tf.random.normal((4000,4000)), dtype=tf.float32)
        B = tf.Variable(tf.random.normal((4000,4000)), dtype=tf.float32)
        C = tf.multiply(A, B)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(C))
    print(target, "process time", time.time() - start, 's')
    
if __name__ == '__main__':
    _process('cpu')
    _process('gpu') if tf.test.is_gpu_available() else print("no gpu avaliable")