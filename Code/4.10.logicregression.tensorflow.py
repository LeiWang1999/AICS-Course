import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
num_points = 1000
vectors_set = []
for i in range (num_points):
    x1 = np.random.normal (0.0, 0.6)
    y1 = x1*0.5+0.3+np.random.normal(0.0,0.3)
    vectors_set.append([x1,y1])
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
# display
plt.scatter (x_data,y_data, c='r')
plt.show()

#生成w,1 维的矩阵，取值[-1,1] 之间的随机数,b 常数
W = tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')
b = tf.Variable(tf.zeros([1]),name='b')
y = W * x_data + b

# 以预测值 y 和实际值之间的均方差作为损失
loss = tf.reduce_mean(tf.square(y-y_data),name='loss')
# 采用梯度下降来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss,name='train')
epochs = 100
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print ('W=', sess.run(W), 'b=', sess.run(b),'loss=', sess.run(loss))
  
    for seg in range (epochs):
        sess.run(train)
        print ('W=', sess.run(W), 'b=', sess.run(b),'loss=', sess.run(loss))
 
    print ('W=', sess.run(W), 'b=', sess.run(b),'loss=', sess.run(loss)) 
    plt.scatter(x_data, y_data, c='r')
    plt.plot(x_data,sess.run(W)*x_data+sess.run(b))
plt.show()