import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-1.0 * x))

def tanh(x):
    return (np.exp(x)-np.exp(-1.0 * x))/(np.exp(x)+np.exp(-1.0 * x))

def ReLU(x):
    return np.maximum(0, x)

def PReLU(x, a=-0.01):
    return np.maximum(a*x, x)

def ELU(x, a=-0.01):
    y = np.zeros_like(x)
    for index, each in enumerate(x):
        y[index] = each if each>0 else a*(np.exp(each)-1.0)
    return y

x=np.linspace(-10,10,256,endpoint=True)#-π to+π的256个值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_ReLU = ReLU(x)
y_PReLU = PReLU(x)
y_ELU = ELU(x)
plt.grid() # 生成网格
plt.plot(x, y_sigmoid, label='sigmoid')
plt.plot(x, y_tanh, label='tanh')
plt.plot(x, y_ReLU, label='ReLU')
plt.plot(x, y_PReLU, label='PReLU')
plt.plot(x, y_ELU, label='ELU')
plt.legend(['sigmoid','tanh', 'ReLU', 'PReLU', 'ELU'])
plt.show()