# coding=utf-8
import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1


def net(image, type=0):
    # 该函数构建图像转换网络，image 为步骤 1 中读入的图像 ndarray 阵列，返回最后一层的输出结果
    # 构建图像转换网络，每一层的输出作为下一层的输入
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)

    # 最后一个卷积层的输出再经过 tanh 函数处理，最后的输出张量 preds 像素值需限定在 [0,255] 范围内
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds


def _conv_layer(net, num_filters, filter_size, strides, relu=True, type=0):
    # 该函数定义了卷积层的计算方法，net 为该卷积层的输入 ndarray 数组，num_filters 表示输出通道数，filter_size 表示卷积核尺
    # 寸，strides 表示卷积步长，该函数最后返回卷积层计算的结果

    # 准备好权重的初值
    weights_init = _conv_init_vars(net, num_filters, filter_size)

    # 输入的 strides 参数为标量，需将其处理成卷积函数能够使用的数据形式
    strides_shape = [1, strides, strides, 1]

    # 进行卷积计算
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')

    # 对卷积计算结果进行批归一化处理
    if type == 0:
        net = _batch_norm(net)
    elif type == 1:
        net = _instance_norm(net)

    if relu:
        # 对归一化结果进行 ReLU 操作
        net = tf.nn.relu(net)

    return net


def _conv_tranpose_layer(net, num_filters, filter_size, strides, type=0):
    # 准备好权重的初值
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    # 输入的 num_filters、strides 参数为标量，需将其处理成转置卷积函数能够使用的数据形式
    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    # 进行转置卷积计算
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    # 对卷积计算结果进行批归一化处理
    if type == 0:
        net = _batch_norm(net)
    elif type == 1:
        net = _instance_norm(net)

    # 对归一化结果进行 ReLU 操作
    net = tf.nn.relu(net)

    return net


def _residual_block(net, filter_size=3, type=0):
    # 调用之前实现的卷积层函数，实现残差块的计算
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net


def _batch_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    axes = list(range(len(net.get_shape()) - 1))
    mu, sigma_sq = tf.nn.moments(net, axes, keep_dims=True)
    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    return tf.nn.batch_normalization(net, mu, sigma_sq, shift, scale, epsilon)


def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init