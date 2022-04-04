# -*- coding: utf-8 -*-

import numpy as np
import struct
import os
import scipy.io
import time
import tensorflow as tf
from tensorflow.python.framework import graph_util

import scipy.misc

os.putenv('MLU_VISIBLE_DEVICES','')

IMAGE_PATH = 'data/cat1.jpg'
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5',

        'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
    )

    data = scipy.io.loadmat(data_path)
    weights = data['layers'][0]

    net = {}
    current = input_image
    #print(current.shape)
    for i, name in enumerate(layers):
        if name[:4] == 'conv':
            # TODO: 从模型中读取权重、偏置加数，计算卷积结果 current
            #print(name)
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = kernels.transpose(1, 0, 2, 3)
            current = _conv_layer(current, kernels, bias)
            #print(current.shape)
        elif name[:4] == 'relu':
            # TODO: 执行 ReLU 计算，计算结果存入 current
            #print(name)
            current = tf.nn.relu(current)
            #print(current.shape)
        elif name[:4] == 'pool':
            # TODO: 执行 pool 计算，计算结果存入 current
            #print(name)
            current = _pool_layer(current)
            #print(current.shape)
        elif name == 'softmax':
            # TODO: 执行 softmax 计算，计算结果存入 current
            #print(name)
            current = tf.nn.softmax(current, axis=1)
        elif name  == 'fc6':
            # TODO: 执行全连接层计算，计算结果存入 current
            #print(name)
            kernels, bias = weights[i][0][0][0][0]
            current = tf.reshape(current, [-1, current.shape[1]*current.shape[2]*current.shape[3]])
            kernels = tf.reshape(kernels, [kernels.shape[0]*kernels.shape[1]*kernels.shape[2], -1])
            #print(current.shape)
            #print(kernels.shape)
            current = tf.nn.bias_add(tf.matmul(current, kernels), bias.flatten())
            #print(current.shape)
        elif name  == 'fc7':
            #print(name)
            kernels, bias = weights[i][0][0][0][0]
            kernels = tf.reshape(kernels, [kernels.shape[0]*kernels.shape[1]*kernels.shape[2], -1])
            current = tf.nn.bias_add(tf.matmul(current, kernels), bias.flatten())
            #print(current.shape)
        elif name  == 'fc8':
            #print(name)
            kernels, bias = weights[i][0][0][0][0]
            kernels = tf.reshape(kernels, [kernels.shape[0]*kernels.shape[1]*kernels.shape[2], -1])
            current = tf.nn.bias_add(tf.matmul(current, kernels), bias.flatten())
            #print(current.shape)

        net[name] = current 

    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias):
    # TODO: 定义卷积层的操作步骤，input 为输入张量，weights 为权重参数，bias 为偏置参数，返回计算的结果
    conv_result = tf.nn.conv2d(input, weights, 1, "SAME")
    conv_result= tf.nn.bias_add(conv_result, bias.flatten())
    return conv_result

def _pool_layer(input):
    # TODO: 定义最大池化的操作步骤，input 为输入张量，返回池化操作后的计算结果
    pool_result = tf.nn.max_pool(input, 2, 2, "VALID")
    return pool_result

def preprocess(image,mean):
    return image - mean

def load_image(path):
    # TODO: 使用 scipy.misc 模块读入输入图像，调用 preprocess 函数对图像进行预处理，并返回形状为（1,244,244,3）的数组 image
    mean = np.array([123.68, 116.779, 103.939])
    # Get the image, resize the image.
    image = scipy.misc.imresize(scipy.misc.imread(path), (224, 224, 3))
    #print(image.shape)
    # Sub the mean.
    image = preprocess(image, mean).reshape(-1, 224, 224, 3)
    #print(image.shape)
    return image

if __name__ == '__main__':
    input_image = load_image(IMAGE_PATH)

    with tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=(1,224,224,3),
                                         name='img_placeholder')
        # TODO: 调用 net 函数，生成 VGG19 网络模型并保存在 nets 中
        nets = net(VGG_PATH, img_placeholder)

        for i in range(10):
            start = time.time()
            # TODO: 计算 nets
            preds = sess.run(nets, feed_dict={img_placeholder:input_image})
            end = time.time()
            delta_time = end - start	
            print("processing time: %s" % delta_time)

        prob = preds['softmax'][0]
        top1 = np.argmax(prob)

        print('Classification result: id = %d, prob = %f' % (top1, prob[top1]))

        print("*** Start Saving Frozen Graph ***")
        # We retrieve the protobuf graph definition
        input_graph_def = sess.graph.as_graph_def()
        output_node_names = ["Softmax"]
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names,
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile("models/vgg19.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("**** Save Frozen Graph Done ****")