# coding=utf-8
import numpy as np
import struct
import os
import time

def img2col(input, height_out, width_out, kernel_size, stride):
    output = np.zeros([input.shape[0], input.shape[1], kernel_size*kernel_size, height_out*width_out])
    height = (input.shape[2] - kernel_size) / stride + 1
    width = (input.shape[3] - kernel_size) / stride + 1
    for idxh in range(height):
        for idxw in range(width):
            output[:, :, :, idxh * width + idxw] = input[:, :, idxh * stride : idxh * stride + kernel_size, idxw * stride : idxw * stride + kernel_size].reshape(input.shape[0], input.shape[1], -1)
    return output

def col2img(input, height_pad, width_pad, kernel_size, channel, padding, stride):
    output = np.zeros([input.shape[0], channel, height_pad, width_pad])
    input = input.reshape(input.shape[0], channel, -1, input.shape[2])
    height = (height_pad - kernel_size) / stride + 1
    width = (width_pad - kernel_size) / stride + 1
    for idxh in range(height):
        for idxw in range(width):
            output[:, :, idxh * stride : idxh * stride + kernel_size, idxw * stride : idxw * stride + kernel_size] += input[:, :, :, idxh * width + idxw].reshape(input.shape[0], channel, kernel_size, -1)
    return output[:, :, padding : height_pad - padding, padding : width_pad - padding]

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.weight[:, :, :, idxc] * self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size]) + self.bias[idxc]
        self.forward_time = time.time() - start_time
        return self.output
    def forward_speedup(self, input):
        # 改进forward函数，使得计算加速
        start_time = time.time()

        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        self.input_col = img2col(self.input_pad, height_out, width_out, self.kernel_size, self.stride)
        self.weights_col = self.weight.transpose(3, 0, 1, 2).reshape(self.weight.shape[-1], -1)
        output = np.matmul(self.weights_col, self.input_col.reshape(self.input_col.shape[0], -1, self.input_col.shape[3])) + self.bias.reshape(-1, 1)
        self.output = output.reshape(self.input.shape[0], self.channel_out, height_out, width_out)

        self.forward_time = time.time() - start_time
        return self.output
    def backward_speedup(self, top_diff):
        # 改进backward函数，使得计算加速
        start_time = time.time()

        height_pad = self.input.shape[2] + self.padding * 2
        width_pad = self.input.shape[3] + self.padding * 2
        bottom_diff_col = np.matmul(self.weights_col.T, top_diff.transpose(1, 2, 3, 0).reshape(self.channel_out, -1))
        bottom_diff_col = bottom_diff_col.reshape(bottom_diff_col.shape[0], -1, self.input.shape[0]).transpose(2, 0, 1)
        bottom_diff = col2img(bottom_diff_col, height_pad, width_pad, self.kernel_size, self.channel_in, self.padding, self.stride)

        self.backward_time = time.time() - start_time
        return bottom_diff
    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        bottom_diff = bottom_diff[:, :, self.padding : self.padding + self.input.shape[2], self.padding : self.padding + self.input.shape[3]]
        self.backward_time = time.time() - start_time
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = np.max(self.input[idxn, idxc, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size])
                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output
    def forward_speedup(self, input):
        # 改进forward函数，使得计算加速
        start_time = time.time()
        
        self.input = input # [N, C, H, W]
        self.height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        self.width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1

        self.input_col = img2col(self.input, self.height_out, self.width_out, self.kernel_size, self.stride).reshape(self.input.shape[0], self.input.shape[1], -1, self.height_out, self.width_out)
        output = self.input_col.max(axis=2, keepdims=True)
        self.max_index = (self.input_col == output)
        self.output = output.reshape(self.input.shape[0], self.input.shape[1], self.height_out, self.width_out)

        return self.output
    def backward_speedup(self, top_diff):
        # 改进backward函数，使得计算加速

        pool_diff = (self.max_index * top_diff[:, :, np.newaxis, :, :]).reshape(self.input.shape[0], -1, self.height_out * self.width_out)
        bottom_diff = col2img(pool_diff, self.input.shape[2], self.input.shape[3], self.kernel_size, self.input.shape[1], 0, self.stride)

        return bottom_diff
    def backward_raw_book(self, top_diff):
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失
                        max_index = np.argwhere(self.max_index[idxn, idxc, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size])[0]
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = top_diff[idxn, idxc, idxh, idxw]
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff
