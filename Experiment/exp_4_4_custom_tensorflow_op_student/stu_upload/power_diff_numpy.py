# -*- coding: utf-8 -*-

import numpy as np

def power_diff_numpy(input_x,input_y,input_z):
    # TODO:完成numpy实现的过程，参考实验教程示例
    # Reshape操作，根据实时风格迁移模型的实际情况，该函数假设input_x和input_y的
    # 最后一个维度的dim_size相同，input_y除了最后的维度，其余dim_size均为1.
    x_shape = np.shape(input_x)
    y_shape = np.shape(input_y)
    x = np.reshape(input_x, (-1, y_shape[-1]))
    x_new_shape = np.shape(x)
    y = np.reshape(input_y, (-1))
    output = []
    # 通过for循环完成计算，每次循环计算y个数的PowerDifference操作
    for i in range(x_new_shape[0]):
        difference = x[i] - y
        power_difference = difference ** input_z
        output.append(power_difference)
    output = np.array(output)
    return output

