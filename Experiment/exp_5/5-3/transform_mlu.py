# coding=utf-8
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import argparse
import numpy as np
import cv2 as cv
import time
from power_diff_numpy import *

os.putenv('MLU_VISIBLE_DEVICES','0')
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('ori_pb')
    parser.add_argument('ori_power_diff_pb')
    parser.add_argument('numpy_pb')
    args = parser.parse_args()
    return args
def run_ori_pb():
    args = parse_arg()
    config = tf.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    # TODO：完成MLU Config配置
    # ...
    model_name = os.path.basename(args.ori_pb).split(".")[0]
    image_name = os.path.basename(args.image).split(".")[0]
    config.mlu_options.offline_model_name = '../../models/offline_models/' + model_name + '.cambricon'
    # config.mlu_options.save_offline_model = True
    g = tf.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(args.ori_pb,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(args.image)
        X = cv.resize(img, (256, 256))

        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            sess.run(tf.global_variables_initializer())

            input_tensor = sess.graph.get_tensor_by_name('X_content:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')

            start_time = time.time()
            ret =sess.run(output_tensor, feed_dict={input_tensor:[X]})
            end_time = time.time()
            print("C++ inference(MLU) origin pb time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_mlu.jpg',img_numpy)



def run_ori_power_diff_pb():
    args = parse_arg()
    config = tf.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    # TODO：完成MLU Config配置
    # ...
    model_name = os.path.basename(args.ori_power_diff_pb).split(".")[0]
    image_name = os.path.basename(args.image).split(".")[0]
    config.mlu_options.offline_model_name = '../../models/offline_models/' + model_name + '.cambricon'
    config.mlu_options.save_offline_model = True

    g = tf.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(args.ori_power_diff_pb,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(args.image)
        X = cv.resize(img, (256, 256))

        with tf.Session(config=config) as sess:
            # TODO：完成PowerDifference Pb模型的推理
            sess.graph.as_default()
            sess.run(tf.global_variables_initializer())
            input_tensor = sess.graph.get_tensor_by_name('X_content:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')
            input_pow = sess.graph.get_tensor_by_name('moments_15/PowerDifference_z:0')

            start_time = time.time()
            ret =sess.run(output_tensor, feed_dict={input_tensor:[X], input_pow:2.0})
            end_time = time.time()
            print("C++ inference(MLU) time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_mlu.jpg',img_numpy)



def run_numpy_pb():
    args = parse_arg()
    config = tf.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    # TODO：完成MLU Config配置
    # ...
    model_name = os.path.basename(args.numpy_pb).split(".")[0]
    image_name = os.path.basename(args.image).split(".")[0]

    g = tf.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(args.numpy_pb,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(args.image)
        X = cv.resize(img, (256, 256))
        with tf.Session(config=config) as sess:
            # TODO：完成Numpy版本 Pb模型的推理
            sess.graph.as_default()
            sess.run(tf.global_variables_initializer())
            input_tensor = sess.graph.get_tensor_by_name('X_content:0')
            out_tmp_tensor_1 = sess.graph.get_tensor_by_name('Conv2D_13:0')
            out_tmp_tensor_2 = sess.graph.get_tensor_by_name('moments_15/StopGradient:0')
            input_pow = 2
            start_time = time.time()
            out_tmp_tensor_1, out_tmp_tensor_2 = sess.run([out_tmp_tensor_1, out_tmp_tensor_2], feed_dict = {input_tensor:[X]})
            output = power_diff_numpy(out_tmp_tensor_1, out_tmp_tensor_2, input_pow)
            input_tensor_new = sess.graph.get_tensor_by_name('moments_15/PowerDifference:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')
            ret = sess.run(output_tensor, feed_dict={input_tensor:[X], input_tensor_new:output})

            end_time = time.time()
            print("Numpy inference(MLU) time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_mlu.jpg',img_numpy)



if __name__ == '__main__':
    run_ori_pb()
    run_ori_power_diff_pb()
    run_numpy_pb()
