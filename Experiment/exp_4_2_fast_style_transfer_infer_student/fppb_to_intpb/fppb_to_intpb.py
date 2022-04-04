from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
import ConfigParser
from tensorflow.contrib import camb_quantize
import calibrate_data

def read_pb(input_model_name):
    input_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(input_model_name,"rb") as f:
        input_graph_def.ParseFromString(f.read())
        f.close()
    return input_graph_def

def create_pb(output_graph_def, output_model_name):
    with tf.io.gfile.GFile(output_model_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())
        f.close()
    print("cpu_pb transform to mlu_int_pb finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('params_file',
                         help = 'Please input params_file(xxx.ini).')

    args = parser.parse_args()
    model_name = args.params_file.rsplit("/", 1)[-1]
    config = ConfigParser.ConfigParser()
    config.read(args.params_file)

    model_param = dict(config.items('model'))
    data_param = dict(config.items('data'))
    preprocess_param = dict(config.items('preprocess'))
    config_param = dict(config.items('config'))
    calibrate_param = {}

    # model_param
    input_graph_def = read_pb(model_param['original_models_path'])
    input_tensor_names = model_param['input_tensor_names'].replace(" ","").split(",")
    output_tensor_names = model_param['output_tensor_names'].replace(" ","").split(",")
    output_model_name = model_param['save_model_path']
    calibrate_param["input_tensor_names"] = input_tensor_names

    # data_param
    num_runs = int(data_param['num_runs'])
    batch_size = int(data_param['batch_size'])
    for key, value in data_param.items():
      calibrate_param[key] = value

    # preprocess_param
    calibration = preprocess_param['calibration']
    for key, value in preprocess_param.items():
      calibrate_param[key] = value

    # config_param
    int_op_list = ["Conv", "FC", "LRN"]
    if 'int_op_list' in config_param:
      int_op_list = config_param['int_op_list'].replace(" ","").split(",")
    use_convfirst = False
    if 'use_convfirst' in config_param:
      use_convfirst = True if config_param['use_convfirst'].lower() in ['true', '1'] \
	                else False
    device_mode = 'clean'
    if 'device_mode' in config_param:
      device_mode = config_param['device_mode']
    if use_convfirst:
      means = preprocess_param['mean'].replace(" ","").split(",")
      means = [ float(mean) for mean in means ]
      input_std = preprocess_param['std'].replace(" ","").split(",")
      input_std = [ float(std) for std in input_std ]
      color_mode = preprocess_param['color_mode']
      convfirst_params = {
            'color_mode' : color_mode,
            'mean_r' : means[0],
            'mean_g' : means[1],
            'mean_b' : means[2],
            'input_std' : input_std[0]
            }
    else:
      convfirst_params = {}

    channel_quantization = False
    if 'channel_quantization' in config_param:
        channel_quantization = True if config_param['channel_quantization'].lower() in ['true', '1'] \
                else False
    weight_quantization_alg = "naive"
    if 'weight_quantization_alg' in config_param:
        weight_quantization_alg = config_param['weight_quantization_alg']
    activation_quantization_alg = "naive"
    if 'activation_quantization_alg' in config_param:
        activation_quantization_alg = config_param['activation_quantization_alg']
    quantization_type = camb_quantize.QuantizeGraph.QUANTIZATION_TYPE_INT8
    if 'quantization_type' in config_param:
        if "int8" == config_param['quantization_type'].lower():
            quantization_type = camb_quantize.QuantizeGraph.QUANTIZATION_TYPE_INT8
        elif "int16" == config_param['quantization_type'].lower():
            quantization_type = camb_quantize.QuantizeGraph.QUANTIZATION_TYPE_INT16
        else:
            raise Exception("Quantization type: {} not supported".format(config_param['quantization_type']))
    debug = False
    if 'debug' in config_param:
      debug = True if config_param["debug"].lower() in ['true', 1] else False

    print("input_tensor_names:           {}".format(input_tensor_names))
    print("output_tensor_names:          {}".format(output_tensor_names))
    print("batch_size:                   {}".format(batch_size))
    print("num_runs:                     {}".format(num_runs))
    print("int_op_list:                  {}".format(int_op_list))
    print("use_convfirst:                {}".format(use_convfirst))
    print("device_mode:                  {}".format(device_mode))
    print("quantization type:            {}".format(quantization_type))
    print("channel_quantization:         {}".format(channel_quantization))
    print("weight_quantization_alg:      {}".format(weight_quantization_alg))
    print("activation_quantization_alg:  {}".format(activation_quantization_alg))
    print("debug:                        {}".format(debug))

    fixer = camb_quantize.QuantizeGraph(
                   input_graph_def = input_graph_def,
                   output_tensor_names = output_tensor_names,
                   use_convfirst = use_convfirst,
                   convfirst_params = convfirst_params,
                   quantization_type = quantization_type,
                   channel_quantization = channel_quantization,
                   weight_quantization_alg = weight_quantization_alg,
                   activation_quantization_alg = activation_quantization_alg,
                   device_mode = device_mode,
                   int_op_list = int_op_list,
                   debug=debug,
                   model_name = model_name)


    cali_data = calibrate_data.get_calibrate_data(calibrate_param, calibration)
    fixer.get_input_max_min(num_runs, cali_data.next)

    output_graph_def = fixer.rewrite_int_graph()

    if not os.path.exists(os.path.dirname(output_model_name)):
      os.system("mkdir -p {}".format(os.path.dirname(output_model_name)))
    create_pb(output_graph_def, output_model_name)
