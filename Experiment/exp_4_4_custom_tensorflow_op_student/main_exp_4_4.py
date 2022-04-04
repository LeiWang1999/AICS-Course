import numpy as np
import os
import time
os.environ['MLU_VISIBLE_DEVICES']=""
import tensorflow as tf
np.set_printoptions(suppress=True)
from stu_upload.power_difference_test_cpu import power_difference_op
from stu_upload.power_diff_numpy import *
from stu_upload.transform_cpu import run_ori_pb, run_ori_power_diff_pb, run_numpy_pb

def test_power_diff():
    value = 256
 
    start = time.time()
    input_x = np.loadtxt("./data/in_x.txt")
    input_y = np.loadtxt("./data/in_y.txt")
    input_pow = np.loadtxt("./data/in_z.txt")
    output = np.loadtxt("./data/out.txt")
    end = time.time()
    print("load data cost " + str((end - start)*1000) + "ms")

    input_x = np.reshape(input_x,(1,value,value,3))
    input_y = np.reshape(input_y,(1,1,1,3))
    output = np.reshape(output, (-1))
    
    start = time.time()
    res = power_difference_op(input_x, input_y, input_pow)
    end = time.time() - start
    print("comput C++ op cost "+ str(end*1000) + "ms" )
    res = np.reshape(res,(-1))
    err = sum(abs(res - output))/sum(output)
    print("C++ op err rate= " + str(err*100) + "%")
    if err < 0.003:
        print('TEST C++ OP PASS.')
    else:
        print('TEST C++ OP FAILED.')
        exit(0)
    
    start = time.time()
    res = power_diff_numpy(input_x, input_y, input_pow)
    end = time.time()
    print("comput numpy op cost "+ str((end-start)*1000) + "ms")
    res = np.reshape(res,(-1))
    err = sum(abs(res - output))/sum(output)
    print("numpy op err rate= "+ str(err*100) + "%")
    if err < 0.003:
        print('TEST NUMPY OP PASS.')
    else:
        print('TEST NUMPY OP FAILED.')
        exit(0)

def test_power_diff_with_bigger_data():
    value = 256

    print('loading data ...')
    start = time.time()
    input_x = np.loadtxt("./data/in_x_bigger.txt")
    input_y = np.loadtxt("./data/in_y_bigger.txt")
    input_pow = np.loadtxt("./data/in_z_bigger.txt")
    output = np.loadtxt("./data/out_bigger.txt")
    end = time.time()
    print("load data cost " + str((end - start)*1000) + "ms")

    input_x = np.reshape(input_x,(1,value,value,16))
    input_y = np.reshape(input_y,(1,1,1,16))
    output = np.reshape(output, (-1))

    start = time.time()
    res = power_difference_op(input_x, input_y, input_pow)
    end = time.time() - start
    print("comput C++ op cost "+ str(end*1000) + "ms" )
    res = np.reshape(res,(-1))
    err = sum(abs(res - output))/sum(output)
    print("C++ op err rate= " + str(err*100) + "%")

    if err < 0.003:
        print('TEST C++ OP WITH BIGGER DATA PASS.')
    else:
        print('TEST C++ OP WITH BIGGER DATA FAILED.')
        exit(0)

    start = time.time()
    res = power_diff_numpy(input_x, input_y,input_pow)
    end = time.time()
    print("comput numpy op cost "+ str((end-start)*1000) + "ms")
    res = np.reshape(res,(-1))
    err = sum(abs(res - output))/sum(output)
    print("numpy op err rate= "+ str(err*100) + "%")

    if err < 0.003:
        print('TEST NUMPY OP WITH BIGGER DATA PASS.')
    else:
        print('TEST NUMPY OP WITH BIGGER DATA FAILED.')
        exit(0)

def main():
    test_power_diff()
    print('---------------------')
    test_power_diff_with_bigger_data()
    print('---------------------')
    image_path = './images/chicago.jpg'
    run_ori_pb('./models/pb_models/udnie.pb', image_path)
    run_ori_power_diff_pb('./stu_upload/udnie_power_diff.pb', image_path)
    run_numpy_pb('./stu_upload/udnie_power_diff_numpy.pb', image_path)

if __name__ == '__main__':
    main()