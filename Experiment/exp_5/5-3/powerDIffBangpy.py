# Copyright (C) [2020] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Bangpy powerdiff code"""
import os, shutil
import numpy as np
import bangpy
from bangpy import tcp, load_module
SHAPE = 256

def power_diff():
    def verify_bp(dtype):
        # TCP容器定义
        bp = tcp.TCP()
        # 内建变量及可变参数声明
        len = bp.Var("len")
        pow = bp.Var("pow")
        core_id = bp.builtin_var("coreId")
        core_dim= bp.builtin_var("coreDim")
        cluster_id = bp.builtin_var("clusterId")
        task_id = bp.Scalar(dtype=bangpy.int32, name="task_id", value=cluster_id * core_dim + core_id)
        # TODO： 计算分片
        quotient = len / SHAPE
        rem = len % SHAPE
        
        # TODO: 条件判断，确保单核运行
        with bp.if_scope(task_id==0):
            # 张量定义
            input1 = bp.Tensor(shape=(len,), name="input1",
                               dtype=dtype, scope="global")
            input2 = bp.Tensor(shape=(len,), name="input2",
                               dtype=dtype, scope="global")
            output = bp.Tensor(shape=(len,), name="output",
                               dtype=dtype, scope="global")
            input1_nram = bp.Tensor(shape=(SHAPE,), name="input1_nram",
                                    dtype=dtype, scope="nram")
            input2_nram = bp.Tensor(shape=(SHAPE,), name="input2_nram",
                                    dtype=dtype, scope="nram")
            # TODO：条件与循环控制
            with bp.for_range(0, quotient) as i:
                # TODO：数据拷入操作 gdram -> nram
                bp.memcpy(input1_nram, input1[i*SHAPE:(i+1)*SHAPE])
                bp.memcpy(input2_nram, input2[i*SHAPE:(i+1)*SHAPE])
                # TODO：计算描述
                bp.subtract(input1_nram, input1_nram, input2_nram)
                bp.memcpy(input2_nram, input1_nram)
                with bp.for_range(0, pow-1) as j:
                    bp.multiply(input1_nram, input1_nram, input2_nram)
                # TODO：数据拷出操作 nram -> gdram
                bp.memcpy(output[i*SHAPE:(i+1)*SHAPE], input1_nram)
                        # TODO：条件与循环控制
            with bp.if_scope(rem != 0):
                # TODO：数据拷入操作 gdram -> nram
                bp.memcpy(input1_nram[:rem], input1[quotient*SHAPE:rem+quotient*SHAPE])
                bp.memcpy(input2_nram[:rem], input2[quotient*SHAPE:rem+quotient*SHAPE])
                # TODO：计算描述
                bp.subtract(input1_nram[:rem], input1_nram[:rem], input2_nram[:rem])
                bp.memcpy(input2_nram[:rem], input1_nram[:rem])
                with bp.for_range(0, pow-1) as j:
                    bp.multiply(input1_nram[:rem], input1_nram[:rem], input2_nram[:rem])
                # TODO：数据拷出操作 nram -> gdram
                bp.memcpy(output[quotient*SHAPE:rem+quotient*SHAPE], input1_nram[:rem])
        # BPL编译           
        f = bp.BuildBANG(inputs=[input1, input2, len, pow], outputs=[output],
                         kernel_name="PowerDifferenceKernel")
        return f
    
    def check_target():
        fvec = verify_bp(bangpy.float16)
        if os.path.exists('./test'):
            shutil.rmtree('./test')
        fvec.save('./test/')
        new_line = []
        with open('./test/device.mlu', "r+") as f:
            line_num = 0
            for line in f:
                if line_num == 0:
                    new_line.append(
                        "__mlu_entry__ void PowerDifferenceKernel( half* input1, half* input2, int pow, half* output, int len) {\n"
                    )
                    line_num = 1
                else:
                    new_line.append(line)
        with open('./test/device.mlu', "w+") as f:
            f.writelines(new_line)
        os.system('cp ./test/device.mlu plugin_power_difference_kernel.mlu')

    check_target()

if __name__ == "__main__":
    power_diff()
