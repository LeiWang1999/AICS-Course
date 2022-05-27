/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#include "cnplugin.h"
#include "spilt_sub_concat_kernel.h"

typedef uint16_t half;

cnmlStatus_t cnmlCreatPluginSBCOpParam(
    cnmlPluginSBCOpParam_t *param,
    int batch_num_
){
    *param = new cnmlPluginSBCOpParam();
    (*param)->batch_num_ = batch_num_;

    return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlDestroyPluginSBCOpParam(
    cnmlPluginSBCOpParam_t *param
    ){
    delete (*param);
    *param = nullptr;

    return CNML_STATUS_SUCCESS;

}

cnmlStatus_t cnmlCreatePluginSBCOp(
    cnmlBaseOp_t *op,
    //cnmlPluginSBCOpParam_t param,
    cnmlTensor_t *SBC_input_tensors,
    cnmlTensor_t *SBC_output_tensors,
    int batch_num_
    ){
    
    //补全cnmlCreatePluginSBCOp
    int input_num = 1;
    int output_num = 1;

    void** InterfacePtr;
    InterfacePtr = reinterpret_cast<void**>(&SBCKernel);

    cnrtKernelParamsBuffer_t params;
    cnrtGetKernelParamsBuffer(&params);
    cnrtKernelParamsBufferMarkInput(params);
    cnrtKernelParamsBufferMarkOutput(params);
    cnrtKernelParamsBufferAddParam(params, &batch_num_, sizeof(int));
    
    cnmlCreatePluginOp(
                        op, 
                        "SBC",
                        InterfacePtr,
                        params,
                        SBC_input_tensors,
                        input_num,
                        SBC_output_tensors,
                        output_num,
                        nullptr,
                        0
    );
                        
    cnrtDestroyKernelParamsBuffer(params);
    return CNML_STATUS_SUCCESS;

}



cnmlStatus_t cnmlComputePluginSBCOpForward(
    cnmlBaseOp_t op,
    void **inputs,
    int input_num,
    void **outputs,
    int output_num,
    cnrtQueue_t queue
    ){
   
    //补全cnmlComputePluginSBCOpForward 
    cnmlComputePluginOpForward_V4(
                                    op,
                                    nullptr,
                                    inputs,
                                    input_num,
                                    nullptr,
                                    outputs,
                                    output_num,
                                    queue,
                                    nullptr
    ); 
    
    return CNML_STATUS_SUCCESS;

}



