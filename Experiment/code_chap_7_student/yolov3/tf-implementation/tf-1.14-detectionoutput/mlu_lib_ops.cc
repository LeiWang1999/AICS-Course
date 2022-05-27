/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/mlu.h"
#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/util.h"
#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_common.h"
#include "third_party/mlu/include/bangc_kernel.h"
#include "third_party/mlu/include/cnplugin.h"

namespace stream_executor {
namespace mlu {
namespace lib {


tensorflow::Status CreateAbsOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateAbsOp(op, input, output));
}

tensorflow::Status ComputeAbsOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output) {
  CNML_RETURN_STATUS(cnmlComputeAbsOpForward_V4(op, nullptr, input, nullptr,
                                                output, queue, nullptr));
}


tensorflow::Status CreateActiveOp(MLUBaseOp** op, MLUActiveFunction function,
                                  MLUTensor* input, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateActiveOp(op, function, input, output));
}

tensorflow::Status ComputeActiveOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* inputs, void* outputs) {
  CNML_RETURN_STATUS(cnmlComputeActiveOpForward_V4(op, nullptr, inputs, nullptr,
                                              outputs, queue, nullptr));
}

tensorflow::Status CreateAddOp(MLUBaseOp** op, MLUTensor* input1,
                               MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateAddOp(op, input1, input2, output));
}
tensorflow::Status ComputeAddOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input1, void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeAddOpForward_V4(op, nullptr, input1, nullptr,
                                                input2, nullptr, output, queue,
                                                nullptr));
}


tensorflow::Status CreateAndOp(MLUBaseOp** op, MLUTensor* input1,
                               MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateAndOp(op, input1, input2, output));
}
tensorflow::Status ComputeAndOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input1, void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeAndOpForward_V4(op, nullptr, input1,
        nullptr, input2, nullptr, output, queue, nullptr));
}


void GetArgmaxOpOutputDim(MLUDimension_t argmax_axis, int ni, int ci, int hi,
                          int wi, int *no, int *co, int *ho, int *wo) {
  if (argmax_axis == MLU_DIM_N) {
    (*no) = 1;
    (*co) = ci;
    (*ho) = hi;
    (*wo) = wi;
  } else if (argmax_axis == MLU_DIM_C) {
    (*no) = ni;
    (*co) = 1;
    (*ho) = hi;
    (*wo) = wi;
  } else if (argmax_axis == MLU_DIM_H) {
    (*no) = ni;
    (*co) = ci;
    (*ho) = 1;
    (*wo) = wi;
  } else if (argmax_axis == MLU_DIM_W) {
    (*no) = ni;
    (*co) = ci;
    (*ho) = hi;
    (*wo) = 1;
  }
}

tensorflow::Status CreateArgmaxOp(MLUBaseOp **op, int argmax_axis,
                                  MLUTensor *input, MLUTensor *output) {
  CNML_RETURN_STATUS(cnmlCreateNdArgmaxOp(op, argmax_axis, input, output));
}

tensorflow::Status ComputeArgmaxOp(MLUBaseOp *op, MLUCnrtQueue *queue,
                                   void *inputs, void *outputs) {
  CNML_RETURN_STATUS(cnmlComputeNdArgmaxOpForward_V2(op, nullptr, inputs,
                                     nullptr, outputs, queue, nullptr));
}


tensorflow::Status CreateBatch2SpaceOp(MLUBaseOp** op, int w_block_size,
                                       int h_block_size, MLUTensor* input,
                                       MLUTensor* output) {
  CNML_RETURN_STATUS(
      cnmlCreateBatch2spaceOp(op, w_block_size, h_block_size, input, output));
}

tensorflow::Status ComputeBatch2SpaceOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* inputs, void* outputs) {
  CNML_RETURN_STATUS(cnmlComputeBatch2spaceOpForward_V4(op, nullptr, inputs,
                                        nullptr, outputs, queue, nullptr));
}

tensorflow::Status CreateBatchMatMulOp(MLUBaseOp** op, MLUTensor* in0,
                                       MLUTensor* in1, bool adj_x, bool adj_y,
                                       MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateBatchDotOp(op, in0, in1, output, adj_x, adj_y));
}

tensorflow::Status ComputeBatchMatMulOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* input0, void* input1,
                                        void* output) {
  CNML_RETURN_STATUS(cnmlComputeBatchDotOpForward_V4(
      op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}


tensorflow::Status CreateBatchNormOp(MLUBaseOp** op, MLUTensor* input,
                                     MLUTensor* mean, MLUTensor* var,
                                     MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateBatchNormOp(op, input, output, mean, var));
}

tensorflow::Status CreateNdBatchNormOp(MLUBaseOp** op, MLUTensor* input,
                                     MLUTensor* mean, MLUTensor* var,
                                     MLUTensor* output, int dim) {
  CNML_RETURN_STATUS(cnmlCreateNdBatchNormOp(op, dim, input,
                                        output, mean, var));
}

tensorflow::Status ComputeBatchNormOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* inputs, void* outputs) {
  CNML_RETURN_STATUS(cnmlComputeBatchNormOpForward_V4(op, nullptr, inputs,
                                        nullptr, outputs, queue, nullptr));
}

tensorflow::Status CreateBertSquadOp(MLUBaseOp** op,
                                     MLUTensor** inputs, MLUTensor** outputs,
                                     MLUTensor** static_tensors,
                                     int static_tensors_num,
                                     int batch_num, int seq_len) {
  CNML_RETURN_STATUS(cnmlCreatePluginBertSquadOp(op, inputs, outputs,
                                        static_tensors, static_tensors_num,
                                        batch_num, seq_len));
}

tensorflow::Status ComputeBertSquadOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void** inputs, void** outputs) {
  CNML_RETURN_STATUS(cnmlComputePluginBertSquadOpForward(op,
                                       nullptr, inputs,
                                       nullptr, outputs,
                                       queue, nullptr));
}


tensorflow::Status CreateBroadcastAddOp(MLUBaseOp** op, MLUTensor* input1,
                                        MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateBroadcastAddOp(op, input1, input2, output));
}

tensorflow::Status ComputeBroadcastAddOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input1, void* input2,
                                         void* output) {
  CNML_RETURN_STATUS(cnmlComputeBroadcastAddOpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}


tensorflow::Status CreateBroadcastMulOp(MLUBaseOp** op, MLUTensor* input1,
                                        MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateBroadcastMultOp(op, input1, input2, output));
}

tensorflow::Status ComputeBroadcastMulOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input1, void* input2,
                                         void* output) {
  CNML_RETURN_STATUS(cnmlComputeBroadcastMultOpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}


tensorflow::Status CreateBroadcastOp(MLUBaseOp** op, MLUTensor* input,
                                     MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdBroadcastOp(op, input, output));
}

tensorflow::Status ComputeBroadcastOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* inputs, void* outputs) {
  CNML_RETURN_STATUS(cnmlComputeNdBroadcastOpForward_V2(op, nullptr, inputs,
                                          nullptr, outputs, queue, nullptr));
}


tensorflow::Status CreateBroadcastSubOp(MLUBaseOp** op, MLUTensor* input1,
                                        MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateBroadcastSubOp(op, input1, input2, output));
}

tensorflow::Status ComputeBroadcastSubOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input1, void* input2,
                                         void* output) {
  CNML_RETURN_STATUS(cnmlComputeBroadcastSubOpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}


tensorflow::Status CreateCastOp(MLUBaseOp** op, MLUCastType cast_type,
                                MLUTensor* input, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateCastOp(op, cast_type, input, output));
}

tensorflow::Status ComputeCastOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeCastOpForward_V4(op, nullptr, input,
        nullptr, output, queue, nullptr));
}


tensorflow::Status CreateClipOp(MLUBaseOp** op, MLUTensor* input,
                                float lower_bound, float upper_bound,
                                MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateClipOp(op, input, output,
        lower_bound, upper_bound));
}

tensorflow::Status ComputeClipOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeClipOpForward_V4(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateConcatOp(MLUBaseOp** op, int dim,
    MLUTensor* inputs[], int input_num, MLUTensor* output) {
  CNML_RETURN_STATUS(
      cnmlCreateNdConcatOp(op, dim, inputs, input_num, &output, 1));
}

tensorflow::Status ComputeConcatOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* inputs[], int input_num, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdConcatOpForward_V2(
      op, NULL, inputs, input_num, NULL, &output, 1, queue, NULL));
}

tensorflow::Status CreateDepthwiseConvOp(MLUBaseOp** op,
    MLUTensor* input,  MLUTensor* output,
    MLUTensor* filter, MLUTensor* bias,
    int stride_height, int stride_width,
    int pad_height, int pad_width) {
  MLUConvDepthwiseOpParam* depthwise_conv_param;
  TF_CNML_CHECK(cnmlCreateConvDepthwiseOpParam_V2(&depthwise_conv_param,
      stride_height, stride_width, pad_height, pad_width));
  TF_CNML_CHECK(cnmlCreateConvDepthwiseOp(op,
      depthwise_conv_param, input, output,
      filter, bias));
  TF_CNML_CHECK(cnmlDestroyConvDepthwiseOpParam(&depthwise_conv_param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeDepthwise_ConvOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeConvDepthwiseOpForward_V4(op,
        nullptr, input,
        nullptr, output,
        queue, nullptr));
}

tensorflow::Status CreateConv2DOp(MLUBaseOp** op,
     MLUTensor* input, MLUTensor* output, MLUTensor* filter,
     MLUTensor* bias, int stride_height, int stride_width,
     int dilation_height, int dilation_width,
     int pad_height, int pad_width) {
   MLUConvOpParam* conv_param;
   TF_CNML_CHECK(cnmlCreateConvOpParam(&conv_param, stride_height, stride_width,
       dilation_height, dilation_width, pad_height, pad_width));
   TF_CNML_CHECK(cnmlCreateConvOp(op, conv_param, input, output, filter, bias));
   TF_CNML_CHECK(cnmlDestroyConvOpParam(&conv_param));
   return tensorflow::Status::OK();
 }

 tensorflow::Status ComputeConv2DOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output) {
   CNML_RETURN_STATUS(cnmlComputeConvOpForward_V4(op,
         nullptr, input,
         nullptr, output,
         queue, nullptr));
 }

tensorflow::Status CreateConvFirstOp(MLUBaseOp** op,
    MLUConvFirstOpParam* param,
    MLUTensor* input, MLUTensor* mean, MLUTensor* output,
    MLUTensor* filter, MLUTensor* bias, MLUTensor* std) {
  CNML_RETURN_STATUS(cnmlCreateConvFirstOp(
      op, param, input,
      mean , output, filter, bias, std));
}

tensorflow::Status ComputeConvFirstOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeConvFirstOpForward_V4(op,
        nullptr, input,
        nullptr, output,
        queue, nullptr));
}

tensorflow::Status CreateConv2DBackpropInputOp(MLUBaseOp** op,
                                  MLUTensor* w, MLUTensor* dy,
                                  MLUTensor* w_param, MLUTensor* dy_param,
                                  MLUTensor* output,
                                  int sh, int sw, int dh, int dw,
                                  int pad_top, int pad_bottom,
                                  int pad_left, int pad_right) {
  MLULOG(3) << "cnmlCreateConvOpBackwardDataParam: " <<
    "sh: " << sh << " sw: " << sw << " dh: " << dh << " dw: " << dw <<
    " pad_top: " << pad_top << " pad_bottom: " << pad_bottom << " pad_left: "
    << pad_left << " pad_right: " << pad_right;
  cnmlConvOpBackwardDataParam_t convbp_param;
  TF_CNML_CHECK(cnmlCreateConvOpBackwardDataParam(&convbp_param, sh, sw,
        dh, dw, pad_top, pad_bottom, pad_left, pad_right));
  TF_CNML_CHECK(cnmlCreateConvOpBackwardData(op, convbp_param,
        dy, dy_param, output, w, w_param));
  CNML_RETURN_STATUS(cnmlDestroyConvOpBackwardDataParam(&convbp_param));
}

tensorflow::Status ComputeConv2DBackpropInputOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* w, void* dy,
    void* w_param, void* dy_param, void* output) {
  MLUInvokeFuncParam_t compute_forw_param = DefaultInvokeParam();
  CNML_RETURN_STATUS(cnmlComputeConvOpBackwardData(op,
        dy, dy_param, w_param, w, output,
        &compute_forw_param, queue));
}

tensorflow::Status CreateCropOp(MLUBaseOp** op, MLUTensor* input,
                                 MLUTensor* output, int startIndexOfN,
                                 int startIndexOfC, int startIndexOfH,
                                 int startIndexOfW, float space_number) {
  MLUCropOpParam* param;

  TF_CNML_CHECK(cnmlCreateCropOpParam(&param, startIndexOfN, startIndexOfC,
                                      startIndexOfH, startIndexOfW,
                                      space_number));
  TF_CNML_CHECK(cnmlCreateCropOp(op, param, input, output));
  TF_CNML_CHECK(cnmlDestroyCropOpParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeCropOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeCropOpForward_V4(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCropAndResizeOp(MLUBaseOp** op,
                                         MLUTensor* input,
                                         MLUTensor* boxes,
                                         MLUTensor* box_ind,
                                         int crop_height,
                                         int crop_width,
                                         float extrapolation_value,
                                         MLUTensor* output) {
  cnmlPluginResizeAndColorCvtParam_t params;
  MLUTensorUtil input_tensor_util(input);
  int batch_num = input_tensor_util.dim_size(0);
  int input_channel = input_tensor_util.dim_size(3);
  int input_height = input_tensor_util.dim_size(1);
  int input_width = input_tensor_util.dim_size(2);
  MLUTensorUtil boxes_tensor_util(boxes);
  int box_number = boxes_tensor_util.dim_size(0);
  int pad_size = 64;
  MLUCoreVersion core_ver = static_cast<MLUCoreVersion>(5);
  cnmlCreatePluginCropFeatureAndResizeOpParam(&params,
        input_height, input_width, crop_height, crop_width, batch_num,
        input_channel, box_number, pad_size, core_ver);

  const int input_num = 3;
  const int output_num = 1;
  MLUTensor* cnml_inputs[input_num];
  MLUTensor* cnml_outputs[output_num];
  cnml_inputs[0] = input;
  cnml_inputs[1] = boxes;
  cnml_inputs[2] = box_ind;
  cnml_outputs[0] = output;
  TF_CNML_CHECK(cnmlCreatePluginCropFeatureAndResizeOp(op, &params,
        cnml_inputs, cnml_outputs));
  TF_CNML_CHECK(cnmlDestroyPluginCropFeatureAndResizeOpParam(&params));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeCropAndResizeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                          void* input, void* boxes,
                                          void* box_ind, void* output) {
  void* in_addr[3] = {input, boxes, box_ind};
  void* out_addr[] = {output};
  cnrtInvokeFuncParam_t compute_forw_param;
  int dp = 1;
  u32_t affinity = 0x01;
  compute_forw_param.data_parallelism = &dp;
  compute_forw_param.affinity = &affinity;
  compute_forw_param.end = CNRT_PARAM_END;
  CNML_RETURN_STATUS(cnmlComputePluginCropFeatureAndResizeOpForward(op,
        in_addr, out_addr, compute_forw_param, queue));
}

// void CreateCustomizedActiveOpParam(MLUCustomizedActiveOpParam **param,
//      float x_start, float x_end, float y_min, int segment_num) {
//    MLU_EXIT_IF_ERROR(cnmlCreateCustomizedActiveOpParam(param,
//            x_start, x_end, y_min, segment_num));
//}
// void DestroyCustomizedActiveOpParam(MLUCustomizedActiveOpParam **param) {
//    MLU_EXIT_IF_ERROR(cnmlDestroyCustomizedActiveOpParam(param));
//}

tensorflow::Status CreateCustomizedActiveOp(MLUBaseOp** op,
                                            void* active_func_ptr,
                                            MLUTensor* input,
                                            MLUTensor* output) {
  float x_start = -25;
  float x_end = 25;
  float y_min = 0;
  int segment_num = 120;
  MLUCustomizedActiveOpParam* param;
  TF_CNML_CHECK(cnmlCreateCustomizedActiveOpParam(&param, x_start, x_end, y_min,
                                                  segment_num));

  TF_CNML_CHECK(
      cnmlCreateCustomizedActiveOp(op, active_func_ptr, param, input, output));

  TF_CNML_CHECK(cnmlDestroyCustomizedActiveOpParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeCustomizedActiveOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                             void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeCustomizedActiveForward_V4(
      op, nullptr, input, nullptr, output, queue, nullptr));
}


tensorflow::Status CreateCycleAddOp(MLUBaseOp** op, int dim, MLUTensor* input1,
                                    MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdCycleAddOp(op, dim, input1, input2, output));
}

tensorflow::Status ComputeCycleAddOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input1, void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleAddOpForward(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}


tensorflow::Status CreateCycleAndOp(MLUBaseOp** op, MLUTensor* input1,
                                    MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdCycleAndOp(op,
        MLUTensorUtil::GetTensorDims(input1) - 1, input1, input2, output));
}

tensorflow::Status ComputeCycleAndOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input1, void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleAndOpForward(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCycleEqualOp(MLUBaseOp** op, MLUTensor* input1,
                                    MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdCycleEqualOp(op,
        MLUTensorUtil::GetTensorDims(input1) - 1, input1, input2, output));
}

tensorflow::Status ComputeCycleEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input1, void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleEqualOpForward(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCycleGreaterOp(MLUBaseOp** op, MLUTensor* in0,
                                        MLUTensor* in1, MLUTensor* output) {
  // CNML_RETURN_STATUS(cnmlCreateNdCycleGreaterOp(op,
  //       MLUTensorUtil::GetTensorDims(in0) - 1, in0, in1, output));
  CNML_RETURN_STATUS(cnmlCreateCycleGreaterOp(op, in0, in1, output));
}

tensorflow::Status ComputeCycleGreaterOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input0, void* input1,
                                         void* output) {
  CNML_RETURN_STATUS(cnmlComputeCycleGreaterOpForward_V4(
      op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCycleGreaterEqualOp(MLUBaseOp** op, MLUTensor* in0,
                                        MLUTensor* in1, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdCycleGreaterEqualOp(op,
        MLUTensorUtil::GetTensorDims(in0) - 1, in0, in1, output));
}

tensorflow::Status ComputeCycleGreaterEqualOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* input0, void* input1, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleGreaterEqualOpForward(
      op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCycleLessOp(MLUBaseOp** op, MLUTensor* in0,
                                     MLUTensor* in1, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdCycleLessOp(op,
        MLUTensorUtil::GetTensorDims(in0) - 1, in0, in1, output));
}

tensorflow::Status ComputeCycleLessOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input0, void* input1,
                                      void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleLessOpForward(
      op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCycleLessEqualOp(MLUBaseOp** op, MLUTensor* in0,
                                     MLUTensor* in1, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdCycleLessEqualOp(op,
        MLUTensorUtil::GetTensorDims(in0) - 1, in0, in1, output));
}

tensorflow::Status ComputeCycleLessEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input0, void* input1,
                                      void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleLessEqualOpForward(
      op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCycleMulOp(MLUBaseOp** op, MLUTensor* in0,
                                    MLUTensor* in1, MLUTensor* output) {
  MLUTensorUtil in0_tensor_util(in0);
  int in0_dims = in0_tensor_util.dims() - 1;
  CNML_RETURN_STATUS(cnmlCreateNdCycleMultOp(op, in0_dims, in0, in1, output));
}

tensorflow::Status ComputeCycleMulOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input0, void* input1, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleMultOpForward(
      op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCycleOrOp(MLUBaseOp** op, MLUTensor* in0,
                                   MLUTensor* in1, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdCycleOrOp(op,
        MLUTensorUtil::GetTensorDims(in0) - 1, in0, in1, output));
}

tensorflow::Status ComputeCycleOrOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input0, void* input1, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleOrOpForward(
        op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCycleSubOp(MLUBaseOp** op, MLUTensor* in0,
                                    MLUTensor* in1, MLUTensor* output) {
  MLUTensorUtil in0_tensor_util(in0);
  int in0_dims = in0_tensor_util.dims() - 1;
  CNML_RETURN_STATUS(cnmlCreateNdCycleSubOp(op, in0_dims, in0, in1, output));
}

tensorflow::Status ComputeCycleSubOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input0, void* input1, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdCycleSubOpForward(
      op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateDeConvOp(MLUBaseOp** op, MLUTensor* in,
                                  MLUTensor* output, MLUTensor* filter,
                                  MLUTensor* bias, int stride_height,
                                  int stride_width, int hu, int hd,
                                  int wl, int wr) {
  MLUDeconvOpParam* param;
  TF_CNML_CHECK(cnmlCreateDeconvOpParam(&param, stride_height, stride_width, hu,
                                        hd, wl, wr));
  TF_CNML_CHECK(
      cnmlCreateDeconvOp(op, param, in, output, filter, bias));
  TF_CNML_CHECK(cnmlDestroyDeconvOpParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeDeConvOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeDeconvOpForward_V4(op, nullptr, input, nullptr,
                                              output, queue, nullptr));
}

tensorflow::Status CreateSnapshotOp(MLUBaseOp** op, MLUTensor* in,
                                        MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateDeviceMemcpyOp(op, in, output));
}

tensorflow::Status ComputeSnapshotOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeDeviceMemcpyOpForward_V4(op,
        nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateConv2DOpParam(MLUConvOpParam** op_param,
    int stride_height, int stride_width, int dilation_height,
    int dilation_width, int pad_height, int pad_width){
  CNML_RETURN_STATUS(cnmlCreateConvOpParam(op_param, stride_height, stride_width,
            dilation_height, dilation_width, pad_height, pad_width));
}

tensorflow::Status CreateQuantConv2DOp(MLUBaseOp** op,
        MLUConvOpParam* param, MLUTensor* input,
        MLUTensor* input_param, MLUTensor* filter,
        MLUTensor* filter_param, MLUTensor* bias,
        MLUTensor* output){
  CNML_RETURN_STATUS(cnmlCreateConvOpTrainingForward(op, param, input,
               input_param, filter, filter_param, bias?bias:nullptr, output));
}

tensorflow::Status ComputeQuantConv2DOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* input, void* input_param,
    void* filter, void* filter_param, void* bias, void* output){
    CNML_RETURN_STATUS(cnmlComputeConvOpTrainingForward(op,
                nullptr, input, nullptr, input_param, nullptr,
                filter, nullptr, filter_param, nullptr, bias?bias:nullptr,
                nullptr, output, queue, nullptr));
}

tensorflow::Status CreateRoundOp(MLUBaseOp** op, MLUTensor* in,
                                 MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateCastOp(op,
        CNML_CAST_FLOAT16_TO_FLOAT16_ROUND_EVEN, in, output));
}

tensorflow::Status ComputeRoundOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeCastOpForward_V4(op,
        nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateConvDepthwiseOp(MLUBaseOp** op, MLUTensor* in,
                                         MLUTensor* filter, MLUTensor* bias,
                                         MLUTensor* output, int stride_height,
                                         int stride_width) {
  MLUConvDepthwiseOpParam* param;
  TF_CNML_CHECK(
      cnmlCreateConvDepthwiseOpParam(&param, stride_height, stride_width));
  TF_CNML_CHECK(cnmlCreateConvDepthwiseOp(op, param, in, output, filter, bias));
  TF_CNML_CHECK(cnmlDestroyConvDepthwiseOpParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeDepthwiseConvOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                          void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeConvDepthwiseOpForward_V4(
      op, nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateEluOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateEluOp(op, in, output));
}

tensorflow::Status ComputeEluOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output) {
  CNML_RETURN_STATUS(cnmlComputeEluOpForward_V4(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateEqualOp(MLUBaseOp** op, MLUTensor* in0, MLUTensor* in1,
                                 MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateEqualOp(op, in0, in1, output));
}

tensorflow::Status ComputeEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input0, void* input1, void* output) {
  CNML_RETURN_STATUS(cnmlComputeEqualOpForward_V4(
      op, nullptr, input0, nullptr, input1, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateErfOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateErfOp(op, in, output));
}

tensorflow::Status ComputeErfOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output) {
  CNML_RETURN_STATUS(cnmlComputeErfOpForward_V4(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateExpOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateExpOp(op, in, output));
}

tensorflow::Status ComputeExpOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output) {
  CNML_RETURN_STATUS(cnmlComputeExpOpForward_V4(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateFloorOp(MLUBaseOp** op, MLUTensor* input,
                                 MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateFloorOp(op, input, output));
}

tensorflow::Status ComputeFloorOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeFloorOpForward_V4(op, nullptr, input, nullptr,
                                             output, queue, nullptr));
}

tensorflow::Status CreateGatherOp(MLUBaseOp** op, MLUTensor* input1,
                                  MLUTensor* input2, MLUTensor* output,
                                  MLUDimension_t gather_mode) {
  CNML_RETURN_STATUS(cnmlCreateGatherV2Op(op, input1,
                                          input2, output, gather_mode));
}

tensorflow::Status ComputeGatherOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input1, void* input2,
                                   void* output) {
  CNML_RETURN_STATUS(cnmlComputeGatherV2OpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateGreaterEqualOp(MLUBaseOp** op, MLUTensor* input1,
                                        MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateGreaterEqualOp(op, input1, input2, output));
}

tensorflow::Status ComputeGreaterEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input1, void* input2,
                                         void* output) {
  CNML_RETURN_STATUS(cnmlComputeGreaterEqualOpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateGreaterOp(MLUBaseOp** op, MLUTensor* input1,
                                   MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateGreaterOp(op, input1, input2, output));
}

tensorflow::Status ComputeGreaterOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input1, void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeGreaterOpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateInterpOp(MLUBaseOp** op, MLUTensor* input,
                                  MLUTensor* output, int output_height,
                                  int output_width, bool align_corners) {
  MLUInterpOpParam* params;
  TF_CNML_CHECK(cnmlCreateInterpOpParam(&params, output_width, output_height,
                                        align_corners));
  TF_CNML_CHECK(cnmlCreateInterpOp(op, input, output, params));
  TF_CNML_CHECK(cnmlDestroyInterpOpParam(&params));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeInterpOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeInterpOpForward_V4(op, nullptr, input, nullptr,
                                              output, queue, nullptr));
}

tensorflow::Status CreateInvertPermutationOp(MLUBaseOp** op,
    MLUTensor* input, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateInvertPermutationOp(op, input, output));
}

tensorflow::Status ComputeInvertPermutationOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeInvertPermutationOpForward(op,
        nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateIsFiniteOp(MLUBaseOp** op, MLUTensor* input,
                                  MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateIsFiniteOp(op, input, output));
}
tensorflow::Status ComputeIsFiniteOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeIsFiniteOpForward(op,
        nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateLessEqualOp(MLUBaseOp** op, MLUTensor* input1,
                                     MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateLessEqualOp(op, input1, input2, output));
}

tensorflow::Status ComputeLessEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input1, void* input2,
                                      void* output) {
  CNML_RETURN_STATUS(cnmlComputeLessEqualOpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateLessOp(MLUBaseOp** op, MLUTensor* input1,
                                MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateLessOp(op, input1, input2, output));
}

tensorflow::Status ComputeLessOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input1, void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeLessOpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateMlpOp(MLUBaseOp** op, MLUTensor* input,
    MLUTensor* output, MLUTensor* filter, MLUTensor* bias) {
  CNML_RETURN_STATUS(cnmlCreateMlpOp(op, input, output, filter, bias));
}

tensorflow::Status ComputeMlpOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeMlpOpForward_V4(op,
        nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateNearestNeighborOp(MLUBaseOp** op,
    MLUTensor* input, MLUTensor* output,
    int output_height, int output_width, bool align_corners) {
  MLUNearestNeighborOpParam *param = nullptr;
  TF_CNML_CHECK(cnmlCreateNearestNeighborOpParam(&param,
        output_width, output_height));
  TF_CNML_CHECK(cnmlSetNearestNeighborAlignCorner(&param, align_corners));
  TF_CNML_CHECK(cnmlCreateNearestNeighborOp(op, input, output, param));
  TF_CNML_CHECK(cnmlDestroyNearestNeighborOpParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeNearestNeighborOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNearestNeighborOpForward_V4(op,
        nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateLrnOp(MLUBaseOp** op, MLUTensor* input,
    MLUTensor* output, MLULrnType lrn_type,
    int local_size, double alph, double beta, double k) {
  cnmlLrnOpParam_t cnml_param = nullptr;
  TF_CNML_CHECK(cnmlCreateLrnOpParam(&cnml_param,
    lrn_type, local_size, alph, beta, k));
  TF_CNML_CHECK(cnmlCreateLrnOp(op, cnml_param, input, output));
  TF_CNML_CHECK(cnmlDestroyLrnOpParam(&cnml_param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeLrnOp(MLUBaseOp* op, MLUCnrtQueue* queue,
  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeLrnOpForward_V4(
      op, nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateNegOp(MLUBaseOp** op, int dim, MLUTensor* input,
                                 MLUTensor* alpha, MLUTensor* beta,
                                 MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdScaleOp(op, dim, input, output, alpha, beta));
}

tensorflow::Status ComputeNegOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdScaleOpForward(op, nullptr, input,
                                            nullptr, output, queue, nullptr));
}

tensorflow::Status CreateOneHotOp(MLUBaseOp** op, MLUTensor* indices,
                                 MLUTensor* output, int *shape, int depth,
                                 float on_value, float off_value, int axis) {
  cnmlPluginOneHotOpParam_t param;
  cnmlCoreVersion_t core_version = CNML_MLU270;
  TF_CNML_CHECK(cnmlCreatePluginOneHotOpParam(
        &param, core_version, shape[0], shape[1], shape[2], shape[3],
		    depth, on_value, off_value, axis));

  cnmlTensor_t cnml_inputs_ptr[1];
  cnmlTensor_t cnml_outputs_ptr[1];
  cnml_inputs_ptr[0] = indices;
  cnml_outputs_ptr[0] = output;
  CNML_RETURN_STATUS(cnmlCreatePluginOneHotOp(op, param, cnml_inputs_ptr, cnml_outputs_ptr));
  TF_CNML_CHECK(cnmlDestroyPluginOneHotOpParam(&param));
}

tensorflow::Status ComputeOneHotOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input[], int in_size, void* output[], int out_size) {
  CNML_RETURN_STATUS(cnmlComputePluginOneHotOpForward(op, input, in_size, output, out_size, queue));
}

tensorflow::Status CreateOnesLikeOp(MLUBaseOp** op, int dim, MLUTensor* input,
                                 MLUTensor* alpha, MLUTensor* beta,
                                 MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdScaleOp(op, dim, input, output, alpha, beta));
}

tensorflow::Status ComputeOnesLikeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdScaleOpForward(op, nullptr, input,
                                            nullptr, output, queue, nullptr));
}

tensorflow::Status CreatePad4Op(MLUBaseOp** op, MLUTensor* input,
    MLUTensor* output, int padding_htop, int padding_hbottom,
    int padding_wleft, int padding_wright, float pad_value) {
  cnmlAddPadOpParam_t  cnml_para = nullptr;
  TF_CNML_CHECK(cnmlCreateAddPadOpParam_V2(&cnml_para, padding_htop,
          padding_hbottom, padding_wleft, padding_wright, pad_value));

  TF_CNML_CHECK(cnmlCreateAddPadOp(op, cnml_para, input, output));
  TF_CNML_CHECK(cnmlDestroyAddPadOpParam(&cnml_para));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputePad4Op(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeAddPadOpForward_V4(
      op, nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreatePoolOp(
    MLUBaseOp** op, MLUTensor* input, MLUTensor* output, bool real,
    const std::vector<int>& kernel_size, const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::vector<std::pair<int, int>>& paddings,
    MLUPoolMode pool_mode, MLUPoolStrategyMode pool_strategy_mode) {
  // prepare pool param
  int paddings_array[paddings.size()][2];
  for (int i = 0; i < paddings.size(); ++i) {
    paddings_array[i][0] = paddings[i].first;
    paddings_array[i][1] = paddings[i].second;
  }
  MLUPoolOpParam* pool_param;
  TF_CNML_CHECK(cnmlCreateNdPoolOpParam(
      &pool_param, pool_mode, pool_strategy_mode,
      real, kernel_size.size(),
      (const_cast<std::vector<int>&>(kernel_size)).data(),
      (const_cast<std::vector<int>&>(dilations)).data(),
      (const_cast<std::vector<int>&>(strides)).data(),
      paddings_array));
  TF_CNML_CHECK(cnmlCreateNdPoolOp(op, pool_param, input, output));
  TF_CNML_CHECK(cnmlDestroyNdPoolOpParam(&pool_param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputePoolOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdPoolOpForward_V2(op, nullptr, input, nullptr,
                                              output, queue, nullptr));
}

tensorflow::Status CreatePowOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output, float c) {
  CNML_RETURN_STATUS(cnmlCreatePowerOp(op, input, output, c));
}

tensorflow::Status ComputePowOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output) {
  CNML_RETURN_STATUS(cnmlComputePowerOpForward_V4(op, nullptr, input, nullptr,
                                             output, queue, nullptr));
}

tensorflow::Status CreateRealDivOp(MLUBaseOp** op, MLUTensor* input1,
                                   MLUTensor* input2, MLUTensor* output,
                                   bool high_precision_flag) {
  TF_CNML_CHECK(cnmlCreateRealDivOp(op, input1, input2, output));
  if(high_precision_flag)
    TF_CNML_CHECK(cnmlSetRealDivHighPrecision(op, high_precision_flag));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeRealDivOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input0, void* input1, void* output0) {
  CNML_RETURN_STATUS(cnmlComputeRealDivOpForward_V4(
      op, nullptr, input0, nullptr, input1, nullptr, output0, queue, nullptr));
}

tensorflow::Status CreateReduceAllOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                MLUTensor* output) {
  int input_dims = MLUTensorUtil::GetTensorDims(input);
  TF_PARAMS_CHECK(input_dims >= 1 && input_dims <= 4,
      "Input dims must be within [1, 4], now ", input_dims);
  TF_PARAMS_CHECK(input_dims > axis, "The axis must be less than input dims");
  cnmlReduce_andDim_t reduce_axis;
  if (axis == input_dims - 1) {
    reduce_axis = CNML_REDUCE_AND_DIM_C;
  } else if (axis == 0) {
    reduce_axis = CNML_REDUCE_AND_DIM_N;
  } else if (axis == 1) {
    reduce_axis = CNML_REDUCE_AND_DIM_H;
  } else {
    reduce_axis = CNML_REDUCE_AND_DIM_W;
  }
  CNML_RETURN_STATUS(cnmlCreateReduceAndOp(op, reduce_axis, input, output));
}

tensorflow::Status ComputeReduceAllOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeReduceAndOpForward(op, nullptr,
        input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateReduceAnyOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                     MLUTensor* output) {
  int input_dims = MLUTensorUtil::GetTensorDims(input);
  TF_PARAMS_CHECK(input_dims >= 1 && input_dims <= 4,
      "Input dims must be within [1, 4], now ", input_dims);
  TF_PARAMS_CHECK(input_dims > axis, "The axis must be less than input dims");
  cnmlReduce_orDim_t reduce_axis;
  if (axis == input_dims - 1) {
    reduce_axis = CNML_REDUCE_OR_DIM_C;
  } else if (axis == 0) {
    reduce_axis = CNML_REDUCE_OR_DIM_N;
  } else if (axis == 1) {
    reduce_axis = CNML_REDUCE_OR_DIM_H;
  } else {
    reduce_axis = CNML_REDUCE_OR_DIM_W;
  }
  CNML_RETURN_STATUS(cnmlCreateReduceOrOp(op, reduce_axis, input, output));
}

tensorflow::Status ComputeReduceAnyOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeReduceOrOpForward(op, nullptr,
        input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateReduceMaxOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                     MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdReduceMaxOp(op, axis, input, output));
}

tensorflow::Status ComputeReduceMaxOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdReduceMaxOpForward_V2(op, nullptr,
        input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateReduceMeanOp(MLUBaseOp** op, int axis,
                                      MLUTensor* input, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdReduceMeanOp(op, axis, input, output));
}

tensorflow::Status ComputeReduceMeanOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                       void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdReduceMeanOpForward_V2(op, nullptr,
        input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateReduceSumOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                     MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdReduceSumOp(op, axis, input, output));
}

tensorflow::Status ComputeReduceSumOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdReduceSumOpForward_V2(op, nullptr,
        input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateReshapeOp(MLUBaseOp** op, MLUTensor* input,
                                   MLUTensor* output) {
  MLUTensorUtil out_tensor_util(output);
  int output_dims = out_tensor_util.dims();
  int* output_shape = out_tensor_util.dim_sizes_array();

  MLUReshapeOpParam* reshape_param;
  TF_CNML_CHECK(
      cnmlCreateNdReshapeOpParam(&reshape_param, output_shape, output_dims));
  TF_CNML_CHECK(cnmlCreateReshapeOp(op, reshape_param, input, output));

  TF_CNML_CHECK(cnmlDestroyReshapeOpParam(&reshape_param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeReshapeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeReshapeOpForward_V4(op, nullptr, input, nullptr,
                                               output, queue, nullptr));
}

tensorflow::Status CreateReverseOp(MLUBaseOp** op, MLUTensor* input,
                                   MLUTensor* output, int axis) {
  int input_dims = MLUTensorUtil::GetTensorDims(input);
  if (input_dims > 4 || input_dims < 2) {
    return tensorflow::errors::InvalidArgument(
      "The dimension size of input must be in [2, 4], now ", input_dims);
  }
  if (input_dims <= axis) {
    return tensorflow::errors::InvalidArgument(
      "The axis must be less than input dims");
  }
  MLUDimension_t reverse_axis;
  if (axis == 0) {
    reverse_axis = MLU_DIM_N;
  } else if (axis == input_dims - 1) {
    reverse_axis = MLU_DIM_C;
  } else if (axis == 1) {
    reverse_axis = MLU_DIM_H;
  } else {
    reverse_axis = MLU_DIM_W;
  }
  CNML_RETURN_STATUS(cnmlCreateReverseOp(op, input, output, reverse_axis));
}

tensorflow::Status ComputeReverseOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeReverseOpForward_V4(op, nullptr, input, nullptr,
                                               output, queue, nullptr));
}

tensorflow::Status CreateRsqrtOp(MLUBaseOp** op, MLUTensor* input,
                                 MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateRsqrtOp(op, input, output));
}

tensorflow::Status ComputeRsqrtOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeRsqrtOpForward_V4(op, nullptr, input, nullptr,
                                             output, queue, nullptr));
}

tensorflow::Status CreateScaleOp(MLUBaseOp** op, int dim, MLUTensor* input,
                                 MLUTensor* alpha, MLUTensor* beta,
                                 MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdScaleOp(op, dim, input, output, alpha, beta));
}

tensorflow::Status ComputeScaleOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdScaleOpForward(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSelectOp(MLUBaseOp** op, MLUTensor* input0,
                                  MLUTensor* input1, MLUTensor* input2,
                                  MLUTensor* output, bool bool_index,
                                  bool batch_index) {
  TF_CNML_CHECK(cnmlCreateDyadicSelectOp(op, input0, input1, input2, output));
  TF_CNML_CHECK(cnmlDyadicSelectOpSetParam(*op, bool_index, batch_index));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeSelectOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input0, void* input1, void* input2,
                                   void* output) {
  CNML_RETURN_STATUS(cnmlComputeDyadicSelectOpForward_V4(
      op, nullptr, input0, nullptr, input1, nullptr, input2, nullptr, output,
      queue, nullptr));
}

tensorflow::Status CreateSeluOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateSeluOp(op, in, output));
}

tensorflow::Status ComputeSeluOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeSeluOpForward_V4(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSoftmaxOp(MLUBaseOp** op, int dim,
                                   MLUTensor* input, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdSoftmaxOp(op, dim, input, output));
}

tensorflow::Status ComputeSoftmaxOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdSoftmaxOpForward_V2(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSoftsignOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateSoftsignOp(op, in, output));
}

tensorflow::Status ComputeSoftsignOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeSoftsignOpForward_V4(op, nullptr,
        input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSpace2BatchOp(MLUBaseOp** op, int w_block_size,
                                       int h_block_size, MLUTensor* input,
                                       MLUTensor* output) {
  CNML_RETURN_STATUS(
      cnmlCreateSpace2batchOp(op, w_block_size, h_block_size, input, output));
}

tensorflow::Status ComputeSpace2BatchOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeSpace2batchOpForward_V4(op, nullptr,
        input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSplitOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                 MLUTensor* outputs[], int output_num) {
  CNML_RETURN_STATUS(cnmlCreateNdSplitOp(op, axis, &input, 1,
        outputs, output_num));
}

tensorflow::Status ComputeSplitOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* outputs[],
                                  int output_num) {
  CNML_RETURN_STATUS(cnmlComputeNdSplitOpForward_V2(
      op, nullptr, &input, 1, nullptr, outputs, output_num, queue, nullptr));
}

tensorflow::Status CreateSqrtOp(MLUBaseOp** op, MLUTensor* input,
                                MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateSqrtOp(op, input, output));
}

tensorflow::Status ComputeSqrtOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeSqrtOpForward_V4(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSquareOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateSquareOp(op, input, output));
}

tensorflow::Status ComputeSquareOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeSquareOpForward_V2(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSquaredDifferenceOp(MLUBaseOp** op, MLUTensor* input1,
                                             MLUTensor* input2,
                                             MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateSquaredDiffOp(op, input1, input2, output));
}

tensorflow::Status ComputeSquaredDifferenceOp(MLUBaseOp* op,
                                              MLUCnrtQueue* queue, void* input1,
                                              void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeSquaredDiffOpForward_V4(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateNdStridedSliceOp(MLUBaseOp** op, MLUTensor* input,
                                          MLUTensor* output, int dim_num,
                                          int begin[], int end[],
                                          int stride[]) {
  MLUStridedSliceOpParam* param;
  TF_CNML_CHECK(
      cnmlCreateNdStridedSliceOpParam(&param, dim_num, begin, end, stride));

  TF_CNML_CHECK(cnmlCreateNdStridedSliceOp(op, param, input, output));
  TF_CNML_CHECK(cnmlDestroyNdStridedSliceOpParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeNdStridedSliceOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                           void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdStridedSliceOpForward_V2(
      op, nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSubOp(MLUBaseOp** op, MLUTensor* input1,
                               MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateSubOp(op, input1, input2, output));
}

tensorflow::Status ComputeSubOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input1, void* input2, void* output) {
  CNML_RETURN_STATUS(cnmlComputeSubOpForward_V4(op, nullptr, input1,
        nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateTileOp(MLUBaseOp** op, MLUTensor* input,
                                MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateTileOp(op, input, output));
}

tensorflow::Status ComputeTileOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeTileOpForward(op, nullptr, input,
        nullptr, output, queue, nullptr));
}

tensorflow::Status CreateTopKOp(MLUBaseOp** op, int k, bool sorted,
                                MLUTensor* input, MLUTensor* values_out,
                                MLUTensor* indices_out) {
  CNML_RETURN_STATUS(cnmlCreateTopkOp_V2(
      op, k, input, values_out, indices_out, CNML_DIM_C,
      (sorted ? CNML_TOPK_OP_MODE_MAX : CNML_TOPK_OP_MODE_MIN)));
}

tensorflow::Status ComputeTopKOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output, void* index) {
  CNML_RETURN_STATUS(cnmlComputeTopkOpForward_V4(op, nullptr, input,
        nullptr, output, nullptr, index, queue, nullptr));
}

tensorflow::Status CreateTransposeProOp(MLUBaseOp** op, MLUTensor* input,
                                        MLUTensor* output, int dim_order[],
                                        int dim_num) {
  MLUTransposeOpParam* param;
  TF_CNML_CHECK(cnmlCreateNdTransposeOpParam(&param, dim_order, dim_num));
  TF_CNML_CHECK(
      cnmlCreateNdTransposeProOp(op, input, output, param));
  TF_CNML_CHECK(cnmlDestroyNdTransposeOpParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeTransposeProOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdTransposeProOpForward_V2(
      op, nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateBiasAddGradOp(MLUBaseOp** op, MLUTensor* input,
                                       MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateBiasAddOpBackwardBias(op, input, output));
}

tensorflow::Status ComputeBiasAddGradOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeBiasAddOpBackwardBias(
      op, nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateCosOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateCosOp(op, input, output));
}

tensorflow::Status ComputeCosOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output) {
  CNML_RETURN_STATUS(cnmlComputeCosOpForward_V4(op, nullptr, input, nullptr,
                                                output, queue, nullptr));
}

tensorflow::Status CreateConvFilterGradOp(MLUBaseOp** op,
    MLUTensor* x, MLUTensor* dy, MLUTensor* x_quant, MLUTensor* dy_quant,
    MLUTensor* dw, int kernel_height, int kernel_width, int stride_height,
    int stride_width, int dilation_height, int dilation_width, int pad_top,
    int pad_bottom, int pad_left, int pad_right) {
  MLUConvOpBackwardParam* param;
  TF_CNML_CHECK(cnmlCreateConvOpBackwardParam(&param, kernel_height,
        kernel_width, stride_height, stride_width, dilation_height,
        dilation_width, pad_top, pad_bottom, pad_left, pad_right));
  TF_CNML_CHECK(cnmlCreateConvOpBackwardFilter(op, param,
        x, dy, x_quant, dy_quant, dw));
  TF_CNML_CHECK(cnmlDestroyConvOpBackwardParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeConvFilterGradOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* x, void* dy, void* x_quant, void* dy_quant, void* dw) {
  CNML_RETURN_STATUS(cnmlComputeConvOpBackwardFilter(op, nullptr, x,
        nullptr, dy, nullptr, x_quant, nullptr, dy_quant, nullptr, dw,
        queue, nullptr));
}

tensorflow::Status CreateFloorDivOp(MLUBaseOp** op, MLUTensor* input1,
                                    MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateFloorDivOp(op, input1, input2, output));
}

tensorflow::Status ComputeFloorDivOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input1, void* input2,
                                     void* output) {
  CNML_RETURN_STATUS(cnmlComputeFloorDivOpForward(op, nullptr, input1, nullptr,
      input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateFusedBatchNormGradOp(
    MLUBaseOp** op, MLUTensor* x, MLUTensor* dy, MLUTensor* mean,
    MLUTensor* variance, MLUTensor* scale, MLUTensor* dx, MLUTensor* d_gamma,
    MLUTensor* d_beta, float epsilon) {
  CNML_RETURN_STATUS(cnmlCreateFusedBatchNormOpBackward(
      op, x, nullptr, dy, mean, variance, scale, dx, d_gamma, d_beta, epsilon));
}

tensorflow::Status ComputeFusedBatchNormGradOp(MLUBaseOp* op,
                                               MLUCnrtQueue* queue, void* x,
                                               void* y, void* dz, void* mean,
                                               void* variance, void* gamma,
                                               void* dx, void* d_gamma,
                                               void* d_beta) {
  CNML_RETURN_STATUS(cnmlComputeFusedBatchNormOpBackward(
      op, nullptr, x, nullptr, y, nullptr, dz, nullptr, mean, nullptr, variance,
      nullptr, gamma, nullptr, dx, nullptr, d_gamma, nullptr, d_beta, queue,
      nullptr));
}

tensorflow::Status CreateFusedBatchNormOp(MLUBaseOp** op, MLUTensor* input,
                                          MLUTensor* es_mean, MLUTensor* es_var,
                                          MLUTensor* gamma, MLUTensor* beta,
                                          MLUTensor* eps, MLUTensor* output,
                                          MLUTensor* batch_mean,
                                          MLUTensor* batch_var, MLUTensor* mean,
                                          MLUTensor* var) {
  CNML_RETURN_STATUS(cnmlCreateFusedBatchNormOp_V2(
      op, input, es_mean, es_var, gamma, beta, eps,
      output, batch_mean, batch_var, mean, var));
}

tensorflow::Status ComputeFusedBatchNormOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                           void* input, void* es_mean,
                                           void* es_var, void* gamma,
                                           void* beta, void* output,
                                           void* batch_mean, void* batch_var,
                                           void* mean, void* var) {
  CNML_RETURN_STATUS(cnmlComputeFusedBatchNormOpForward_V2(
      op, nullptr, input, nullptr, es_mean, nullptr, es_var, nullptr, gamma,
      nullptr, beta, nullptr, output, nullptr, batch_mean, nullptr, batch_var,
      nullptr, mean, nullptr, var, queue, nullptr));
}

tensorflow::Status CreateL2LossOp(MLUBaseOp** op, MLUTensor* x,
                                  MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateMseOp(op, x, output, nullptr, nullptr));
}

tensorflow::Status ComputeL2LossOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input_label, void* input_predicted,
                                   void* output) {
  CNML_RETURN_STATUS(cnmlComputeMseOpForward(op, nullptr, input_label, nullptr,
                                             input_predicted, nullptr, output,
                                             queue, nullptr));
}

tensorflow::Status CreateListDiffOp(MLUBaseOp** op, MLUTensor* x, MLUTensor* y,
                                    MLUTensor* output_data,
                                    MLUTensor* output_index) {
  CNML_RETURN_STATUS(cnmlCreateListDiffOp(op, x, y, output_data, output_index));
}

tensorflow::Status ComputeListDiffOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* inputX, void* inputY, void* out,
                                     void* out_idx) {
  CNML_RETURN_STATUS(cnmlComputeListDiffOpForward(op, nullptr, inputX, nullptr,
                                                  inputY, nullptr, out, nullptr,
                                                  out_idx, queue, nullptr));
}

tensorflow::Status CreateLogOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output) {
   CNML_RETURN_STATUS(cnmlCreateLogOp(op, input, output));
}

tensorflow::Status ComputeLogOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeLogOpForward_V4(op, nullptr, input, nullptr,
                                                output, queue, nullptr));
}

tensorflow::Status CreateLogicalNotOp(MLUBaseOp** op, MLUTensor* input,
                                      MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNotOp(op, input, output));
}

tensorflow::Status ComputeLogicalNotOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                       void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNotOpForward_V4(op, nullptr, input, nullptr,
                                                output, queue, nullptr));
}

tensorflow::Status CreateLogicalOrOp(MLUBaseOp** op, MLUTensor* input1,
                                     MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateOrOp(op, input1, input2, output));
}

tensorflow::Status ComputeLogicalOrOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input1, void* input2,
                                      void* output) {
  CNML_RETURN_STATUS(cnmlComputeOrOpForward_V4(op, nullptr, input1, nullptr,
                                               input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateReciprocalOp(MLUBaseOp** op, MLUTensor* input,
                                      MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateBasicDivOp(op, input, output));
}

tensorflow::Status ComputeReciprocalOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                       void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeBasicDivOpForward(op, nullptr, input, nullptr,
                                                  output, queue, nullptr));
}

tensorflow::Status CreateMaximumOp(MLUBaseOp** op, MLUTensor* input1,
                                   MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateMaximumOp(op, input1, input2, output));
}

tensorflow::Status ComputeMaximumOpForward(MLUBaseOp* op, MLUCnrtQueue* queue,
                                           void* input1, void* input2,
                                           void* output) {
  CNML_RETURN_STATUS(cnmlComputeMaximumOpForward_V2(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateMinimumOp(MLUBaseOp** op, MLUTensor* input1,
                                   MLUTensor* input2, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateMinimumOp(op, input1, input2, output));
}

tensorflow::Status ComputeMinimumOpForward(MLUBaseOp* op, MLUCnrtQueue* queue,
                                           void* input1, void* input2,
                                           void* output) {
  CNML_RETURN_STATUS(cnmlComputeMinimumOpForward_V2(
      op, nullptr, input1, nullptr, input2, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateMaxPoolIndexOp(MLUBaseOp** op,
                                        MLUTensor* tensor_in,
                                        MLUTensor* output_no_use,
                                        MLUTensor* index,
                                        int window_height, int window_width,
                                        int stride_height, int stride_width,
                                        int padding_rows, int padding_cols,
                                        int dilation_height, int dilation_width,
                                        MLUPoolMode pool_mode,
                                        MLUPoolStrategyMode strategy_mode,
                                        bool real) {
  cnmlPoolOpParam* pool_param_ptr  = nullptr;
  TF_CNML_CHECK(cnmlCreatePoolOpParam(&pool_param_ptr,
                                      window_height, window_width,
                                      stride_height, stride_width,
                                      padding_rows, padding_cols,
                                      dilation_height, dilation_width,
                                      pool_mode, strategy_mode,
                                      true));
  TF_CNML_CHECK(cnmlCreatePoolIndexOp(op, pool_param_ptr, tensor_in,
                                      output_no_use, index));

  CNML_RETURN_STATUS(cnmlDestroyPoolOpParam(&pool_param_ptr));
}

tensorflow::Status ComputeMaxPoolIndexOp(MLUBaseOp* op,
                                         MLUCnrtQueue *queue,
                                         void* input,
                                         void* output_no_use,
                                         void* index) {
  CNML_RETURN_STATUS(cnmlComputePoolIndexOpForward(op, nullptr, input, nullptr,
                                                   output_no_use, nullptr,
                                                   index, queue, nullptr));
}

tensorflow::Status CreatePoolBackwardOp(MLUBaseOp** op, MLUTensor* out_backprop,
    MLUTensor* index, MLUTensor* output,
    int window_height, int window_width,
    int stride_height, int stride_width,
    int pad_left, int pad_right,
    int pad_up, int pad_down,
    MLUPoolBackwardMode poolbp_mode,
    MLUPoolBackwardStrategyMode padding_mode) {
  MLUPoolOpBackwardParam* poolbp_param_ptr = nullptr;
  // the last parameter is for pytorch, we just set it to false.
  TF_CNML_CHECK(cnmlCreatePoolOpBackwardParam(&poolbp_param_ptr,
                                              window_height,
                                              window_width, stride_height,
                                              stride_width, pad_left,
                                              pad_right, pad_up,
                                              pad_down, poolbp_mode,
                                              padding_mode, false));

  // in avgpoolbp mode, 3th parameter [index] is not used.
  TF_CNML_CHECK(cnmlCreatePoolOpBackward(op, out_backprop, index, output,
                                         poolbp_param_ptr));

  CNML_RETURN_STATUS(cnmlDestroyPoolOpBackwardParam(&poolbp_param_ptr));
}

tensorflow::Status ComputePoolBackwardOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* out_backprop, void* index,
                                         void* output) {
  CNML_RETURN_STATUS(cnmlComputePoolOpBackward(op, nullptr, out_backprop,
                                               nullptr, index, nullptr,
                                               output, queue, nullptr));
}

tensorflow::Status CreateQuantifyOp(MLUBaseOp** op, MLUTensor* input,
                                    MLUTensor* oldQuanParams,
                                    MLUTensor* oldMovPos, MLUTensor* interval,
                                    MLUTensor* output, MLUTensor* quanParams,
                                    MLUTensor* movPos) {
  CNML_RETURN_STATUS(cnmlCreateQuantifyOp(op, input, oldQuanParams, oldMovPos,
                                          output, quanParams, movPos,
                                          interval));
}

tensorflow::Status ComputeQuantifyOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input, void* input_param,
                                     void* input_mp, void* interval,
                                     void* output, void* output_param,
                                     void* output_mp) {
  CNML_RETURN_STATUS(cnmlComputeQuantifyOpForward(
      op, nullptr, input, nullptr, input_param, nullptr, input_mp,
      nullptr, output, nullptr, output_param, nullptr, output_mp,
      nullptr, interval, queue, nullptr));
}

tensorflow::Status CreateQuantMatMulOp(MLUBaseOp** op,
                                       MLUTensor* input,
                                       MLUTensor* filter,
                                       MLUTensor* input_param,
                                       MLUTensor* filter_param,
                                       MLUTensor* bias,
                                       MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateMlpOpTrainingForward(op,
      input, input_param, filter, filter_param, bias ? bias : nullptr, output));
}

tensorflow::Status ComputeQuantMatMulOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* input, void* filter,
                                        void* input_param, void* filter_param,
                                        void* bias, void* output) {
  CNML_RETURN_STATUS(cnmlComputeMlpOpTrainingForward(op,
        nullptr, input,
        nullptr, input_param,
        nullptr, filter,
        nullptr, filter_param,
        nullptr, bias ? bias : nullptr,
        nullptr, output,
        queue, nullptr));
}

tensorflow::Status CreateRandomUniformOp(MLUBaseOp** op,
                                         MLUTensor* output, int seed) {
  MLURandomUniformParam* param;
  TF_CNML_CHECK(cnmlCreateRandomUniformOpParam(&param, CNML_RNG_MT19937, 0., 1.));
  TF_CNML_CHECK(cnmlCreateRandomUniformOp(op, param, output));
  TF_CNML_CHECK(cnmlSetRandomSeed(*op, seed));
  CNML_RETURN_STATUS(cnmlDestroyRandomUniformOpParam(&param));
}

tensorflow::Status ComputeRandomUniformOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                          void* output) {
  CNML_RETURN_STATUS(
      cnmlComputeRandomUniformOpForward(op, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateRangeOp(MLUBaseOp** op,
                                 MLUTensor* start, MLUTensor* limit, MLUTensor* delta,
				 int size, MLUTensor* output,
				 cnmlPluginRangeOpParam_t param) {
  MLUTensor* inputs_ptr[3];
  MLUTensor* outputs_ptr[1];
  inputs_ptr[0] = start;
  inputs_ptr[1] = limit;
  inputs_ptr[2] = delta;
  outputs_ptr[0] = output;
  CNML_RETURN_STATUS(cnmlCreatePluginRangeOp(op,
      param, inputs_ptr, outputs_ptr));
}

tensorflow::Status ComputeRangeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* inputs[], int input_num,
                                  void* outputs[], int output_num) {
  CNML_RETURN_STATUS(cnmlComputePluginRangeOpForward(op,
      inputs, input_num, outputs, output_num, queue));
}

tensorflow::Status CreateReduceProdOp(MLUBaseOp** cnml_reduce_prod_op_ptr_ptr,
                                      int axis, MLUTensor* input,
                                      MLUTensor* output) {
  int input_dims = MLUTensorUtil::GetTensorDims(input);
  TF_PARAMS_CHECK(input_dims >= 1 && input_dims <= 4,
      "Input dims must be within [1, 4], now ", input_dims);
  TF_PARAMS_CHECK(input_dims > axis, "The axis must be less than input dims");
  MLUDimension_t reduce_axis;
  if (axis == input_dims - 1) {
    reduce_axis = MLU_DIM_C;
  } else if (axis == 0) {
    reduce_axis = MLU_DIM_N;
  } else if (axis == 1) {
    reduce_axis = MLU_DIM_H;
  } else {
    reduce_axis = MLU_DIM_W;
  }
  CNML_RETURN_STATUS(cnmlCreateReduceProductOp(cnml_reduce_prod_op_ptr_ptr,
                                               reduce_axis, input, output));
}

tensorflow::Status ComputeReduceProdOp(MLUBaseOp* cnml_reduce_prod_op_ptr,
                                       MLUCnrtQueue* queue, void* input,
                                       void* output) {
  CNML_RETURN_STATUS(cnmlComputeReduceProductOpForward_V2(
      cnml_reduce_prod_op_ptr, nullptr, input, nullptr, output, queue,
      nullptr));
}

tensorflow::Status CreateReluGradOp(MLUBaseOp** op, MLUTensor* dy, MLUTensor* x,
                                    MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateReluOpBackward(op, x, dy, output));
}

tensorflow::Status ComputeReluGradOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* x, void* dy, void* output) {
  CNML_RETURN_STATUS(cnmlComputeReluOpBackward(op, nullptr, x, nullptr, dy,
                                     nullptr, output, queue, nullptr));
}

tensorflow::Status CreateRsqrtOpBackward(MLUBaseOp** op, MLUTensor* x,
                                         MLUTensor* dy, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateRsqrtGradOp(op, x, dy, output));
}

tensorflow::Status ComputeRsqrtOpBackward(MLUBaseOp* op, MLUCnrtQueue* queue,
                                          void* y, void* dy, void* output) {
  CNML_RETURN_STATUS(cnmlComputeRsqrtGradOpForward(op, nullptr, y, nullptr, dy,
      nullptr, output, queue, nullptr));
}

tensorflow::Status CreateScatterNdOpParam(MLUScatterNdOpParam** param,
                                          cnmlDimension_t axis,
                                          int scatter_length) {
  CNML_RETURN_STATUS(cnmlCreateScatterOpParam(param, axis, scatter_length));
}

tensorflow::Status DestroyScatterNdOpParam(MLUScatterNdOpParam** param) {
  CNML_RETURN_STATUS(cnmlDestroyScatterOpParam(param));
}

tensorflow::Status CreateScatterNdOp(MLUBaseOp** op, MLUTensor* input1,
                                     MLUTensor* input2, MLUTensor* output,
                                     MLUScatterNdOpParam* param) {
  CNML_RETURN_STATUS(cnmlCreateScatterOp(op, input2, input1, output, param));
}

tensorflow::Status ComputeScatterNdOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* index, void* output) {
  CNML_RETURN_STATUS(cnmlComputeScatterOpForward(
      op, nullptr, input, nullptr, index, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateSinOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateSinOp(op, input, output));
}

tensorflow::Status ComputeSinOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output) {
  CNML_RETURN_STATUS(cnmlComputeSinOpForward_V4(op, nullptr, input, nullptr,
                                                output, queue, nullptr));
}

tensorflow::Status CreateSoftmaxXentWithLogitsOp(MLUBaseOp** op,
                                                 int dim,
                                                 MLUTensor* input,
                                                 MLUTensor* label,
                                                 MLUTensor* output,
                                                 MLUTensor* back_out) {
  CNML_RETURN_STATUS(
      cnmlCreateNdSoftmaxCeLogitsOp(op, dim, input, label, output, back_out));
}

tensorflow::Status ComputeSoftmaxXentWithLogitsOp(MLUBaseOp* op,
                                               MLUCnrtQueue* queue,
                                               void* input, void* label,
                                               void* output, void* back_out) {
  CNML_RETURN_STATUS(cnmlComputeNdSoftmaxCeLogitsOpForward(
      op, nullptr, input, nullptr, label, nullptr, output,
      nullptr, back_out, queue, nullptr));
}

tensorflow::Status CreateStridedSliceOpBackward(
    MLUBaseOp** op_ptr, MLUTensor* input, MLUTensor* output,
    const std::vector<int>& begin, const std::vector<int>& end,
    const std::vector<int>& strides) {
  MLUStridedSliceBackwardParam* param_ptr;
  TF_CNML_CHECK(cnmlCreateStridedSliceOpBackwardParam(&param_ptr,
        /*nb*/begin[0], /*cb*/begin[3], /*hb*/begin[1], /*wb*/begin[2],
        /*ne*/end[0], /*ce*/end[3], /*he*/end[1], /*we*/end[2],
        /*ns*/strides[0], /*cs*/strides[3],
        /*hs*/strides[1], /*ws*/strides[2]));
  TF_CNML_CHECK(cnmlCreateStridedSliceOpBackward(
      op_ptr, param_ptr, input, output));
  CNML_RETURN_STATUS(cnmlDestroyStridedSliceOpBackwardParam(&param_ptr));
}

tensorflow::Status ComputeStridedSliceGradOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                             void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeStridedSliceOpBackward(
      op, nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateUniqueOp(MLUBaseOp** op, MLUTensor* input,
                                  MLUTensor* output, MLUTensor* idx) {
  CNML_RETURN_STATUS(cnmlCreateUniqueOp(op, input, output, idx));
}

tensorflow::Status ComputeUniqueOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output, void* index) {
  CNML_RETURN_STATUS(cnmlComputeUniqueOpForward(
      op, nullptr, input, nullptr, output, nullptr, index, queue, nullptr));
}

tensorflow::Status CreateUnsortedSegmentSumOp(MLUBaseOp** op,
                                              MLUTensor* data,
                                              MLUTensor* segment_ids,
                                              MLUTensor* output,
                                              int num_segments,
                                              int data_dims) {
  cnmlScatterOpParam_t param;
  cnmlDimension_t axis = (data_dims == 1) ? CNML_DIM_C : CNML_DIM_N;

  TF_CNML_CHECK(cnmlCreateScatterOpParam(&param, axis, num_segments));
  TF_CNML_CHECK(cnmlCreateScatterOp(op, data, segment_ids, output, param));
  TF_CNML_CHECK(cnmlDestroyScatterOpParam(&param));
  return tensorflow::Status::OK();
}

tensorflow::Status ComputeUnsortedSegmentSumOp(MLUBaseOp* op,
                                               MLUCnrtQueue* queue,
                                               void* data, void* segment_ids,
                                               void* output) {
  CNML_RETURN_STATUS(cnmlComputeScatterOpForward(op, nullptr, data, nullptr,
                                                 segment_ids, nullptr, output,
                                                 queue, nullptr));
}

tensorflow::Status CreateZerosLikeOp(MLUBaseOp** op, int dim, MLUTensor* input,
                                 MLUTensor* alpha, MLUTensor* beta,
                                 MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdScaleOp(op, dim, input, output, alpha, beta));
}

tensorflow::Status ComputeZerosLikeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdScaleOpForward(op, nullptr, input,
                                            nullptr, output, queue, nullptr));
}

tensorflow::Status CreateLogSoftmaxOp(MLUBaseOp** op, int dim,
                                  MLUTensor* input, MLUTensor* output) {
  CNML_RETURN_STATUS(cnmlCreateNdLogSoftmaxOp(op, input, output, dim));
}
tensorflow::Status ComputeLogSoftmaxOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output) {
CNML_RETURN_STATUS(cnmlComputeNdLogSoftmaxOpForward_V2(op, nullptr,
                                              input, nullptr, output,
                                              queue, nullptr));
}

tensorflow::Status CreateNonMaxSuppressionOp(MLUBaseOp** op,
    MLUTensor* input_boxes, MLUTensor* input_scores, MLUTensor* output,
    cnmlPluginNonMaxSuppressionOpParam_t param) {
  int input_num = 2;
  int output_num = 1;
  int static_num = 2;
  MLUTensor* inputs_ptr[2];
  MLUTensor* outputs_ptr[1];
  inputs_ptr[0] = input_boxes;
  inputs_ptr[1] = input_scores;
  outputs_ptr[0] = output;
  CNML_RETURN_STATUS(cnmlCreatePluginNonMaxSuppressionOp(
        op, param, inputs_ptr, input_num, outputs_ptr,
        output_num, static_num));
}
tensorflow::Status ComputeNonMaxSuppressionOp(MLUBaseOp* op,
    void* input_boxes, void* input_scores, void* output,
    MLUCnrtQueue* queue) {
  int input_num = 2;
  int output_num = 1;
  void *inputs_ptr[2];
  void *outputs_ptr[1];
  inputs_ptr[0] = input_boxes;
  inputs_ptr[1] = input_scores;
  outputs_ptr[0] = output;
  CNML_RETURN_STATUS(cnmlComputePluginNonMaxSuppressionOpForward(
        op, nullptr, inputs_ptr, input_num, nullptr, outputs_ptr,
        output_num, queue, nullptr));
}

tensorflow::Status CreateMatrixBandPartOp(MLUBaseOp** op,
                                          MLUTensor* input,
                                          MLUTensor* output,
                                          int num_lower,
                                          int num_upper) {
  CNML_RETURN_STATUS(cnmlCreateMatrixBandPartOp(op, input, output,
                                                num_lower, num_upper));
}

tensorflow::Status ComputeMatrixBandPartOp(MLUBaseOp* op,
                                           MLUCnrtQueue* queue,
                                           void* input,
                                           void* output) {
  CNML_RETURN_STATUS(cnmlComputeMatrixBandPartOpForward(
        op, nullptr, input, nullptr, output, queue, nullptr));
}

tensorflow::Status CreateLeakyReluOp(MLUBaseOp** op,
    MLUTensor* features, MLUTensor* alpha, MLUTensor* output,
    int dim) {
  CNML_RETURN_STATUS(cnmlCreateNdPreluOp(op, dim, features,
        output, alpha));
}
tensorflow::Status ComputeLeakyReluOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* features, void* output) {
  CNML_RETURN_STATUS(cnmlComputeNdPreluOpForward(op,
        nullptr, features, nullptr, output, queue, nullptr));
}

// TODO:补齐下面create和compute两个函数
tensorflow::Status CreateYolov3DetectionOutputOp(...) {
  ......
}

tensorflow::Status ComputeYolov3DetectionOutputOp(...) {
  ......

  cnmlComputePluginYolov3DetectionOutputOpForward(
      op, inputs, input_num, outputs, output_num, &compute_forw_param, queue);
}

/*tensorflow::Status CreatePowerDifferenceOp(MLUBaseOp** op, MLUTensor* input1,
                                             MLUTensor* input2,
                                             int input3,
                                             MLUTensor* output, int len) {
  MLUTensor* inputs_ptr[2] = {input1, input2};
  MLUTensor* outputs_ptr[1] = {output};

  CNML_RETURN_STATUS(cnmlCreatePluginPowerDifferenceOp(op, inputs_ptr, input3, outputs_ptr, len));
}

tensorflow::Status ComputePowerDifferenceOp(MLUBaseOp* op,
                                              MLUCnrtQueue* queue, void* input1,
                                              void* input2, void* output) {
  void* inputs_ptr[2] = {input1, input2};
  void* outputs_ptr[1] = {output};
  CNML_RETURN_STATUS(cnmlComputePluginPowerDifferenceOpForward(
                                         op, inputs_ptr, outputs_ptr, queue));
}*/

}  // namespace lib
}  // namespace mlu
}  // namespace stream_executor
