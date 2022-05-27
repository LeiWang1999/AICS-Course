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


#ifndef TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_API_LIB_OPS_MLU_LIB_OPS_H_
#define TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_API_LIB_OPS_MLU_LIB_OPS_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mlu.h"


namespace stream_executor {
namespace mlu {
namespace lib {

/******************************************************/

tensorflow::Status CreateAbsOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output);
tensorflow::Status ComputeAbsOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output);
/******************************************************/

tensorflow::Status CreateActiveOp(MLUBaseOp** op, MLUActiveFunction function,
                                  MLUTensor* input, MLUTensor* output);
tensorflow::Status ComputeActiveOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* inputs, void* outputs);
/******************************************************/

tensorflow::Status CreateAddOp(MLUBaseOp** op, MLUTensor* input1,
                               MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeAddOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input1, void* input2, void* output);
/******************************************************/

tensorflow::Status CreateAndOp(MLUBaseOp** op, MLUTensor* input1,
                               MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeAndOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input1, void* input2, void* output);
/******************************************************/

void GetArgmaxOpOutputDim(MLUDimension_t argmax_axis, int ni, int ci, int hi,
                          int wi, int *no, int *co, int *ho, int *wo);
tensorflow::Status CreateArgmaxOp(MLUBaseOp **op, int argmax_axis,
                                  MLUTensor *input, MLUTensor *output);
tensorflow::Status ComputeArgmaxOp(MLUBaseOp *op, MLUCnrtQueue *queue,
                                   void *inputs, void *outputs);
/******************************************************/

tensorflow::Status CreateBatch2SpaceOp(MLUBaseOp** op, int w_block_size,
                                       int h_block_size, MLUTensor* input,
                                       MLUTensor* output);
tensorflow::Status ComputeBatch2SpaceOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* inputs, void* outputs);
/******************************************************/

tensorflow::Status CreateBatchMatMulOp(MLUBaseOp** op, MLUTensor* in0,
                                       MLUTensor* in1, bool adj_x, bool adj_y,
                                       MLUTensor* output);
tensorflow::Status ComputeBatchMatMulOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* input0, void* input1,
                                        void* output);
//TODO:补齐create和compute函数声明
/******************************************************/
tensorflow::Status CreateBatchMatMulV2Op(......);
tensorflow::Status ComputeBatchMatMulV2Op(.....);
/******************************************************/

tensorflow::Status CreateBatchNormOp(MLUBaseOp** op, MLUTensor* input,
                                     MLUTensor* mean, MLUTensor* var,
                                     MLUTensor* output);
tensorflow::Status CreateNdBatchNormOp(MLUBaseOp** op, MLUTensor* input,
                                     MLUTensor* mean, MLUTensor* var,
                                     MLUTensor* output, int dim);
tensorflow::Status ComputeBatchNormOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* inputs, void* outputs);
/******************************************************/

tensorflow::Status CreateBertSquadOp(MLUBaseOp** op,
                                     MLUTensor** inputs, MLUTensor** outputs,
                                     MLUTensor** static_tensors,
                                     int static_tensors_num,
                                     int batch_num, int seq_len);

tensorflow::Status ComputeBertSquadOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void** inputs, void** outputs);

/******************************************************/

tensorflow::Status CreateBroadcastAddOp(MLUBaseOp** op, MLUTensor* input1,
                                        MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeBroadcastAddOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input1, void* input2,
                                         void* output);
/******************************************************/

tensorflow::Status CreateBroadcastMulOp(MLUBaseOp** op, MLUTensor* input1,
                                        MLUTensor* input2, MLUTensor* output);

tensorflow::Status ComputeBroadcastMulOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input1, void* input2,
                                         void* output);
/******************************************************/

tensorflow::Status CreateBroadcastOp(MLUBaseOp** op, MLUTensor* input,
                                     MLUTensor* output);
tensorflow::Status ComputeBroadcastOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* inputs, void* outputs);
/******************************************************/

tensorflow::Status CreateBroadcastSubOp(MLUBaseOp** op, MLUTensor* input1,
                                        MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeBroadcastSubOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input1, void* input2,
                                         void* output);
/******************************************************/

tensorflow::Status CreateCastOp(MLUBaseOp** op, MLUCastType cast_type,
                                MLUTensor* input, MLUTensor* output);
tensorflow::Status ComputeCastOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output);
/******************************************************/

tensorflow::Status CreateClipOp(MLUBaseOp** op, MLUTensor* input,
                                float lower_bound, float upper_bound,
                                MLUTensor* output);
tensorflow::Status ComputeClipOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output);
/******************************************************/

tensorflow::Status CreateConcatOp(MLUBaseOp** op, int dim,
    MLUTensor* inputs[], int input_num, MLUTensor* output);
tensorflow::Status ComputeConcatOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* inputs[], int input_num, void* output);
/******************************************************/

tensorflow::Status CreateDepthwiseConvOp(MLUBaseOp** op,
    MLUTensor* input,  MLUTensor* output,
    MLUTensor* filter, MLUTensor* bias,
    int stride_height, int stride_width,
    int pad_height, int pad_width);
tensorflow::Status ComputeDepthwise_ConvOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* input, void* output);
/******************************************************/

tensorflow::Status CreateConv2DOp(MLUBaseOp** op,
     MLUTensor* input, MLUTensor* output, MLUTensor* filter,
     MLUTensor* bias, int stride_height, int stride_width,
     int dilation_height, int dilation_width,
     int pad_height, int pad_width);
 tensorflow::Status ComputeConv2DOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output);
 /******************************************************/

tensorflow::Status CreateConvFirstOp(MLUBaseOp** op,
    MLUConvFirstOpParam* param,
    MLUTensor* input, MLUTensor* mean, MLUTensor* output,
    MLUTensor* filter, MLUTensor* bias, MLUTensor* std);

tensorflow::Status ComputeConvFirstOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input, void* output);
/******************************************************/

tensorflow::Status CreateConv2DBackpropInputOp(MLUBaseOp** op,
                                  MLUTensor* w, MLUTensor* dy,
                                  MLUTensor* w_param, MLUTensor* dy_param,
                                  MLUTensor* output,
                                  int sh, int sw, int dh, int dw,
                                  int pad_top, int pad_bottom,
                                  int pad_left, int pad_right);
tensorflow::Status ComputeConv2DBackpropInputOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* w, void* dy,
    void* w_param, void* dy_param, void* output);
/******************************************************/

tensorflow::Status CreateConv2DOpParam(MLUConvOpParam** op_param,
    int stride_height, int stride_width, int dilation_height,
    int dilation_width, int pad_height, int pad_width);

tensorflow::Status CreateQuantConv2DOp(MLUBaseOp** op,
    MLUConvOpParam* param, MLUTensor* input,
    MLUTensor* input_param, MLUTensor* filter,
    MLUTensor* filter_param, MLUTensor* bias,
    MLUTensor* output);

tensorflow::Status ComputeQuantConv2DOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* input, void* input_param,
    void* filter, void* filter_param, void* bias, void* output);

/******************************************************/

tensorflow::Status CreateCropOp(MLUBaseOp** op, MLUTensor* input,
                                 MLUTensor* output, int startIndexOfN,
                                 int startIndexOfC, int startIndexOfH,
                                 int startIndexOfW, float space_number);
tensorflow::Status ComputeCropOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

tensorflow::Status CreateCropAndResizeOp(MLUBaseOp** op,
                                         MLUTensor* input, MLUTensor* boxes,
                                         MLUTensor* box_ind,
                                         int crop_height,
                                         int crop_width,
                                         float extrapolation_value,
                                         MLUTensor* output);
tensorflow::Status ComputeCropAndResizeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                          void* input, void* boxes,
                                          void* box_ind, void* output);
/******************************************************/

tensorflow::Status CreateCustomizedActiveOp(MLUBaseOp** op,
                                            void* active_func_ptr,
                                            MLUTensor* input,
                                            MLUTensor* output);
tensorflow::Status ComputeCustomizedActiveOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                             void* input, void* output);
/******************************************************/

tensorflow::Status CreateCycleAddOp(MLUBaseOp** op, int dim, MLUTensor* input1,
                                    MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeCycleAddOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input1, void* input2, void* output);
/******************************************************/

tensorflow::Status CreateCycleAndOp(MLUBaseOp** op, MLUTensor* input1,
                                    MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeCycleAndOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input1, void* input2, void* output);
/******************************************************/

tensorflow::Status CreateCycleEqualOp(MLUBaseOp** op, MLUTensor* input1,
                                    MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeCycleEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input1, void* input2, void* output);
/******************************************************/

tensorflow::Status CreateCycleGreaterOp(MLUBaseOp** op, MLUTensor* in0,
                                        MLUTensor* in1, MLUTensor* output);
tensorflow::Status ComputeCycleGreaterOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input0, void* input1,
                                         void* output);
/******************************************************/

tensorflow::Status CreateCycleGreaterEqualOp(MLUBaseOp** op, MLUTensor* in0,
                                        MLUTensor* in1, MLUTensor* output);
tensorflow::Status ComputeCycleGreaterEqualOp(MLUBaseOp* op,
                                         MLUCnrtQueue* queue,
                                         void* input0, void* input1,
                                         void* output);
/******************************************************/

tensorflow::Status CreateCycleLessOp(MLUBaseOp** op, MLUTensor* in0,
                                     MLUTensor* in1, MLUTensor* output);
tensorflow::Status ComputeCycleLessOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input0, void* input1,
                                      void* output);
/******************************************************/

tensorflow::Status CreateCycleLessEqualOp(MLUBaseOp** op, MLUTensor* in0,
                                     MLUTensor* in1, MLUTensor* output);
tensorflow::Status ComputeCycleLessEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input0, void* input1,
                                      void* output);
/******************************************************/

tensorflow::Status CreateCycleMulOp(MLUBaseOp** op, MLUTensor* in0,
                                    MLUTensor* in1, MLUTensor* output);
tensorflow::Status ComputeCycleMulOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input0, void* input1, void* output);
/******************************************************/

tensorflow::Status CreateCycleOrOp(MLUBaseOp** op, MLUTensor* in0,
                                   MLUTensor* in1, MLUTensor* output);
tensorflow::Status ComputeCycleOrOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input0, void* input1, void* output);
/******************************************************/

tensorflow::Status CreateCycleSubOp(MLUBaseOp** op, MLUTensor* in0,
                                    MLUTensor* in1, MLUTensor* output);
tensorflow::Status ComputeCycleSubOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input0, void* input1, void* output);
/******************************************************/

tensorflow::Status CreateDeConvOp(MLUBaseOp** op, MLUTensor* in,
                                  MLUTensor* output, MLUTensor* filter,
                                  MLUTensor* bias, int stride_height,
                                  int stride_width, int hu, int hd,
                                  int wl, int wr);
tensorflow::Status ComputeDeConvOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output);
/******************************************************/

tensorflow::Status CreateSnapshotOp(MLUBaseOp** op, MLUTensor* in,
                                    MLUTensor* output);
tensorflow::Status ComputeSnapshotOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input, void* output);
/******************************************************/

tensorflow::Status CreateRoundOp(MLUBaseOp** op, MLUTensor* in,
                                 MLUTensor* output);
tensorflow::Status ComputeRoundOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

tensorflow::Status CreateConvDepthwiseOp(MLUBaseOp** op, MLUTensor* in,
                                         MLUTensor* filter, MLUTensor* bias,
                                         MLUTensor* output, int stride_height,
                                         int stride_width);
tensorflow::Status ComputeDepthwiseConvOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                          void* input, void* output);
/******************************************************/

tensorflow::Status CreateEluOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output);
tensorflow::Status ComputeEluOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output);
/******************************************************/

tensorflow::Status CreateEqualOp(MLUBaseOp** op, MLUTensor* in0, MLUTensor* in1,
                                 MLUTensor* output);
tensorflow::Status ComputeEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input0, void* input1, void* output);
/******************************************************/

tensorflow::Status CreateErfOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output);
tensorflow::Status ComputeErfOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output);
/******************************************************/

tensorflow::Status CreateExpOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output);
tensorflow::Status ComputeExpOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output);
/******************************************************/

tensorflow::Status CreateFloorOp(MLUBaseOp** op, MLUTensor* input,
                                 MLUTensor* output);
tensorflow::Status ComputeFloorOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/


tensorflow::Status CreateGatherOp(MLUBaseOp** op, MLUTensor* input1,
                                  MLUTensor* input2, MLUTensor* output,
                                  MLUDimension_t gather_mode);
tensorflow::Status ComputeGatherOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input1, void* input2,
                                   void* output);
/******************************************************/

tensorflow::Status CreateGreaterEqualOp(MLUBaseOp** op, MLUTensor* input1,
                                        MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeGreaterEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input1, void* input2,
                                         void* output);
/******************************************************/

tensorflow::Status CreateGreaterOp(MLUBaseOp** op, MLUTensor* input1,
                                   MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeGreaterOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input1, void* input2, void* output);
/******************************************************/

tensorflow::Status CreateInterpOp(MLUBaseOp** op, MLUTensor* input,
                                  MLUTensor* output, int output_height,
                                  int output_width, bool align_corners);
tensorflow::Status ComputeInterpOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output);
/******************************************************/

tensorflow::Status CreateInvertPermutationOp(MLUBaseOp** op,
    MLUTensor* input, MLUTensor* output);
tensorflow::Status ComputeInvertPermutationOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* input, void* output);
/******************************************************/

tensorflow::Status CreateIsFiniteOp(MLUBaseOp** op, MLUTensor* input,
                                  MLUTensor* output);
tensorflow::Status ComputeIsFiniteOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output);
/******************************************************/
tensorflow::Status CreateLessEqualOp(MLUBaseOp** op, MLUTensor* input1,
                                     MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeLessEqualOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input1, void* input2,
                                      void* output);
/******************************************************/

tensorflow::Status CreateLessOp(MLUBaseOp** op, MLUTensor* input1,
                                MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeLessOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input1, void* input2, void* output);
/******************************************************/

tensorflow::Status CreateMlpOp(MLUBaseOp** op, MLUTensor* input,
    MLUTensor* output, MLUTensor* filter, MLUTensor* bias);
tensorflow::Status ComputeMlpOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* input, void* output);
/******************************************************/

tensorflow::Status CreateNearestNeighborOp(MLUBaseOp** op,
    MLUTensor* input, MLUTensor* output,
    int output_height, int output_width, bool align_corners);
tensorflow::Status ComputeNearestNeighborOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* input, void* output);

/******************************************************/

tensorflow::Status CreateLrnOp(MLUBaseOp** op, MLUTensor* input,
    MLUTensor* output, MLULrnType lrn_type, int local_size,
    double alph, double beta, double k);
tensorflow::Status ComputeLrnOp(MLUBaseOp* op, MLUCnrtQueue* queue,
  void* input, void* output);
/******************************************************/

tensorflow::Status CreateNegOp(MLUBaseOp** op, int dim, MLUTensor* input,
                           MLUTensor* alpha, MLUTensor* beta,
                           MLUTensor* output);
tensorflow::Status ComputeNegOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

tensorflow::Status CreateOneHotOp(MLUBaseOp** op, MLUTensor* input,
                                MLUTensor* output, int *shape, int depth,
                                float on_value, float off_value, int axis);
tensorflow::Status ComputeOneHotOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* inputs[], int in_size,
                                  void* outputs[], int out_size);
/******************************************************/

tensorflow::Status CreateOnesLikeOp(MLUBaseOp** op, int dim, MLUTensor* input,
                                MLUTensor* alpha, MLUTensor* beta,
                                MLUTensor* output);
tensorflow::Status ComputeOnesLikeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

tensorflow::Status CreatePad4Op(MLUBaseOp** op, MLUTensor* input,
     MLUTensor* output, int padding_htop, int padding_hbottom,
     int padding_wleft, int padding_wright, float pad_value);
tensorflow::Status ComputePad4Op(MLUBaseOp* op, MLUCnrtQueue* queue,
     void* input, void* output);
/******************************************************/

tensorflow::Status CreatePoolOp(
    MLUBaseOp** op, MLUTensor* input, MLUTensor* output, bool real,
    const std::vector<int>& kernel_size, const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::vector<std::pair<int, int>>& paddings,
    MLUPoolMode pool_mode, MLUPoolStrategyMode pool_strategy_mode);
tensorflow::Status ComputePoolOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output);
/******************************************************/

tensorflow::Status CreatePowOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output, float c);
tensorflow::Status ComputePowOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output);
/******************************************************/

tensorflow::Status CreateRealDivOp(MLUBaseOp** op, MLUTensor* input1,
                                   MLUTensor* input2, MLUTensor* output,
                                   bool high_precision_flag);
tensorflow::Status ComputeRealDivOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input0, void* input1, void* output0);
/******************************************************/

tensorflow::Status CreateReduceAllOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                     MLUTensor* output);
tensorflow::Status ComputeReduceAllOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output);
/******************************************************/

tensorflow::Status CreateReduceAnyOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                     MLUTensor* output);
tensorflow::Status ComputeReduceAnyOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output);
/******************************************************/

tensorflow::Status CreateReduceMaxOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                     MLUTensor* output);
tensorflow::Status ComputeReduceMaxOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output);
/******************************************************/

tensorflow::Status CreateReduceMeanOp(MLUBaseOp** op, int axis,
                                      MLUTensor* input, MLUTensor* output);
tensorflow::Status ComputeReduceMeanOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                       void* input, void* output);
/******************************************************/

tensorflow::Status CreateReduceSumOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                     MLUTensor* output);
tensorflow::Status ComputeReduceSumOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output);
/******************************************************/

tensorflow::Status CreateReshapeOp(MLUBaseOp** op, MLUTensor* input,
                                   MLUTensor* output);
tensorflow::Status ComputeReshapeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input, void* output);
/******************************************************/

tensorflow::Status CreateReverseOp(MLUBaseOp** op, MLUTensor* input,
                                   MLUTensor* output, int axis);
tensorflow::Status ComputeReverseOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input, void* output);
/******************************************************/

tensorflow::Status CreateRoundOp(MLUBaseOp** op, MLUTensor* input,
                                 MLUTensor* output);
tensorflow::Status ComputeRoundOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

tensorflow::Status CreateRsqrtOp(MLUBaseOp** op, MLUTensor* input,
                                 MLUTensor* output);
tensorflow::Status ComputeRsqrtOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

tensorflow::Status CreateScaleOp(MLUBaseOp** op, int dim, MLUTensor* input,
                                 MLUTensor* alpha, MLUTensor* beta,
                                 MLUTensor* output);
tensorflow::Status ComputeScaleOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

tensorflow::Status CreateSelectOp(MLUBaseOp** op, MLUTensor* input0,
                                  MLUTensor* input1, MLUTensor* input2,
                                  MLUTensor* output, bool bool_index,
                                  bool batch_index);
tensorflow::Status ComputeSelectOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input0, void* input1, void* input2,
                                   void* output);
/******************************************************/

tensorflow::Status CreateSeluOp(MLUBaseOp** op, MLUTensor* in,
                               MLUTensor* output);
tensorflow::Status ComputeSeluOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input, void* output);
/******************************************************/

tensorflow::Status CreateSoftmaxOp(MLUBaseOp** op, int dim,
                                   MLUTensor* input, MLUTensor* output);
tensorflow::Status ComputeSoftmaxOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                    void* input, void* output);
/******************************************************/

tensorflow::Status CreateSpace2BatchOp(MLUBaseOp** op, int w_block_size,
                                       int h_block_size, MLUTensor* input,
                                       MLUTensor* output);
tensorflow::Status ComputeSpace2BatchOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* input, void* output);
/******************************************************/

tensorflow::Status CreateSplitOp(MLUBaseOp** op, int axis, MLUTensor* input,
                                 MLUTensor* outputs[], int output_num);
tensorflow::Status ComputeSplitOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* outputs[],
                                  int output_num);
/******************************************************/

tensorflow::Status CreateSqrtOp(MLUBaseOp** op, MLUTensor* input,
                                MLUTensor* output);
tensorflow::Status ComputeSqrtOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output);
/******************************************************/

tensorflow::Status CreateSquareOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output);
tensorflow::Status ComputeSquareOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input, void* output);
/******************************************************/

tensorflow::Status CreateSquaredDifferenceOp(MLUBaseOp** op, MLUTensor* input1,
                                             MLUTensor* input2,
                                             MLUTensor* output);
tensorflow::Status ComputeSquaredDifferenceOp(MLUBaseOp* op,
                                              MLUCnrtQueue* queue, void* input1,
                                              void* input2, void* output);
/******************************************************/

tensorflow::Status CreateNdStridedSliceOp(MLUBaseOp** op, MLUTensor* input,
                                          MLUTensor* output, int dim_num,
                                          int begin[], int end[],
                                          int stride[]);
tensorflow::Status ComputeNdStridedSliceOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                           void* input, void* output);
/******************************************************/

tensorflow::Status CreateSubOp(MLUBaseOp** op, MLUTensor* input1,
                               MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeSubOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                void* input1, void* input2, void* output);
/******************************************************/

tensorflow::Status CreateTileOp(MLUBaseOp** op, MLUTensor* input,
                                MLUTensor* output);
tensorflow::Status ComputeTileOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output);
/******************************************************/

tensorflow::Status CreateTopKOp(MLUBaseOp** op, int k, bool sorted,
                                MLUTensor* input, MLUTensor* values_out,
                                MLUTensor* indices_out);
tensorflow::Status ComputeTopKOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                 void* input, void* output, void* index);
/******************************************************/

tensorflow::Status CreateTransposeProOp(MLUBaseOp** op, MLUTensor* input,
                                        MLUTensor* output, int dim_order[],
                                        int dim_num);
tensorflow::Status ComputeTransposeProOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* input, void* output);
/******************************************************/

tensorflow::Status CreateBiasAddGradOp(MLUBaseOp** op, MLUTensor* input,
                                       MLUTensor* output);
tensorflow::Status ComputeBiasAddGradOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* input, void* output);
/******************************************************/

tensorflow::Status CreateCosOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output);
tensorflow::Status ComputeCosOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output);
/******************************************************/

tensorflow::Status CreateConvFilterGradOp(MLUBaseOp** op,
    MLUTensor* x, MLUTensor* dy, MLUTensor* x_quant, MLUTensor* dy_quant,
    MLUTensor* dw, int kernel_height, int kernel_width, int stride_height,
    int stride_width, int dilation_height, int dilation_width, int pad_top,
    int pad_bottom, int pad_left, int pad_right);
tensorflow::Status ComputeConvFilterGradOp(MLUBaseOp* op, MLUCnrtQueue* queue,
    void* x, void* dy, void* x_quant, void* dy_quant, void* dw);
/******************************************************/

tensorflow::Status CreateFloorDivOp(MLUBaseOp** op, MLUTensor* input1,
                                    MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeFloorDivOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input1, void* input2,
                                     void* output);
/******************************************************/

tensorflow::Status CreateFusedBatchNormGradOp(
    MLUBaseOp** op, MLUTensor* x, MLUTensor* dy, MLUTensor* mean,
    MLUTensor* variance, MLUTensor* scale, MLUTensor* dx, MLUTensor* d_gamma,
    MLUTensor* d_beta, float epsilon);
tensorflow::Status ComputeFusedBatchNormGradOp(MLUBaseOp* op,
                                               MLUCnrtQueue* queue, void* x,
                                               void* y, void* dz, void* mean,
                                               void* variance, void* gamma,
                                               void* dx, void* d_gamma,
                                               void* d_beta);
/******************************************************/

tensorflow::Status CreateFusedBatchNormOp(MLUBaseOp** op, MLUTensor* input,
                                          MLUTensor* es_mean, MLUTensor* es_var,
                                          MLUTensor* gamma, MLUTensor* beta,
                                          MLUTensor* eps, MLUTensor* output,
                                          MLUTensor* batch_mean,
                                          MLUTensor* batch_var, MLUTensor* mean,
                                          MLUTensor* var);
tensorflow::Status ComputeFusedBatchNormOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                           void* input, void* es_mean,
                                           void* es_var, void* gamma,
                                           void* beta, void* output,
                                           void* batch_mean, void* batch_var,
                                           void* mean, void* var);
/******************************************************/

tensorflow::Status CreateL2LossOp(MLUBaseOp** op, MLUTensor* x,
                                  MLUTensor* output);
tensorflow::Status ComputeL2LossOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input_label, void* input_predicted,
                                   void* output);
/******************************************************/

tensorflow::Status CreateListDiffOp(MLUBaseOp** op, MLUTensor* x, MLUTensor* y,
                                    MLUTensor* output_data,
                                    MLUTensor* output_index);
tensorflow::Status ComputeListDiffOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* inputX, void* inputY, void* out,
                                     void* out_idx);
/******************************************************/

tensorflow::Status CreateLogOp(MLUBaseOp** op, MLUTensor* input,
                                      MLUTensor* output);
tensorflow::Status ComputeLogOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                       void* input, void* output);
/******************************************************/

tensorflow::Status CreateLogicalNotOp(MLUBaseOp** op, MLUTensor* input,
                                      MLUTensor* output);
tensorflow::Status ComputeLogicalNotOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                       void* input, void* output);
/******************************************************/

tensorflow::Status CreateLogicalOrOp(MLUBaseOp** op, MLUTensor* input1,
                                     MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeLogicalOrOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input1, void* input2,
                                      void* output);
/******************************************************/

tensorflow::Status CreateReciprocalOp(MLUBaseOp** op, MLUTensor* input,
                                      MLUTensor* output);
tensorflow::Status ComputeReciprocalOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* output);
/******************************************************/

tensorflow::Status CreateMaximumOp(MLUBaseOp** op, MLUTensor* input1,
                                   MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeMaximumOpForward(MLUBaseOp* op, MLUCnrtQueue* queue,
                                           void* input1, void* input2,
                                           void* output);
/******************************************************/

tensorflow::Status CreateMinimumOp(MLUBaseOp** op, MLUTensor* input1,
                                   MLUTensor* input2, MLUTensor* output);
tensorflow::Status ComputeMinimumOpForward(MLUBaseOp* op, MLUCnrtQueue* queue,
                                           void* input1, void* input2,
                                           void* output);
/******************************************************/

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
                                        bool real);
tensorflow::Status ComputeMaxPoolIndexOp(MLUBaseOp* op,
                                         MLUCnrtQueue *queue,
                                         void* input,
                                         void* output_no_use,
                                         void* index);
/******************************************************/

tensorflow::Status CreatePoolBackwardOp(MLUBaseOp** op, MLUTensor* out_backprop,
    MLUTensor* index, MLUTensor* output,
    int window_height, int window_width,
    int stride_height, int stride_width,
    int pad_left, int pad_right,
    int pad_up, int pad_down,
    MLUPoolBackwardMode poolbp_mode,
    MLUPoolBackwardStrategyMode padding_mode);
tensorflow::Status ComputePoolBackwardOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                         void* out_backprop, void* index,
                                         void* output);
/******************************************************/

tensorflow::Status CreateQuantifyOp(MLUBaseOp** op, MLUTensor* input,
                                    MLUTensor* oldQuanParams,
                                    MLUTensor* oldMovPos, MLUTensor* output,
                                    MLUTensor* quanParams, MLUTensor* movPos,
                                    MLUTensor* interval);
tensorflow::Status ComputeQuantifyOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* input, void* input_param,
                                     void* input_mp, void* output,
                                     void* output_param, void* output_mp,
                                     void* interval);
/******************************************************/

tensorflow::Status CreateQuantMatMulOp(MLUBaseOp** op,
                                       MLUTensor* input,
                                       MLUTensor* filter,
                                       MLUTensor* input_param,
                                       MLUTensor* filter_param,
                                       MLUTensor* bias,
                                       MLUTensor* output);
tensorflow::Status ComputeQuantMatMulOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                        void* input, void* filter,
                                        void* input_param, void* filter_param,
                                        void* bias, void* output);
/******************************************************/

tensorflow::Status CreateRandomUniformOp(MLUBaseOp** op,
                                         MLUTensor* output, int seed);
tensorflow::Status ComputeRandomUniformOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                          void* output);
/******************************************************/

tensorflow::Status CreateRangeOp(MLUBaseOp** op,
                                 MLUTensor* start, MLUTensor* limit, MLUTensor* delta,
				 int size, MLUTensor* output,
                                 cnmlPluginRangeOpParam_t param);
tensorflow::Status ComputeRangeOp(MLUBaseOp* range_op, MLUCnrtQueue* queue,
                                  void* inputs[], int input_num,
                                  void* outputs[], int output_num);
/******************************************************/

tensorflow::Status CreateReduceProdOp(MLUBaseOp** cnml_reduce_prod_op_ptr_ptr,
                                      int axis, MLUTensor* input,
                                      MLUTensor* output);
tensorflow::Status ComputeReduceProdOp(MLUBaseOp* cnml_reduce_prod_op_ptr,
                                       MLUCnrtQueue* queue, void* input,
                                       void* output);
/******************************************************/

tensorflow::Status CreateReluGradOp(MLUBaseOp** op, MLUTensor* dy, MLUTensor* x,
                                    MLUTensor* output);
tensorflow::Status ComputeReluGradOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                     void* x, void* dy, void* output);
/******************************************************/

tensorflow::Status CreateRsqrtOpBackward(MLUBaseOp** op, MLUTensor* x,
                                         MLUTensor* dy, MLUTensor* output);
tensorflow::Status ComputeRsqrtOpBackward(MLUBaseOp* op, MLUCnrtQueue* queue,
                                          void* y, void* dy, void* output);
/******************************************************/

tensorflow::Status CreateScatterNdOpParam(MLUScatterNdOpParam** param,
                                          cnmlDimension_t axis,
                                          int scatter_length);
tensorflow::Status DestroyScatterNdOpParam(MLUScatterNdOpParam** param);
tensorflow::Status CreateScatterNdOp(MLUBaseOp** op, MLUTensor* input1,
                                     MLUTensor* input2, MLUTensor* output,
                                     MLUScatterNdOpParam* param);
tensorflow::Status ComputeScatterNdOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                      void* input, void* index, void* output);
/******************************************************/

tensorflow::Status CreateSinOp(MLUBaseOp** op, MLUTensor* input,
                               MLUTensor* output);
tensorflow::Status ComputeSinOp(MLUBaseOp* op, MLUCnrtQueue* queue, void* input,
                                void* output);
tensorflow::Status CreateSoftmaxXentWithLogitsOp(MLUBaseOp** op,
                                                 int dim,
                                                 MLUTensor* input,
                                                 MLUTensor* label,
                                                 MLUTensor* output,
						 MLUTensor* back_out);
tensorflow::Status ComputeSoftmaxXentWithLogitsOp(MLUBaseOp* op,
                                                  MLUCnrtQueue* queue,
                                                  void* input, void* label,
                                                  void* output, void* back_out);
/******************************************************/

tensorflow::Status CreateStridedSliceOpBackward(
    MLUBaseOp** op_ptr, MLUTensor* input, MLUTensor* output,
    const std::vector<int>& begin, const std::vector<int>& end,
    const std::vector<int>& strides);
tensorflow::Status ComputeStridedSliceGradOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                             void* input, void* output);
/******************************************************/

tensorflow::Status CreateUniqueOp(MLUBaseOp** op, MLUTensor* input,
                                  MLUTensor* output, MLUTensor* idx);
tensorflow::Status ComputeUniqueOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                   void* input, void* output, void* index);
/******************************************************/

tensorflow::Status CreateUnsortedSegmentSumOp(MLUBaseOp** op,
                                              MLUTensor* data,
                                              MLUTensor* segment_ids,
                                              MLUTensor* output,
                                              int num_segments,
                                              int data_dims);
tensorflow::Status ComputeUnsortedSegmentSumOp(MLUBaseOp* op,
                                               MLUCnrtQueue* queue,
                                               void* data, void* segment_ids,
                                               void* output);
/******************************************************/

tensorflow::Status CreateZerosLikeOp(MLUBaseOp** op, int dim, MLUTensor* input,
                                 MLUTensor* alpha, MLUTensor* beta,
                                 MLUTensor* output);
tensorflow::Status ComputeZerosLikeOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

/******************************************************/

tensorflow::Status CreateLogSoftmaxOp(MLUBaseOp** op, int dim,
                                 MLUTensor* input, MLUTensor* output);
tensorflow::Status ComputeLogSoftmaxOp(MLUBaseOp* op, MLUCnrtQueue* queue,
                                  void* input, void* output);
/******************************************************/

/******************************************************/

tensorflow::Status CreateNonMaxSuppressionOp(MLUBaseOp** op,
    MLUTensor* input_boxes, MLUTensor* input_scores, MLUTensor* output,
    cnmlPluginNonMaxSuppressionOpParam_t param);
tensorflow::Status ComputeNonMaxSuppressionOp(MLUBaseOp* op,
    void* input_boxes, void* input_scores, void* output,
    MLUCnrtQueue* queue);
/******************************************************/

tensorflow::Status CreateMatrixBandPartOp(MLUBaseOp** op,
                                          MLUTensor* input,
                                          MLUTensor* output,
                                          int num_lower,
                                          int num_upper);
tensorflow::Status ComputeMatrixBandPartOp(MLUBaseOp* op,
                                           MLUCnrtQueue* queue,
                                           void* input,
                                           void* output);
/******************************************************/
/******************************************************/

tensorflow::Status CreateLeakyReluOp(MLUBaseOp** op,
    MLUTensor* features, MLUTensor* alpha, MLUTensor* output,
    int dim);
tensorflow::Status ComputeLeakyReluOp(MLUBaseOp* op,
    MLUCnrtQueue* queue, void* features, void* output);
/******************************************************/
}  // namespace lib
}  // namespace mlu
}  // namespace stream_executor
#endif  // TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_API_LIB_OPS_MLU_LIB_OPS_H_
