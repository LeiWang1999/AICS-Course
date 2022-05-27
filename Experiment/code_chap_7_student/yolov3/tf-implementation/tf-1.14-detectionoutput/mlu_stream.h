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

// The Stream is used in conjunction with the MLUStreamExecutor "parent" to
// perform actions with a linear stream of dependencies. Dependencies can also
// be created between Streams to do task management (i.e. limit which tasks
// can be performed concurrently and specify what task dependencies exist).

#ifndef TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_STREAM_H_

#include <vector>
#include <complex>
#include <functional>
#include <memory>
#include <map>
#include <set>
#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/platform/default/mutex.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/stream_executor/mlu/mlu_stream_executor.h"
#include "tensorflow/stream_executor/mlu/mlu_stream_records.h"
#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_common.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"




namespace stream_executor {
namespace mlu {

using tensorflow::Tensor;
using tensorflow::TensorBuffer;
using tensorflow::Allocator;
using tensorflow::OpKernelContext;
using tensorflow::AllocatorAttributes;
using tensorflow::StepStatsCollector;
using tensorflow::TensorFormat;
using tensorflow::TensorShape;
using tensorflow::NodeDef;


// Represents a stream of dependent computations on a MLU device.
//
// The operations within a stream execute linearly and asynchronously until
// BlockHostUntilDone() is invoked, which synchronously joins host code with
// the execution of the stream.
//
// If any given operation fails when entraining work for the stream, ok() will
// indicate that an error has occurred. After initialization, once a stream is
// !ok(), it will never be ok().
//
// Thread-safe post-initialization.
class MLUStream : public internal::StreamInterface {
 public:
  explicit MLUStream(MLUStreamExecutor *parent);
  ~MLUStream();

  bool Init();
  void Destroy();

  // Common op functions
  int device_ordinal() { return device_ordinal_; }
  unsigned long GetCnrtDev() { return dev_; }
  MLUCoreVersion GetMLUCoreVersion() const { return core_version_; }
  int GetCoreNum() const { return core_num_; }
  MLUCnrtQueue* GetCnrtQueue() { return cnrt_queue_; }

  Status sync() {
    if (device_ordinal_ == -1) {
      return Status::OK();
    }
    return lib::MLUCnrtSyncQueue(cnrt_queue_);
  }

  // MLU Ops implementations
  Status AvgPoolBackProp(OpKernelContext* ctx,
                         Tensor* tensor_in, Tensor* out_backprop,
                         Tensor* output, std::string pad_mode,
                         int window_height, int window_width,
                         int stride_height, int stride_width,
                         int pad_left, int pad_right,
                         int pad_up, int pad_down) {
    ops::MLUPoolBackpropParam op_param(window_height, window_width,
                                       stride_height, stride_width,
                                       pad_left, pad_right,
                                       pad_up, pad_down, pad_mode);
    return CommonOpImpl<ops::MLUAvgPoolBackprop>(ctx,
         {tensor_in, out_backprop}, {output}, static_cast<void*>(&op_param));
  }

  Status Yolov3DetectionOutput(OpKernelContext* ctx,
                        Tensor* tensor_input0,
                        Tensor* tensor_input1,
                        Tensor* tensor_input2,
                        int batchNum,
                        int inputNum,
                        int classNum,
                        int maskGroupNum,
                        int maxBoxNum,
                        int netw,
                        int neth,
                        float confidence_thresh,
                        float nms_thresh,
                        int* inputWs,
                        int* inputHs,
                        float* biases,
                        Tensor* output1,
                        Tensor* output2){
    ops::MLUYolov3DetectionOutputOpParam op_param(
                        batchNum,
                        inputNum,
                        classNum,
                        maskGroupNum,
                        maxBoxNum,
                        netw,
                        neth,
                        confidence_thresh,
                        nms_thresh,
                        inputWs,
                        inputHs,
                        biases);
//TODO:补齐下面函数操作
    return CommonOpImpl<ops::MLUYolov3DetectionOutput>(......);
  }

  Status MaxPoolBackProp(OpKernelContext* ctx,
                         Tensor* tensor_in, Tensor* out_backprop,
                         Tensor* output, std::string pad_mode,
                         int window_height, int window_width,
                         int stride_height, int stride_width,
                         int pad_left, int pad_right,
                         int pad_up, int pad_down) {
    ops::MLUPoolBackpropParam op_param(window_height, window_width,
                                       stride_height, stride_width,
                                       pad_left, pad_right,
                                       pad_up, pad_down, pad_mode);
    return CommonOpImpl<ops::MLUMaxPoolBackprop>(ctx,
         {tensor_in, out_backprop}, {output}, static_cast<void*>(&op_param));
  }

  Status Abs(OpKernelContext* ctx, Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUAbs>(ctx, {input}, {output});
  }

  Status Add(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_ADD;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Addn(OpKernelContext* ctx,
      std::vector<Tensor*> inputs, int num, Tensor* output) {
    return CommonOpImpl<ops::MLUAddn>(ctx,
        inputs, {output}, static_cast<void*>(&num));
  }

  Status And(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_AND;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status BinaryOp(OpKernelContext* ctx, ops::MLUCwiseMethod method,
      Tensor* input1, Tensor* input2, Tensor* output) {
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Cos(OpKernelContext* ctx, Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUCos>(ctx, {input}, {output});
  }

  Status LogicalOr(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_OR;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Log1p(OpKernelContext* ctx,
      Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLULog1p>(ctx,
      {input}, {output}, nullptr);
  }

  Status Clone(OpKernelContext* ctx, Tensor* input, Tensor* output);

  Status Gather(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output, int axis) {
    ops::MLUGatherOpParam p(axis);
    return CommonOpImpl<ops::MLUGather>(ctx, {input1, input2}, {output},
    static_cast<void*>(&p));
  }

  Status ScatterNd(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    return CommonOpImpl<ops::MLUScatterNd>(ctx, {input1, input2},
        {output}, nullptr);
  }

  Status Split(OpKernelContext* ctx, const std::vector<Tensor*>& inputs,
      int split_dim, const std::vector<Tensor*>& outputs) {
    ops::MLUSplitOpParam op_param(split_dim);
    return CommonOpImpl<ops::MLUSplit>(ctx, inputs, outputs,
        static_cast<void*>(&op_param));
  }

  Status Square(OpKernelContext* ctx, Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUSquare>(ctx, {input}, {output});
  }

  Status Sub(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_SUB;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Maximum(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_MAX;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Minimum(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_MIN;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Mul(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_MUL;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status RealDiv(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_DIV;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Pow(OpKernelContext* ctx, Tensor* input, Tensor* output, float pow_c) {
    return CommonOpImpl<ops::MLUPow>(ctx, {input}, {output},
        static_cast<void*>(&pow_c));
  }

  Status IsFinite(OpKernelContext* ctx, Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUIsFinite>(ctx, {input}, {output},
        nullptr);
  }

  Status Exp(OpKernelContext* ctx, Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUExp>(ctx, {input}, {output},
        nullptr);
  }

  Status Pack(OpKernelContext* ctx,
      std::vector<Tensor *> inputs, Tensor *output, int axis) {
    return CommonOpImpl<ops::MLUPack>(ctx, inputs, {output},
        static_cast<void*>(&axis));
  }

  Status Unpack(OpKernelContext* ctx, std::vector<Tensor *> inputs,
      std::vector<Tensor *> outputs, int axis) {
    return CommonOpImpl<ops::MLUUnpack>(ctx, inputs, outputs,
        static_cast<void*>(&axis));
  }

  Status Fill(OpKernelContext* ctx, Tensor* output, float value) {
    return CommonOpImpl<ops::MLUFill>(ctx, {}, {output},
        static_cast<void*>(&value));
  }

  Status ZerosLike(OpKernelContext* ctx, Tensor* output) {
    float value = 0.0f;
    return CommonOpImpl<ops::MLUFill>(ctx, {}, {output},
        static_cast<void*>(&value));
  }

  Status OnesLike(OpKernelContext* ctx, Tensor* output) {
    float value = 1.0f;
    return CommonOpImpl<ops::MLUFill>(ctx, {}, {output},
        static_cast<void*>(&value));
  }

  Status Neg(OpKernelContext* ctx,
      Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUNeg>(ctx,
      {input}, {output}, nullptr);
  }

  Status BatchNorm(OpKernelContext* ctx, Tensor* input, Tensor* mean,
      Tensor* var, Tensor* output, int dim);

  Status BertSquad(OpKernelContext* ctx,
      std::vector<Tensor*> inputs, std::vector<Tensor*> outputs,
      std::vector<MLUTensorType> inputs_mlu_tensors_type,
      int* params);

  Status FusedBatchNorm(OpKernelContext* ctx,
      Tensor* input, Tensor* es_mean, Tensor* es_var, Tensor* gamma,
      Tensor* beta, Tensor* eps, Tensor* output, Tensor* batch_mean,
      Tensor* batch_var, Tensor* mean, Tensor* var, bool is_training);

  Status Pad4(OpKernelContext* ctx,
      Tensor* input, int padding_htop, int padding_hbottom,
      int padding_wleft, int padding_wright,
      float pad_value, Tensor* output) {
        ops::MLUPad4OpParam p(padding_htop, padding_hbottom,
          padding_wleft, padding_wright, pad_value);
        return CommonOpImpl<ops::MLUPad4>(ctx, {input}, {output},
        static_cast<void*>(&p));
      }

  Status BatchMatMul(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2,
      bool adj_x, bool adj_y, Tensor* output) {
    ops::MLUBatchMatmulOpParam op_param(adj_x, adj_y);
    return CommonOpImpl<ops::MLUBatchMatmul>(ctx, {input1, input2},
        {output}, static_cast<void*>(&op_param));
  }


  Status Lrn(OpKernelContext* ctx,
      Tensor* input, Tensor* output, MLULrnType type,
      int local_size, double alpha, double beta, double k,
      MLUDataType dtype = MLU_DATA_FLOAT16,
      int position = DEFAULT_QUANT_POSITION,
      float scale = DEFAULT_QUANT_SCALE) {
      ops::MLULrnOpParam p(type, local_size, alpha, beta, k,
        dtype, position, scale);
      return CommonOpImpl<ops::MLULrn>(ctx, {input}, {output},
        static_cast<void*>(&p));
      }

  Status Slice(OpKernelContext* ctx, Tensor* input, Tensor* output,
      const std::vector<int64>& starts) {
    return CommonOpImpl<ops::MLUCrop>(ctx, {input}, {output},
        static_cast<void*>(const_cast<int64*>(starts.data())));
  }

  Status StridedSlice(OpKernelContext* ctx,
      Tensor* input, const std::vector<int> &begin,
      const std::vector<int> &end, const std::vector<int> &strides,
      Tensor* output, const bool is_identity,
      const std::vector<int> &strided_slice_shape,
      const std::vector<int> &final_shape) {
    ops::MLUStridedSliceOpParam p(begin, end, strides,
        is_identity, strided_slice_shape, final_shape);
    return CommonOpImpl<ops::MLUStridedSlice>(ctx, {input}, {output},
        static_cast<void*>(&p));
  }

  Status Softmax(OpKernelContext* ctx,
      Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUSoftmax>(ctx, {input}, {output},
        nullptr);
  }

  Status LogSoftmax(OpKernelContext* ctx,
      Tensor* input, Tensor* output, int dim) {
    return CommonOpImpl<ops::MLULogSoftmax>(ctx, {input}, {output},
        static_cast<void*>(&dim));
  }

  Status SoftmaxXentWithLogits(OpKernelContext* ctx,
      Tensor* input, Tensor* label, Tensor* output, Tensor* back_out, int in_dim) {
    return CommonOpImpl<ops::MLUSoftmaxXentWithLogits>(ctx,
        {input, label}, {output, back_out}, static_cast<void*>(&in_dim));
  }

  Status Conv2D(OpKernelContext* ctx,
      Tensor* input, Tensor* filter, Tensor* bias,
      int row_stride, int col_stride,
      int padding_rows, int padding_cols,
      int dilation_rows, int dilation_cols,
      Tensor* output, TensorFormat tensor_format);

  Status QuantConv2D(OpKernelContext* ctx,
      Tensor* input_quantized,
      Tensor* input_quantized_param,
      Tensor* filter_quantized,
      Tensor* filter_quantized_param,
      Tensor* bias, int row_stride,
      int col_stride, int padding_rows,
      int padding_cols, int dilation_rows,
      int dilation_cols, Tensor* output){

    ops::MLUCNNOpsParam op_param;
    op_param.strides = {row_stride, col_stride};
    op_param.paddings = {padding_rows, padding_cols};
    op_param.dilations = {dilation_rows, dilation_cols};

    return CommonOpImpl<ops::MLUQuantConv2D>(ctx,
                {input_quantized, filter_quantized,
                input_quantized_param, filter_quantized_param, bias},
                {output}, static_cast<void*>(&op_param));
  }

  Status Conv2DBackpropInput(OpKernelContext* ctx,
      Tensor* w, Tensor* dy, Tensor* output,
      Tensor* w_param, Tensor* dy_param,
      int sh, int sw, int dh, int dw, int pad_top, int pad_bottom,
      int pad_left, int pad_right) {
    ops::MLUCNNOpsParam op_param;
    op_param.strides = {sh, sw};
    op_param.dilations = {dh, dw};
    op_param.paddings = {pad_top, pad_bottom, pad_left, pad_right};
    return CommonOpImpl<ops::MLUConv2DBackpropInput>(ctx,
        {w, dy, w_param, dy_param}, {output},
        static_cast<void*>(&op_param));
  }

  Status IntDeconv(OpKernelContext* ctx,
      Tensor* input, Tensor* filter, Tensor* bias,
      int stride_h, int stride_w,
      int hu, int hd,
      int dila_h, int dila_w,
      Tensor* output, TensorFormat tensor_format,
      MLUDataType dtype,
      float input_scale, float filter_scale);

  Status Conv2DBackpropFilter(OpKernelContext* ctx,
      Tensor* input, Tensor* out_backprop, Tensor* filter_backprop,
      Tensor* input_param, Tensor* out_backprop_param,
      int kernel_h, int kernel_w,
      int stride_h, int stride_w,
      int dilation_h, int dilation_w,
      int pad_top, int pad_bottom,
      int pad_left, int pad_right) {
    ops::MLUCNNOpsParam op_param;
    op_param.kernel_sizes = {kernel_h, kernel_w};
    op_param.strides = {stride_h, stride_w};
    op_param.dilations = {dilation_h, dilation_w};
    op_param.paddings = {pad_top, pad_bottom, pad_left, pad_right};
    return CommonOpImpl<ops::MLUConv2DBackpropFilter>(ctx,
        {input, out_backprop, input_param, out_backprop_param},
        {filter_backprop}, static_cast<void*>(&op_param));
  }

  Status DepthwiseConv2D(OpKernelContext* ctx,
    Tensor* input, Tensor* filter, Tensor* bias,
    int stride_height, int stride_width, int pad_height, int pad_width,
    Tensor* output, TensorFormat tensor_format);

  Status Active(OpKernelContext* ctx,
      Tensor* input, Tensor* output, MLUActiveFunction active_func) {
    return CommonOpImpl<ops::MLUActive>(ctx, {input}, {output},
        static_cast<void*>(&active_func));
  }

  Status Elu(OpKernelContext* ctx,
      Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUElu>(ctx, {input}, {output});
  }

  Status Selu(OpKernelContext* ctx,
      Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUSelu>(ctx, {input}, {output});
  }

  Status CustomizedActive(OpKernelContext* ctx,
      Tensor* input, Tensor* output, MLUActiveFunction active_func) {
    return CommonOpImpl<ops::MLUCustomizedActive>(ctx, {input}, {output},
        static_cast<void*>(&active_func));
  }

  Status Cast(OpKernelContext* ctx,
      Tensor* input, Tensor* output, MLUCastType cast_type) {
    return CommonOpImpl<ops::MLUCast>(ctx, {input}, {output},
        static_cast<void*>(&cast_type));
  }

  Status Pooling(OpKernelContext* ctx,
      Tensor* input, MLUPoolMode pool_mode,
      MLUPoolStrategyMode pool_strategy_mode, bool real,
      const std::vector<int>& kernel_size,
      const std::vector<int>& strides,
      const std::vector<std::pair<int, int>>& paddings,
      const std::vector<int>& dilations,
      Tensor* output, TensorFormat tensor_format) {
    ops::MLUPoolingOpParam p(pool_mode, pool_strategy_mode, real,
        kernel_size, strides, paddings, dilations, tensor_format);
    return CommonOpImpl<ops::MLUPooling>(ctx, {input}, {output},
        static_cast<void*>(&p));
  }

  Status FractionalPooling(OpKernelContext* ctx,
      Tensor* input, MLUPoolMode pool_mode,
      int* row_sequence, int row_num,
      int* col_sequence, int col_num,
      bool overlapping, Tensor* output);

  Status EltwiseAdd(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output);

  Status EltwiseSub(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output);

  Status EltwiseMul(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output);

  Status Quantify(OpKernelContext* ctx,
      Tensor *input, Tensor *old_quantparam,
      Tensor *old_movpos, Tensor *interval,
      Tensor *output, Tensor *quantparam,
      Tensor *movpos){
      return CommonOpImpl<ops::MLUQuantify>(ctx,
            {input, old_quantparam, old_movpos, interval},
            {output, quantparam, movpos}, nullptr);
  }

  Status FirstLayerConv2D(OpKernelContext* ctx,
      Tensor* input, Tensor* filter, Tensor* mean,
      Tensor* std, Tensor* bias, int row_stride, int col_stride,
      int pad_l, int pad_r, int pad_t, int pad_b,
      Tensor* output, TensorFormat tensor_format,
      MLUDataType dtype, int input_position, float input_scale,
      const std::vector<int>& filter_positions,
      const std::vector<float>& filter_scales);

  Status IntConv2D(OpKernelContext* ctx,
      Tensor* input, Tensor* filter, Tensor* bias,
      int row_stride, int col_stride,
      int pad_l, int pad_r, int pad_t, int pad_b,
      Tensor* output, TensorFormat tensor_format, MLUDataType dtype,
      int input_position, float input_scale,
      const std::vector<int>& filter_positions,
      const std::vector<float>& filter_scales,
      int dilation_rows, int dilation_cols);

  Status Concat(OpKernelContext* ctx, std::vector<Tensor*> inputs,
                Tensor* output, int axis) {
    ops::MLUConcatOpParam op_param(axis);
    return CommonOpImpl<ops::MLUConcat>(ctx, inputs, {output},
                                        static_cast<void*>(&op_param));
  }

  Status Interp(OpKernelContext* ctx,
      Tensor *input, int output_height, int output_width,
      bool align_corners, Tensor *output) {
     ops::MLUResizeOpsParam op_param(ops::MLUResizeOpsParam::RESIZE_INTERP,
         output_height, output_width, align_corners);
     return CommonOpImpl<ops::MLUResize>(ctx, {input}, {output},
         static_cast<void*>(&op_param));
  }

  Status CropAndResize(OpKernelContext* ctx,
      Tensor *input, Tensor* boxes, Tensor* index,
      int crop_height, int crop_width, float extrapolation_value,
      Tensor *output) {
    ops::MLUCropAndResizeOpParam p(crop_height, crop_width,
        extrapolation_value);
    return CommonOpImpl<ops::MLUCropAndResize>(ctx, {input, boxes, index},
        {output}, static_cast<void*>(&p));
  }

  Status NearestNeighbor(OpKernelContext* ctx, Tensor *input, int output_height,
      int output_width, bool align_corners, Tensor *output) {
     ops::MLUResizeOpsParam op_param(
         ops::MLUResizeOpsParam::RESIZE_NEAREST_NEIGHBOR,
         output_height, output_width, align_corners);
     return CommonOpImpl<ops::MLUResize>(ctx, {input}, {output},
         static_cast<void*>(&op_param));
  }

  Status Floor(OpKernelContext* ctx,
      Tensor *input, Tensor *output) {
        return CommonOpImpl<ops::MLUFloor>(ctx, {input}, {output},
         nullptr);
  }

  Status FloorDiv(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output) {
     return CommonOpImpl<ops::MLUFloorDiv>(ctx, {input1, input2}, {output});
  }

  Status Greater(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_GT;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status GreaterEqual(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_GE;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Less(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_LT;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status LessEqual(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_LE;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status Equal(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2, Tensor *output) {
    ops::MLUCwiseMethod method = ops::MLU_CWISE_EQ;
    return CommonOpImpl<ops::MLUDyadicCwise>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&method));
  }

  Status UnaryOp(OpKernelContext* ctx,
      string op, Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUUnary>(ctx, {input}, {output},
        static_cast<void*>(&op));
  }

  Status Rsqrt(OpKernelContext* ctx,
      Tensor *input, Tensor *output) {
    return CommonOpImpl<ops::MLURsqrt>(ctx, {input}, {output});
  }

  Status RsqrtBackprop(OpKernelContext* ctx,
      Tensor *x, Tensor *dy, Tensor *output) {
    return CommonOpImpl<ops::MLURsqrtBackprop>(ctx,
        {x, dy}, {output}, nullptr);
  }

  Status L2Loss(OpKernelContext* ctx,
      Tensor *x, Tensor *output) {
    return CommonOpImpl<ops::MLUL2Loss>(ctx,
        {x}, {output}, nullptr);
  }

  Status ListDiff(OpKernelContext* ctx,
      Tensor *input1, Tensor *input2,
      Tensor *output_data, Tensor *output_index) {
    return CommonOpImpl<ops::MLUListDiff>(ctx,
        {input1, input2}, {output_data, output_index}, nullptr);
  }

  Status Sqrt(OpKernelContext* ctx,
      Tensor *input, Tensor *output) {
    return CommonOpImpl<ops::MLUSqrt>(ctx, {input}, {output},
        nullptr);
  }

  Status Reshape(OpKernelContext* ctx, Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUReshape>(ctx, {input}, {output});
  }

  Status NonMaxSuppression(OpKernelContext* ctx,
      Tensor* boxes, Tensor* scores, int max_output_size_val,
      float iou_threshold_val, float scores_threshold_val,
      Tensor* output) {
    ops::MLUNonMaxSuppressionOpParam op_param(max_output_size_val,
        iou_threshold_val, scores_threshold_val);
    return CommonOpImpl<ops::MLUNonMaxSuppression>(ctx, {boxes, scores},
        {output}, static_cast<void*>(&op_param));
  }

  Status ReduceAll(OpKernelContext* ctx, Tensor* input, Tensor* output,
      const std::vector<int>& axes, bool unified_axes) {
    ops::MLUReduceOpsParam op_param(ops::MLUReduceOpsParam::REDUCE_ALL,
        axes, unified_axes);
    return CommonOpImpl<ops::MLUReduce>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status ReduceAny(OpKernelContext* ctx, Tensor* input, Tensor* output,
      const std::vector<int>& axes, bool unified_axes) {
    ops::MLUReduceOpsParam op_param(ops::MLUReduceOpsParam::REDUCE_ANY,
        axes, unified_axes);
    return CommonOpImpl<ops::MLUReduce>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status ReduceMax(OpKernelContext* ctx, Tensor* input, Tensor* output,
      const std::vector<int>& axes, bool unified_axes) {
    ops::MLUReduceOpsParam op_param(ops::MLUReduceOpsParam::REDUCE_MAX,
        axes, unified_axes);
    return CommonOpImpl<ops::MLUReduce>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status ReduceMean(OpKernelContext* ctx, Tensor* input, Tensor* output,
      const std::vector<int>& axes, bool unified_axes) {
    ops::MLUReduceOpsParam op_param(ops::MLUReduceOpsParam::REDUCE_MEAN,
        axes, unified_axes);
    return CommonOpImpl<ops::MLUReduce>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status ReduceSum(OpKernelContext* ctx, Tensor* input, Tensor* output,
      const std::vector<int>& axes, bool unified_axes) {
    ops::MLUReduceOpsParam op_param(ops::MLUReduceOpsParam::REDUCE_SUM,
        axes, unified_axes);
    return CommonOpImpl<ops::MLUReduce>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status ReduceProd(OpKernelContext* ctx, Tensor* input, Tensor* output,
      const std::vector<int>& axes, bool unified_axes) {
    ops::MLUReduceOpsParam op_param(ops::MLUReduceOpsParam::REDUCE_PROD,
        axes, unified_axes);
    return CommonOpImpl<ops::MLUReduce>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status SegmentReduction(OpKernelContext* ctx, Tensor* data,
                          Tensor* segment_ids, Tensor* output,
                          int num_segments, std::string task_name) {
    ops::MLUSegmentReductionOpParam op_param(num_segments, task_name);
    return CommonOpImpl<ops::MLUSegmentReductionOp>(ctx, {data, segment_ids},
        {output}, static_cast<void*>(&op_param));
    }

  Status Select(OpKernelContext* ctx, Tensor* input0, Tensor* input1,
                Tensor* input2, Tensor* output) {
    return CommonOpImpl<ops::MLUSelect>(ctx, {input0, input1, input2},
                                        {output});
  }

  Status Where(OpKernelContext* ctx,
      Tensor* input, Tensor* output0, Tensor* output1);

  Status Argmax(OpKernelContext* ctx,
      Tensor* input, int input_axis, Tensor* output) {
    return CommonOpImpl<ops::MLUArgmax>(ctx, {input}, {output},
        static_cast<void*>(&input_axis));
  }

  Status InvertPermutation(OpKernelContext* ctx,
      Tensor *input, Tensor *output) {
    return CommonOpImpl<ops::MLUInvertPermutation>(
        ctx, {input}, {output}, nullptr);
  }

  Status TransposePro(OpKernelContext* ctx,
      Tensor *input, Tensor *output,
      const std::vector<int> &perms, bool need_transpose) {
    return CommonOpImpl<ops::MLUTranspose>(ctx, {input}, {output},
        need_transpose ? static_cast<void*>(
          const_cast<std::vector<int>*>(&perms)) : nullptr);
  }

  Status ReverseSequence(OpKernelContext* ctx, Tensor* input,
      Tensor* output, int batch_dim, int seq_dim,
      const std::vector<int>& seq_lengths) {
    ops::MLUReverseSequenceOpParam op_param(batch_dim,
        seq_dim, seq_lengths);
    return CommonOpImpl<ops::MLUReverseSequence>(ctx,
        {input}, {output}, static_cast<void*>(&op_param));
  }

  Status ReverseV2(OpKernelContext* ctx,
      Tensor* input, const std::vector<int>& axis_vec,
      Tensor* output) {
    return CommonOpImpl<ops::MLUReverseV2>(ctx,
        {input}, {output},
        static_cast<void*>(const_cast<std::vector<int>*>(&axis_vec)));
  }

  Status TopK(OpKernelContext* ctx,
      Tensor* input, int k,
      Tensor* values_out, Tensor* indices_out, bool sorted) {
    ops::MLUTopKOpParam p(k, sorted);
    return CommonOpImpl<ops::MLUTopK>(ctx, {input},
        {values_out, indices_out}, static_cast<void*>(&p));
  }

  Status Space2Batch(OpKernelContext* ctx,
      Tensor* input, const std::vector<int>& block_shape,
      const std::vector<int>& paddings,
      const std::vector<int64>& padded, Tensor* output) {
    ops::MLUSpace2BatchOpParam p(block_shape, paddings, padded);
    return CommonOpImpl<ops::MLUSpace2Batch>(ctx,
        {input}, {output}, static_cast<void*>(&p));
  }

  Status Batch2Space(OpKernelContext* ctx,
      Tensor* input, int internal_block_dims,
      bool need_crop, std::vector<int> block_shape_vec,
      int64 output_batch, int block_dims, int64* temp_out,
      std::vector<int64> crops_vec, Tensor* output) {
    ops::MLUBatch2SpaceOpParam op_param(need_crop, internal_block_dims,
        block_shape_vec, output_batch, block_dims, temp_out, crops_vec);
    return CommonOpImpl<ops::MLUBatch2Space>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status DepthToSpace(OpKernelContext* ctx,
      Tensor* input, int block_size, Tensor* output){
    return CommonOpImpl<ops::MLUDepthToSpace>(ctx, {input}, {output},
        static_cast<void*>(&block_size));
  }

  Status Round(OpKernelContext* ctx, Tensor *input, Tensor *output) {
    return CommonOpImpl<ops::MLURound>(ctx, {input}, {output});
  }

  Status Snapshot(OpKernelContext* ctx, Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUSnapshot>(ctx, {input}, {output});
  }

  Status Mlp(OpKernelContext* ctx,
    Tensor* input, Tensor* filter, Tensor* bias, Tensor* output,
    const ops::MLUCNNOpsParam& op_param);

  Status QuantMatMul(OpKernelContext* ctx,
      Tensor* input, Tensor* weight,
      Tensor* input_quantized_param, Tensor* weight_quantized_param,
      Tensor* bias, Tensor* output,
      const ops::MLUQuantMatMulOpParam& op_param);

  Status Erf(OpKernelContext* ctx,
      Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUErf>(ctx, {input}, {output},
        nullptr);
  }

  Status SquaredDifference(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output) {
    return CommonOpImpl<ops::MLUSquaredDifference>(ctx,
        {input1, input2}, {output}, nullptr);
  }

  Status OneHot(OpKernelContext* ctx,
      Tensor *indices, Tensor *output, int depth,
      float on_value, float off_value, int axis){
    ops::MLUOneHotOpParam op_param(depth, on_value, off_value, axis);
    return CommonOpImpl<ops::MLUOneHot>(ctx, {indices}, {output},
        static_cast<void*>(&op_param));
  }

  Status Range(OpKernelContext* ctx,
      Tensor* start, Tensor* limit, Tensor*  delta,
      int size, Tensor* output) {
    return CommonOpImpl<ops::MLURange>(ctx, {start, limit, delta}, {output},
        static_cast<void*>(&size));
  }

  Status Sin(OpKernelContext* ctx,
      Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUSin>(ctx, {input}, {output}, nullptr);
  }

  Status Unique(OpKernelContext* ctx, Tensor* input,
      Tensor* output, Tensor* idx){
    return CommonOpImpl<ops::MLUUnique>(ctx, {input}, {output, idx}, nullptr);
	}

  Status BiasAdd(OpKernelContext* ctx,
      Tensor* input, Tensor* bias, Tensor* output, string data_format) {
    ops::MLUBiasAddOpParam op_param(data_format);
    return CommonOpImpl<ops::MLUBiasAdd>(ctx, {input, bias}, {output},
        static_cast<void*>(&op_param));
  }

  Status BiasAddGrad(OpKernelContext* ctx,
      Tensor* input, Tensor* output) {
    return CommonOpImpl<ops::MLUBiasAddGrad>(ctx, {input}, {output}, nullptr);
  }

  Status ReluBackprop(OpKernelContext* ctx,
      Tensor* dy, Tensor* x, Tensor* output) {
    return CommonOpImpl<ops::MLUReluBackprop>(ctx,
        {dy, x}, {output}, nullptr);
  }

  Status StridedSliceGrad(OpKernelContext* ctx,
      Tensor* dy, const std::vector<int>& begin,
      const std::vector<int>& end, const std::vector<int>& strides,
      Tensor* output) {
    ops::MLUStridedSliceOpParam op_param(begin, end, strides,
        /*is_identity*/false, /*strided_slice_shape*/{},
        /*final_shape*/{});
    return CommonOpImpl<ops::MLUStridedSliceGrad>(ctx, {dy}, {output},
        static_cast<void*>(&op_param));
  }

  Status RandomUniform(OpKernelContext* ctx,
      Tensor* output, int seed) {
    ops::MLURandomUniformOpParam op_param(seed);
    return CommonOpImpl<ops::MLURandomUniform>(ctx, {}, {output},
        static_cast<void*>(&op_param));
  }

  Status FusedBatchNormBackprop(OpKernelContext* ctx,
      Tensor *dy, Tensor *x, Tensor *scale, Tensor *mean, Tensor *var,
      Tensor *dx, Tensor *d_gamma, Tensor *d_beta,
      float epsilon) {
    return CommonOpImpl<ops::MLUFusedBatchNormBackprop>(ctx,
        {dy, x, scale, mean, var}, {dx, d_gamma, d_beta},
        static_cast<void*>(&epsilon));
  }

  Status Tile(OpKernelContext* ctx,
      Tensor* input, Tensor* output,
      int *multiple_array, int multiple_num) {
    ops::MLUTileOpParam op_param(multiple_array, multiple_num);
    return CommonOpImpl<ops::MLUTile>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status MatrixBandPart(OpKernelContext* ctx,
      Tensor* input, int num_lower, int num_upper, Tensor* output) {
    ops::MLUMatrixBandPartOpParam op_param(num_lower, num_upper);
    return CommonOpImpl<ops::MLUMatrixBandPart>(ctx, {input}, {output},
        static_cast<void*>(&op_param));
  }

  Status LeakyRelu(OpKernelContext* ctx,
      Tensor* features, float alpha, Tensor* output) {
    ops::MLULeakyReluOpParam op_param(alpha);
    return CommonOpImpl<ops::MLULeakyRelu>(ctx, {features},
        {output}, static_cast<void*>(&op_param));
  }


 public:
  MLUStreamExecutor *parent() const { return parent_; }

  // Set the StepStatsCollector member
  void set_stats_collector(StepStatsCollector* stats_collector);

  // todo(tonghengwen) : delete ctx in the future
  FusionOp* GetCachedFusionOp(OpKernelContext* ctx) {
    return parent_->GetCurrentFusionOp();
  }

  std::shared_ptr<ops::MLUBaseOpWrapper> GetCachedMLUOp(OpKernelContext* ctx);
  void AddMLUOpToCache(OpKernelContext* ctx,
      std::shared_ptr<ops::MLUBaseOpWrapper> op);


  // this is used only in Native mode (Zhu Ceng Mo shi)
  template <typename T>
  Status GetOrCreateMLUOp(std::shared_ptr<ops::MLUBaseOpWrapper> *op,
      OpKernelContext* ctx,
      std::vector<Tensor*> inputs,
      std::vector<Tensor*> outputs,
      void *param = nullptr,
      std::vector<MLUTensorType> inputs_mlu_tensors_type = {}) {
    std::shared_ptr<ops::MLUBaseOpWrapper> shared_op_ptr;
    shared_op_ptr = GetCachedMLUOp(ctx);
    if (shared_op_ptr) {
      MLULOG(1) << "Cache hit for " << ctx->op_kernel().name();
      *op = shared_op_ptr;
      return Status::OK();
    }

    std::vector<MLUTensor *> in_mlu_tensors;
    std::vector<MLUTensor *> out_mlu_tensors;

    TF_STATUS_CHECK(CreateMLUTensorFromTensorBatch(inputs, in_mlu_tensors,
        inputs_mlu_tensors_type));

    TF_STATUS_CHECK(CreateMLUTensorFromTensorBatch(outputs, out_mlu_tensors));

    shared_op_ptr = std::make_shared<T>(in_mlu_tensors,
        out_mlu_tensors, param);

    TF_PARAMS_CHECK(shared_op_ptr != nullptr, "Failed to create MLU OP");
    TF_PARAMS_CHECK(shared_op_ptr->Success(), "Failed to create MLU OP");

    shared_op_ptr->SetIOMLUTensorsOwner(true);

    // add names for debug
    shared_op_ptr->op_name_ = ctx->op_kernel().name();

    // compile the OP
    TF_STATUS_CHECK(shared_op_ptr->Compile(GetMLUCoreVersion(), GetCoreNum()));

    // add op the cache
    AddMLUOpToCache(ctx, shared_op_ptr);
    *op = shared_op_ptr;
    return Status::OK();
  }

  template <typename T>
  Status GetOrCreateMLUOpSpecial(std::shared_ptr<ops::MLUBaseOpWrapper> *op,
      OpKernelContext* ctx,
      std::vector<Tensor*> inputs,
      std::vector<Tensor*> outputs,
      void *param = nullptr,
      std::vector<MLUTensorType> inputs_mlu_tensors_type = {}) {
    std::shared_ptr<ops::MLUBaseOpWrapper> shared_op_ptr;
    shared_op_ptr = GetCachedMLUOp(ctx);
    if (shared_op_ptr) {
      MLULOG(1) << "Cache hit for " << ctx->op_kernel().name();
      *op = shared_op_ptr;
      return Status::OK();
    }

    shared_op_ptr = std::make_shared<T>(inputs,
        outputs, param, inputs_mlu_tensors_type);

    TF_PARAMS_CHECK(shared_op_ptr != nullptr, "Failed to create MLU OP");
    TF_PARAMS_CHECK(shared_op_ptr->Success(), "Failed to create MLU OP");

    shared_op_ptr->SetIOMLUTensorsOwner(true);

    // add names for debug
    shared_op_ptr->op_name_ = ctx->op_kernel().name();

    // compile the OP
    TF_STATUS_CHECK(shared_op_ptr->Compile(GetMLUCoreVersion(), GetCoreNum()));

    // add op the cache
    AddMLUOpToCache(ctx, shared_op_ptr);
    *op = shared_op_ptr;
    return Status::OK();
  }


  // Don't use OP cache if creating from MLU tensors
  template <typename T>
  Status CreateMLUOp(std::shared_ptr<ops::MLUBaseOpWrapper> *op,
      OpKernelContext* ctx,
      std::vector<MLUTensor*> inputs,
      std::vector<MLUTensor*> outputs,
      void *param) {
    std::shared_ptr<ops::MLUBaseOpWrapper> shared_op_ptr =
        std::make_shared<T>(inputs, outputs, param);

    TF_PARAMS_CHECK(shared_op_ptr != nullptr, "Failed to create MLU OP");
    TF_PARAMS_CHECK(shared_op_ptr->Success(), "Failed to create MLU OP");

    // add names for debug
    shared_op_ptr->op_name_ = ctx->op_kernel().name();
    *op = shared_op_ptr;
    return Status::OK();
  }

  /* Caution!
   * 1. All inputs must come from ctx->inputs
   * 2. All outputs must be the same order as ctx->mutable_outputs.
   * 3. All MLU tensors could be created by CreateMLUTensorFromTensor
   * DO NOT USE CommonOpImpl if your kernel doesn't apply.
   * */
  template<class OpType>
  Status CommonOpImpl(OpKernelContext* ctx, std::vector<Tensor*> inputs,
      std::vector<Tensor*> outputs, void* op_param = nullptr,
      std::vector<MLUTensorType> inputs_mlu_tensors_type = {}) {
    FusionOp* fusion_op = GetCachedFusionOp(ctx);

    auto process_fusion = [&]() {
      if (inputs_mlu_tensors_type.empty()) {
        inputs_mlu_tensors_type.resize(inputs.size(), MLU_TENSOR);
      }
      TF_PARAMS_CHECK(inputs.size() == inputs_mlu_tensors_type.size(),
          "inputs_mlu_tensors_type size must be equal to inputs size");

      // process inputs
      std::vector<MLUTensor*> mlu_inputs(inputs.size());
      for (int i = 0; i < inputs.size(); ++i) {
        TF_STATUS_CHECK(fusion_op->GetOrCreateInputMLUTensor(&mlu_inputs[i],
             ctx, inputs.at(i), inputs_mlu_tensors_type.at(i)));
      }

      // process outputs
      std::vector<MLUTensor*> mlu_outputs(outputs.size());
      for (int i = 0; i < outputs.size(); ++i) {
        TF_STATUS_CHECK(fusion_op->CreateAndInsertOutputMLUTensor(
            &mlu_outputs[i], ctx, outputs.at(i), i));
      }

      // Create Op
      std::shared_ptr<ops::MLUBaseOpWrapper> op;
      TF_STATUS_CHECK(CreateMLUOp<OpType>(&op, ctx,
            mlu_inputs, mlu_outputs, op_param));
      TF_STATUS_CHECK(fusion_op->Fuse(op->GetMLUBaseOp()));
      return Status::OK();
    };

    if (fusion_op) {
      return process_fusion();
    } else {
      std::shared_ptr<ops::MLUBaseOpWrapper> op;
      TF_STATUS_CHECK(GetOrCreateMLUOp<OpType>(&op, ctx,
            inputs, outputs, op_param, inputs_mlu_tensors_type));
      MLUCnrtQueue *queue = GetCnrtQueue();
      TF_STATUS_CHECK(op->ComputeUseTensor(inputs, outputs, queue));
    }
    return Status::OK();
  }

 /* Status PowerDifference(OpKernelContext* ctx,
      Tensor* input1, Tensor* input2, Tensor* output, int input3) {
    return CommonOpImpl<ops::MLUPowerDifference>(ctx,
        {input1, input2}, {output}, static_cast<void*>(&input3));
  }*/

 private:
  const unsigned long dev_;
  const int device_ordinal_;
  const MLUCoreVersion core_version_;
  const int core_num_;

  MLUStreamExecutor* const parent_;
  MLUCnrtQueue* cnrt_queue_ = nullptr;

  mutex collector_mu_;
  StepStatsCollector* stats_collector_ = nullptr GUARDED_BY(collector_mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(MLUStream);
};

inline MLUStream* AsMLUStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return static_cast<MLUStream*>(stream->implementation());
}

}  // namespace mlu
}  // namespace stream_executor
#endif  // TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_STREAM_H_
