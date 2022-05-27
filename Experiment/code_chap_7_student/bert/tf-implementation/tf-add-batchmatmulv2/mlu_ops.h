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


#ifndef TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_API_OPS_MLU_OPS_H_
#define TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_API_OPS_MLU_OPS_H_

#include <functional>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/platform/mlu.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_baseop_wrapper.h"
#include "tensorflow/stream_executor/mlu/mlu_api/tf_mlu_intf.h"
#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/util.h"

namespace stream_executor {
namespace mlu {
namespace ops {


struct MLUBatch2SpaceOpParam {
  bool need_crop_;
  int internal_block_dims_;
  const std::vector<int>& block_shape_vec_;
  int64 output_batch_;
  int block_dims_;
  int64 *temp_out_;
  const std::vector<int64>& crops_vec_;
  MLUBatch2SpaceOpParam(bool need_crop,
                        int internal_block_dims,
                        const std::vector<int>& block_shape_vec,
                        int64 output_batch,
                        int block_dims,
                        int64 *temp_out,
                        const std::vector<int64>& crops_vec)
    : need_crop_(need_crop), internal_block_dims_(internal_block_dims),
      block_shape_vec_(block_shape_vec), output_batch_(output_batch),
      block_dims_(block_dims), temp_out_(temp_out), crops_vec_(crops_vec) {}
};

struct MLUBatchMatmulOpParam {
  bool adj_x_;
  bool adj_y_;
  MLUBatchMatmulOpParam(bool x, bool y) : adj_x_(x), adj_y_(y) {}
};

//TODO:补齐成员声明和构造函数
struct MLUBatchMatMulV2OpParam {
  ......
  MLUBatchMatMulV2OpParam(......)
    : ...... {}
};

struct MLUBiasAddOpParam {
  string data_format_;
  explicit MLUBiasAddOpParam(string data_format)
    : data_format_(data_format) {}
};

struct MLUConv2DBackpropInputOpParam {
  int sh_;
  int sw_;
  int dh_;
  int dw_;
  int pad_top_;
  int pad_bottom_;
  int pad_left_;
  int pad_right_;
  std::vector<int> output_shape_;
  MLUConv2DBackpropInputOpParam(int sh, int sw, int dh, int dw,
      int pad_top, int pad_bottom, int pad_left, int pad_right,
      std::vector<int> output_shape) :
      sh_(sh), sw_(sw), dh_(dh), dw_(dw), pad_top_(pad_top),
      pad_bottom_(pad_bottom), pad_left_(pad_left), pad_right_(pad_right),
      output_shape_(output_shape) {}
};

struct MLUBinaryOpParam {
  std::string op_name_;
  explicit MLUBinaryOpParam(std::string op_name) : op_name_(op_name) {}
};

struct MLUConcatOpParam {
  int axis_;
  explicit MLUConcatOpParam(int axis) : axis_(axis) {}
};

struct MLUCropAndResizeOpParam {
  int crop_height_;
  int crop_width_;
  float extrapolation_value_;
  MLUCropAndResizeOpParam(int crop_height, int crop_width,
      float extrapolation_value)
    : crop_height_(crop_height),
      crop_width_(crop_width),
      extrapolation_value_(extrapolation_value) {}
};

struct MLUIntDeconvOpParam {
  int stride_h_;
  int stride_w_;
  int hu_;
  int hd_;
  int wl_;
  int wr_;
  MLUDataType onchip_dtype_;  // INT8 or INT16
  float input_scale_;
  float filter_scale_;
  tensorflow::TensorFormat tensor_format_;
  MLUIntDeconvOpParam(int stride_h, int stride_w,
      int hu, int hd, int wl, int wr,
      MLUDataType onchip_dtype,
      float input_scale, float filter_scale,
      tensorflow::TensorFormat tensor_format)
    : stride_h_(stride_h), stride_w_(stride_w), hu_(hu), hd_(hd),
      wl_(wl), wr_(wr), onchip_dtype_(onchip_dtype),
      input_scale_(input_scale), filter_scale_(filter_scale),
      tensor_format_(tensor_format) {}
};

struct MLUGatherOpParam {
  int axis_;
  explicit MLUGatherOpParam(int axis) : axis_(axis) {}
};

struct MLUSplitOpParam {
  int split_dim_;
  explicit MLUSplitOpParam(int split_dim)
    : split_dim_(split_dim){}
};

struct MLULrnOpParam {
  MLULrnType lrn_type_;
  int local_size_;
  double alph_;
  double beta_;
  double k_;
  MLUDataType dtype_;
  int position_;
  float scale_;
  MLULrnOpParam(MLULrnType lrn_type, int local_size, double alph,
      double beta, double k, MLUDataType dtype, int position,
      float scale):lrn_type_(lrn_type), local_size_(local_size),
      alph_(alph), beta_(beta), k_(k), dtype_(dtype),
      position_(position), scale_(scale){}
};

struct MLUPad4OpParam{
  int padding_htop_;
  int padding_hbottom_;
  int padding_wleft_;
  int padding_wright_;
  float pad_value_;
  MLUPad4OpParam(int padding_htop, int padding_hbottom, int padding_wleft,
  int padding_wright, float pad_value):
  padding_htop_(padding_htop),
  padding_hbottom_(padding_hbottom),
  padding_wleft_(padding_wleft),
  padding_wright_(padding_wright),
  pad_value_(pad_value){}
};

struct MLUPoolingOpParam {
  MLUPoolMode pool_mode_;
  MLUPoolStrategyMode pool_strategy_mode_;
  bool real_;
  const std::vector<int>& kernel_size_;
  const std::vector<int>& strides_;
  const std::vector<std::pair<int, int>>& paddings_;
  const std::vector<int>& dilations_;
  tensorflow::TensorFormat tensor_format_;
  MLUPoolingOpParam(MLUPoolMode pool_mode,
      MLUPoolStrategyMode pool_strategy_mode,
      bool real,
      const std::vector<int>& kernel_size,
      const std::vector<int>& strides,
      const std::vector<std::pair<int, int>>& paddings,
      const std::vector<int>& dilations,
      tensorflow::TensorFormat tensor_format)
    : pool_mode_(pool_mode),
      pool_strategy_mode_(pool_strategy_mode),
      real_(real),
      kernel_size_(kernel_size),
      strides_(strides),
      paddings_(paddings),
      dilations_(dilations),
      tensor_format_(tensor_format){}
};

struct MLUQuantMatMulOpParam {
  bool transpose_a_;
  bool transpose_b_;
  MLUQuantMatMulOpParam(bool transpose_a, bool transpose_b)
    : transpose_a_(transpose_a), transpose_b_(transpose_b) {}
};

struct MLUTileOpParam {
  int *array_;
  int multiple_num_;
  MLUTileOpParam(int *array, int num) : array_(array), multiple_num_(num) {}
};

struct MLUTopKOpParam {
  int k_;
  bool sorted_;
  MLUTopKOpParam(int k, bool sorted) : k_(k), sorted_(sorted) {}
};

struct MLUDepthwiseConvOpParam {
  int stride_height_;
  int stride_width_;
  int pad_height_;
  int pad_width_;
  tensorflow::TensorFormat tensor_format_;
  MLUDepthwiseConvOpParam(int stride_height, int stride_width,
                          int pad_height, int pad_width,
                          tensorflow::TensorFormat tensor_format)
      : stride_height_(stride_height),
        stride_width_(stride_width),
        pad_height_(pad_height),
        pad_width_(pad_width),
        tensor_format_(tensor_format) {}
};

struct MLUConv2DOpParam {
   int stride_height_;
   int stride_width_;
   int dilation_height_;
   int dilation_width_;
   int pad_height_;
   int pad_width_;
   tensorflow::TensorFormat tensor_format_;
   MLUDataType onchip_dtype_;
   MLUConv2DOpParam(int stride_height, int stride_width,
       int dilation_height, int dilation_width,
       int pad_height, int pad_width,
       tensorflow::TensorFormat tensor_format,
       MLUDataType onchip_dtype)
       : stride_height_(stride_height),
         stride_width_(stride_width),
         dilation_height_(dilation_height),
         dilation_width_(dilation_width),
         pad_height_(pad_height),
         pad_width_(pad_width),
         tensor_format_(tensor_format),
         onchip_dtype_(onchip_dtype){}
};

struct MLUIntConv2DOpParam {
  int row_stride_;
  int col_stride_;
  int pad_l_;
  int pad_r_;
  int pad_t_;
  int pad_b_;
  int dilation_rows_;
  int dilation_cols_;
  MLUDataType onchip_dtype_;  // INT8 or INT16
  int input_position_;
  float input_scale_;
  std::vector<int> filter_positions_;
  std::vector<float> filter_scales_;
  tensorflow::TensorFormat tensor_format_;

  MLUIntConv2DOpParam(int row_stride, int col_stride,
                      int pad_l, int pad_r, int pad_t, int pad_b,
                      int dilation_rows, int dilation_cols,
                      MLUDataType onchip_dtype,
                      int input_position, float input_scale,
                      const std::vector<int>& filter_positions,
                      const std::vector<float>& filter_scales,
                      tensorflow::TensorFormat tensor_format)
        : row_stride_(row_stride),
          col_stride_(col_stride),
          pad_l_(pad_l),
          pad_r_(pad_r),
          pad_t_(pad_t),
          pad_b_(pad_b),
          dilation_rows_(dilation_rows),
          dilation_cols_(dilation_cols),
          onchip_dtype_(onchip_dtype),
          input_position_(input_position),
          input_scale_(input_scale),
          filter_positions_(filter_positions),
          filter_scales_(filter_scales),
          tensor_format_(tensor_format) {}
};

struct MLUCNNOpsParam {
  enum ComputeDataType {
    COMPUTE_FLOAT,
    COMPUTE_INT16,
    COMPUTE_INT8,
  };
  tensorflow::TensorFormat input_format;
  tensorflow::TensorFormat filter_format;
  std::vector<int> kernel_sizes;
  std::vector<int> strides;
  std::vector<int> paddings;
  std::vector<int> dilations;
  ComputeDataType compute_dtype;
  int input_position;
  float input_scale;
  int filter_position;
  float filter_scale;
  MLUCNNOpsParam() :
    input_format(tensorflow::FORMAT_NHWC),
    filter_format(tensorflow::FORMAT_HWCN),
    kernel_sizes({}), strides({}), paddings({}), dilations({}),
    compute_dtype(COMPUTE_FLOAT),
    input_position(DEFAULT_QUANT_POSITION), input_scale(DEFAULT_QUANT_SCALE),
    filter_position(DEFAULT_QUANT_POSITION), filter_scale(DEFAULT_QUANT_SCALE)
    {}
  static std::string ComputeDataTypeString(ComputeDataType dtype) {
    switch (dtype) {
      case COMPUTE_FLOAT:
        return "FLOAT";
      case COMPUTE_INT16:
        return "INT16";
      case COMPUTE_INT8:
        return "INT8";
      default:
        LOG(FATAL) << "Invalid data type: " << static_cast<int>(dtype);
        return "INVALID_DATA_TYPE";
    }
  }
  std::string DebugString() const {
    std::ostringstream oss;
    oss << "MLUCNNOpsParam"
        << " {input {format: " << tensorflow::ToString(input_format)
        << ", position: " << input_position
        << ", scale: " << input_scale << "}"
        << ", filter {format: " << tensorflow::ToString(filter_format)
        << ", position: " << filter_position
        << ", scale: " << filter_scale << "}"
        << ", compute_dtype: " << ComputeDataTypeString(compute_dtype);
    auto vector_string = [&oss](const std::vector<int>& vec,
        const std::string& vec_name) {
      if (!vec.empty()) {
        oss << ", " << vec_name << ": [";
        std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(oss, ","));
        oss << "]";
      }
    };
    vector_string(kernel_sizes, "kernel_sizes");
    vector_string(strides, "strides");
    vector_string(paddings, "paddings");
    vector_string(dilations, "dilations");
    oss << "}";
    return oss.str();
  }
};

struct MLUSpace2BatchOpParam {
  const std::vector<int>& block_shape_;  // w,h
  const std::vector<int>& paddings_;
  const std::vector<int64>& padded_;
  MLUSpace2BatchOpParam(const std::vector<int>& block_shape,
      const std::vector<int>& paddings, const std::vector<int64>& padded)
    : block_shape_(block_shape),
      paddings_(paddings),
      padded_(padded) {}
};

struct MLUStridedSliceOpParam {
  const std::vector<int>& begin_;
  const std::vector<int>& end_;
  const std::vector<int>& strides_;
  bool is_identity_;
  const std::vector<int>& strided_slice_shape_;
  const std::vector<int>& final_shape_;
  MLUStridedSliceOpParam(const std::vector<int>& begin,
      const std::vector<int>& end, const std::vector<int>& strides,
      bool is_identity, const std::vector<int>& strided_slice_shape,
      const std::vector<int>& final_shape)
    : begin_(begin), end_(end), strides_(strides),
      is_identity_(is_identity), strided_slice_shape_(strided_slice_shape),
      final_shape_(final_shape) {}
};

struct MLURandomUniformOpParam {
  int seed_;
  explicit MLURandomUniformOpParam(int seed) : seed_(seed) {}
};

struct MLUOneHotOpParam {
  int depth;
  float on_value;
  float off_value;
  int axis;
  MLUOneHotOpParam(int depth_in, float on_value_in, float off_value_in, int axis_in)
    : depth(depth_in), on_value(on_value_in), off_value(off_value_in),
    axis(axis_in) {}
};

struct MLUResizeOpsParam {
  enum ResizeMethod {
    RESIZE_INTERP = 1,
    RESIZE_NEAREST_NEIGHBOR = 2
  };
  ResizeMethod resize_method;
  int out_height;
  int out_width;
  bool align_corners;
  MLUResizeOpsParam(ResizeMethod method, int height, int width, bool align)
    : resize_method(method), out_height(height), out_width(width),
    align_corners(align) {}
};

struct MLUReduceOpsParam {
  enum ReduceMethod {
    REDUCE_MAX = 1,
    REDUCE_SUM = 2,
    REDUCE_MEAN = 3,
    REDUCE_PROD = 4,
    REDUCE_ALL = 5,
    REDUCE_ANY = 6
  };
  ReduceMethod reduce_method;
  std::vector<int> reduce_axes;
  bool unified_axes;  // sorted and unique
  MLUReduceOpsParam(
      ReduceMethod method, const std::vector<int>& axes, bool unified)
    : reduce_method(method), reduce_axes(axes), unified_axes(unified) {}
};

struct MLUSegmentReductionOpParam {
  int num_segments_;
  std::string task_name_;
  MLUSegmentReductionOpParam(int num_segment, std::string task_name)
    : num_segments_(num_segment), task_name_(task_name) {}
};

struct MLUNonMaxSuppressionOpParam {
  int max_output_size_val;
  float iou_threshold_val;
  float scores_threshold_val;
  MLUNonMaxSuppressionOpParam(
      int max_output_size_val, float iou_threshold_val, float scores_threshold_val)
    : max_output_size_val(max_output_size_val),
    iou_threshold_val(iou_threshold_val),
    scores_threshold_val(scores_threshold_val) {}
};

struct MLUMaxPoolingIndexParam {
  std::string pad_mode_;
  int window_height_;
  int window_width_;
  int stride_height_;
  int stride_width_;
  int padding_height_;
  int padding_width_;
  MLUMaxPoolingIndexParam(std::string pad_mode, int window_height,
       int window_width, int stride_height, int stride_width,
       int padding_height, int padding_width) : pad_mode_(pad_mode),
    window_height_(window_height), window_width_(window_width),
    stride_height_(stride_height), stride_width_(stride_width),
    padding_height_(padding_height), padding_width_(padding_width) {}
};

struct MLUPoolBackpropParam {
  std::string pad_mode_;
  int window_height_;
  int window_width_;
  int stride_height_;
  int stride_width_;
  int pad_left_;
  int pad_right_;
  int pad_up_;
  int pad_down_;
  MLUPoolBackpropParam(int window_height, int window_width,
                          int stride_height, int stride_width,
                          int pad_left, int pad_right,
                          int pad_up, int pad_down,
                          std::string pad_mode) : pad_mode_(pad_mode),
    window_height_(window_height), window_width_(window_width),
    stride_height_(stride_height), stride_width_(stride_width),
    pad_left_(pad_left), pad_right_(pad_right),
    pad_up_(pad_up), pad_down_(pad_down) {}
};

enum MLUCwiseMethod {
  MLU_CWISE_ADD = 1,
  MLU_CWISE_SUB = 2,
  MLU_CWISE_MUL = 3,
  MLU_CWISE_DIV = 4,
  MLU_CWISE_MAX = 5,
  MLU_CWISE_MIN = 6,
  MLU_CWISE_AND = 7,
  MLU_CWISE_OR = 8,
  MLU_CWISE_EQ = 9,
  MLU_CWISE_GE = 10,
  MLU_CWISE_GT = 11,
  MLU_CWISE_LE = 12,
  MLU_CWISE_LT = 13
};

struct MLUReverseSequenceOpParam {
  int batch_dim;
  int seq_dim;
  std::vector<int> seq_lengths;
  MLUReverseSequenceOpParam(int batch_axis, int seq_axis,
      const std::vector<int>& seq_lens)
    : batch_dim(batch_axis), seq_dim(seq_axis), seq_lengths(seq_lens) {}
};

struct MLUMatrixBandPartOpParam {
  int num_lower;
  int num_upper;
  MLUMatrixBandPartOpParam(int input_num_lower, int input_num_upper)
    : num_lower(input_num_lower), num_upper(input_num_upper) {}
};

struct MLULeakyReluOpParam {
  float alpha;
  explicit MLULeakyReluOpParam(float alpha) : alpha(alpha) {}
};

#define MLU_OP_CTOR_STATUS_CHECK(...)    \
  do {                                   \
    Status status = (__VA_ARGS__);       \
    if (!TF_PREDICT_TRUE(status.ok())) { \
      success_ = false;                  \
      LOG(ERROR) << status.ToString();   \
      return;                            \
    }                                    \
  } while (0)


#define DECLARE_OP_CLASS(NAME)                                                \
  class NAME : public MLUBaseOpWrapper {                                      \
   public:                                                                    \
    NAME(std::vector<MLUTensor *> &inputs, std::vector<MLUTensor *> &outputs, \
         void *param) : MLUBaseOpWrapper(inputs, outputs) {                   \
      MLU_OP_CTOR_STATUS_CHECK(CreateMLUOp(inputs, outputs, param));          \
      success_ = true;                                                        \
    }                                                                         \
    NAME(std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs,       \
         void *param, std::vector<MLUTensorType> mlu_tensors_type = {});      \
    tensorflow::Status Compute(const std::vector<void *> &inputs,             \
                   const std::vector<void *> &outputs,                        \
                   cnrtQueue_t queue) override;                               \
                                                                              \
   private:                                                                   \
   tensorflow::Status CreateMLUOp(std::vector<MLUTensor *> &inputs,           \
                       std::vector<MLUTensor *> &outputs, void *param);       \
  };


DECLARE_OP_CLASS(MLUAbs);
DECLARE_OP_CLASS(MLUAddn);
DECLARE_OP_CLASS(MLUBatchMatmul);
DECLARE_OP_CLASS(MLUBatch2Space);
DECLARE_OP_CLASS(MLUBiasAdd);
DECLARE_OP_CLASS(MLUBiasAddGrad);
DECLARE_OP_CLASS(MLUBinary);
DECLARE_OP_CLASS(MLUConcat);
DECLARE_OP_CLASS(MLUConv2DBackpropInput);
DECLARE_OP_CLASS(MLUCos);
DECLARE_OP_CLASS(MLUConv2D);
DECLARE_OP_CLASS(MLUElu);
DECLARE_OP_CLASS(MLUDepthwiseConv);
DECLARE_OP_CLASS(MLUFill);
DECLARE_OP_CLASS(MLUFirstLayerConv2D);
DECLARE_OP_CLASS(MLUFloor);
DECLARE_OP_CLASS(MLUFloorDiv);
DECLARE_OP_CLASS(MLUGather);
DECLARE_OP_CLASS(MLUIntConv2D);
DECLARE_OP_CLASS(MLUQuantConv2D);
DECLARE_OP_CLASS(MLUIsFinite);
DECLARE_OP_CLASS(MLULrn);
DECLARE_OP_CLASS(MLULog1p);
DECLARE_OP_CLASS(MLUNeg);
DECLARE_OP_CLASS(MLURange);
DECLARE_OP_CLASS(MLUReshape);
DECLARE_OP_CLASS(MLURsqrt);
DECLARE_OP_CLASS(MLUSelect);
DECLARE_OP_CLASS(MLUSelu);
DECLARE_OP_CLASS(MLUSoftsign);
DECLARE_OP_CLASS(MLUSplit);
DECLARE_OP_CLASS(MLUSquare);
DECLARE_OP_CLASS(MLUBatchNorm);
DECLARE_OP_CLASS(MLUCast);
DECLARE_OP_CLASS(MLUCropAndResize);
DECLARE_OP_CLASS(MLUCustomizedActive);
DECLARE_OP_CLASS(MLUIntDeconv);
DECLARE_OP_CLASS(MLUPad4);
DECLARE_OP_CLASS(MLUPooling);
DECLARE_OP_CLASS(MLUPow);
DECLARE_OP_CLASS(MLURandomUniform);
DECLARE_OP_CLASS(MLURealDiv);
DECLARE_OP_CLASS(MLUSpace2Batch);
DECLARE_OP_CLASS(MLUStridedSlice);
DECLARE_OP_CLASS(MLUStridedSliceGrad);
DECLARE_OP_CLASS(MLUTile);
DECLARE_OP_CLASS(MLUTopK);
DECLARE_OP_CLASS(MLUZerosLike);
DECLARE_OP_CLASS(MLUResize);
DECLARE_OP_CLASS(MLUActive);
DECLARE_OP_CLASS(MLUCrop);
DECLARE_OP_CLASS(MLUReduce);
DECLARE_OP_CLASS(MLUExp);
DECLARE_OP_CLASS(MLUErf);
DECLARE_OP_CLASS(MLUQuantify);
DECLARE_OP_CLASS(MLUOneHot);
DECLARE_OP_CLASS(MLUSnapshot);
DECLARE_OP_CLASS(MLUSoftmax);
DECLARE_OP_CLASS(MLUSoftmaxXentWithLogits);
DECLARE_OP_CLASS(MLUSqrt);
DECLARE_OP_CLASS(MLULogSoftmax);
DECLARE_OP_CLASS(MLUPack);
DECLARE_OP_CLASS(MLUSquaredDifference);
DECLARE_OP_CLASS(MLUTranspose);
DECLARE_OP_CLASS(MLUInvertPermutation);
DECLARE_OP_CLASS(MLUDyadicCwise);
DECLARE_OP_CLASS(MLUUnpack);
DECLARE_OP_CLASS(MLUArgmax);
DECLARE_OP_CLASS(MLUMlp);
DECLARE_OP_CLASS(MLUNonMaxSuppression);
DECLARE_OP_CLASS(MLUSin);
DECLARE_OP_CLASS(MLURound);
DECLARE_OP_CLASS(MLURsqrtBackprop);
DECLARE_OP_CLASS(MLUMatrixBandPart);
DECLARE_OP_CLASS(MLUL2Loss);
DECLARE_OP_CLASS(MLUListDiff);
DECLARE_OP_CLASS(MLUReluBackprop);
DECLARE_OP_CLASS(MLUMaxPoolingIndex);
DECLARE_OP_CLASS(MLUMaxPoolBackprop);
DECLARE_OP_CLASS(MLUAvgPoolBackprop);
DECLARE_OP_CLASS(MLUConv2DBackpropFilter);
DECLARE_OP_CLASS(MLUFusedBatchNormBackprop);
DECLARE_OP_CLASS(MLUReverseSequence);
DECLARE_OP_CLASS(MLUReverseV2);
DECLARE_OP_CLASS(MLUUnary);
DECLARE_OP_CLASS(MLUUnique);
DECLARE_OP_CLASS(MLUQuantMatMul);
DECLARE_OP_CLASS(MLUScatterNd);
DECLARE_OP_CLASS(MLUSegmentReductionOp);
DECLARE_OP_CLASS(MLUFusedBatchNorm);
DECLARE_OP_CLASS(MLUDepthToSpace);
DECLARE_OP_CLASS(MLUBertSquad);
DECLARE_OP_CLASS(MLULeakyRelu);
DECLARE_OP_CLASS(MLUBatchMatMulV2);
}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor
#endif  // TENSORFLOW_STREAM_EXECUTOR_MLU_MLU_API_OPS_MLU_OPS_H_
