// Copyright [2018] <Cambricon>
#ifndef TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_V2_OP_MLU_H_
#define TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_V2_OP_MLU_H_
#if CAMBRICON_MLU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/mlu_op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/stream_executor/mlu/mlu_stream.h"

namespace tensorflow {

template <typename T>
class MLUBatchMatMulV2 : public MLUOpKernel {
 public:
  explicit MLUBatchMatMulV2(OpKernelConstruction* context)
    : MLUOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));

    // position and scale
    int input1_position, input2_position;
    float input1_scale, input2_scale;
    // input1_position
    if (context->GetAttr("input1_position", &input1_position).ok()) {
      pos_0_ = input1_position;
    } else {
      pos_0_ = 0;
    }
    // input1_scale
    if (context->GetAttr("input1_scale", &input1_scale).ok()) {
      scale_0_ = input1_scale;
    } else {
      scale_0_ = 1;
    }
    // TODO:补齐input2_position
    if (context->GetAttr("input2_position", &input2_position).ok()) {
      ......
    } else {
      ......
    }
    // TODO:补齐input2_scale
    if (context->GetAttr("input2_scale", &input2_scale).ok()) {
      ......
    } else {
      ......
    }
  }

  void ComputeOnMLU(OpKernelContext* ctx) override {
    se::mlu::MLUStream* stream = static_cast<se::mlu::MLUStream*>(
      ctx->op_device_context()->stream()->implementation());
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.dims() == in1.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        in0.shape().DebugString(), " vs. ",
                                        in1.shape().DebugString()));

    const int ndims = in0.dims();
    OP_REQUIRES(
        ctx, in0.dims() >= 2,
        errors::InvalidArgument("In[0] ndims must be >= 2: ", in0.dims()));
    OP_REQUIRES(
        ctx, in1.dims() >= 2,
        errors::InvalidArgument("In[1] ndims must be >= 2: ", in1.dims()));

    TensorShape out_shape;
    for (int i = 0; i < ndims - 2; ++i) {
      OP_REQUIRES(ctx, in0.dim_size(i) == in1.dim_size(i),
                  errors::InvalidArgument(
                      "In[0].dim(", i, ") and In[1].dim(", i,
                      ") must be the same: ", in0.shape().DebugString(), " vs ",
                      in1.shape().DebugString()));
      out_shape.AddDim(in0.dim_size(i));
    }

    auto d0 = in0.dim_size(ndims - 2);
    auto d1 = in0.dim_size(ndims - 1);
    auto d2 = in1.dim_size(ndims - 2);
    auto d3 = in1.dim_size(ndims - 1);

    if (adj_x_) std::swap(d0, d1);
    if (adj_y_) std::swap(d2, d3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", adj_x_, " ", adj_y_));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
   
    trans_flag_ = false;
    if (adj_x_ == false && adj_y_ == false) trans_flag_ = true;
    Tensor* in0_tensor = const_cast<Tensor*>(&in0);
    Tensor* in1_tensor = const_cast<Tensor*>(&in1);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    dim_0_ = in0.dim_size(0);
    dim_1_ = in0.dim_size(1);
    m_ = d0;
    n_ = d3;
    k_ = d1;

    OP_REQUIRES_OK(ctx, stream->BatchMatMulV2(ctx, in0_tensor, in1_tensor,
        scale_0_, pos_0_, scale_1_, pos_1_, trans_flag_, dim_0_, dim_1_, m_, n_, k_,
        output));
  };

 private:
  bool adj_x_;
  bool adj_y_;
  float scale_0_;
  int pos_0_;
  float scale_1_;
  int pos_1_;
  bool trans_flag_;  
  int dim_0_;
  int dim_1_;
  int m_;
  int n_;
  int k_;
};
}  // namespace tensorflow
#endif  // CAMBRICON_MLU
#endif  // TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_V2_OP_MLU_H_
