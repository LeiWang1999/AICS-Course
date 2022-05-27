#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OP_S_B_C_MLU_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OP_S_B_C_MLU_H_
#if CAMBRICON_MLU
#include <string>
#include <iostream>
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/mlu_op_kernel.h"
#include "tensorflow/stream_executor/mlu/mlu_stream.h"

namespace tensorflow {
template <typename T>
class MLUSBCOp : public MLUOpKernel {
 public:
  explicit MLUSBCOp(OpKernelConstruction* ctx) :
          MLUOpKernel(ctx) {}

  void ComputeOnMLU(OpKernelContext* ctx) override {

    if (!ctx->ValidateInputsAreSameShape(this)) return;
    //auto* stream = ctx->op_device_context()->mlu_stream();
    //auto* mlustream_exec = ctx->op_device_context()->mlu_stream()->parent();
    se::mlu::MLUStream* stream = static_cast<se::mlu::MLUStream*>(
        ctx->op_device_context()->stream()->implementation());
    Tensor input = ctx->input(0);

    // TODO:参数检查与处理
    const Tensor& a = ctx->input(0);
    int batch_size = a.dim_size(0);

    /* string op_parameter = ctx->op_kernel().type_string() + "/" + input.shape().DebugString();

    MLU_OP_CHECK_UNSUPPORTED(mlustream_exec, op_parameter, ctx); */

    TensorShape shape = TensorShape(input.shape());

    //TODO:输出形状推断及输出内存分配
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));

    // 调用MLUStream层接口完成算子计算
    OP_REQUIRES_OK(ctx, stream->SBC(ctx, &input, output, batch_size));
  }
};

}  // namespace tensorflow


#endif  // CAMBRICON_MLU
#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OP_S_B_C_MLU_H_
