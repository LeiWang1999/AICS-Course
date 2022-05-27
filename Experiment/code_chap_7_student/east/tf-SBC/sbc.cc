/*Copyright 2018 Cambricon*/
#if CAMBRICON_MLU

#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/tf_mlu_intf.h"

namespace stream_executor {
namespace mlu {
namespace ops {


Status MLUSBC::CreateMLUOp(std::vector<MLUTensor *> &inputs,
                            std::vector<MLUTensor *> &outputs, void *param) {
  //TODO:补齐create实现
  TF_PARAMS_CHECK(inputs.size() > 0, "Missing input");
  TF_PARAMS_CHECK(outputs.size() > 0, "Missing output");

  MLUBaseOp *op_ptr = nullptr;
  MLUTensor *input = inputs.at(0);
  MLUTensor *output = outputs.at(0);

  int batch_num_ = *((int *)param);

  MLULOG(3) << "CreateSBCOp"
            << ", input: " << lib::MLUTensorUtil(input).DebugString()
            << ", output: " << lib::MLUTensorUtil(output).DebugString();
  
  TF_STATUS_CHECK(lib::CreateSBCOp(&op_ptr, input, output, batch_num_));

  base_ops_.push_back(op_ptr);

  return Status::OK();
}

Status MLUSBC::Compute(const std::vector<void *> &inputs,
                        const std::vector<void *> &outputs, cnrtQueue_t queue) {
  //TODO:补齐compute实现
  void *input = inputs.at(0);
  void *output = outputs.at(0);

  TF_STATUS_CHECK(lib::ComputeSBCOp(base_ops_.at(0), input, output, queue));

  TF_CNRT_CHECK(cnrtSyncQueue(queue));

  return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor

#endif  // CAMBRICON_MLU
