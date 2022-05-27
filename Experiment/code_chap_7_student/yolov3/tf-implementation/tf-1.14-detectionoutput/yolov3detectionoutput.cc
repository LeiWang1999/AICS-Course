#if CAMBRICON_MLU
#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"

namespace stream_executor {
namespace mlu {
namespace ops {


Status MLUYolov3DetectionOutput::CreateMLUOp(std::vector<MLUTensor*> &inputs, \
    std::vector<MLUTensor*> &outputs, void *param) {
    //TODO:补齐create函数实现
    return Status::OK();
}

Status MLUYolov3DetectionOutput::Compute(const std::vector<void *> &inputs,
    const std::vector<void *> &outputs, cnrtQueue_t queue) {
  
    //TODO:补齐compute函数实现
    return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor
#endif  // CAMBRICON_MLU
