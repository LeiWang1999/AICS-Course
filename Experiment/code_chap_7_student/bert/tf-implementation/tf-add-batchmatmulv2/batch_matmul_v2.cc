/*Copyright 2018 Cambricon*/

#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/tf_mlu_intf.h"

namespace stream_executor {
namespace mlu {
namespace ops {


Status MLUBatchMatMulV2::CreateMLUOp(std::vector<MLUTensor *> &inputs,
                                     std::vector<MLUTensor *> &outputs,
                                     void *param) {
  TF_PARAMS_CHECK(inputs.size() > 1, "Missing input");
  TF_PARAMS_CHECK(outputs.size() > 0, "Missing output");

  MLUTensor *in0 = inputs.at(0);
  MLUTensor *in1 = inputs.at(1);
  MLUTensor *output = outputs.at(0);

  MLULOG(3) << "CreateBatchMatMulV2Op, input1: "
            << lib::MLUTensorUtil(in0).DebugString()
            << ", input2: " << lib::MLUTensorUtil(in1).DebugString()
            << ", output: " << lib::MLUTensorUtil(output).DebugString();

  MLUBatchMatMulV2OpParam *op_param = static_cast<MLUBatchMatMulV2OpParam *>(param);

  float scale_0 = op_param->scale_0_;
  int pos_0 = op_param->pos_0_;
  float scale_1 = op_param->scale_1_;
  int pos_1 = op_param->pos_1_;
  bool trans_flag = op_param->trans_flag_;
  int dim_0 = op_param->dim_0_;
  int dim_1 = op_param->dim_1_;
  int m = op_param->m_;
  int n = op_param->n_;
  int k = op_param->k_;
  cnmlCoreVersion_t core_version = CNML_MLU270;

//
  bool* flags = (bool*)malloc(1 * sizeof(bool));
  flags[0] = trans_flag;
  extra_ = static_cast<void*>(flags);
  //LOG(INFO) << "trans_flag: "
  //          << trans_flag;
  if (trans_flag) {
    lib::MLUTensorUtil input_util(in1);
    tensorflow::TensorShape input_shape_t = {dim_0, dim_1, n, k};
    std::vector<int> input_shape_t_vec(input_shape_t.dims());
    for (int i = 0; i < input_shape_t.dims(); ++i) {
      input_shape_t_vec[i] = input_shape_t.dim_size(i);
    }
    MLUTensor* input_transpose = nullptr;
    TF_STATUS_CHECK(lib::CreateMLUTensor(&input_transpose, MLU_TENSOR,
          input_util.dtype(), input_shape_t_vec));
    //LOG(INFO) << "CreateTransposeProOp"
    //          << ", input: " << input_util.DebugString()
    //          << ", output: "
    //          << lib::MLUTensorUtil(input_transpose).DebugString();
    MLUBaseOp* input_transpose_op = nullptr;
    std::vector<int> input_perms = {0, 1, 3, 2};
    TF_STATUS_CHECK(lib::CreateTransposeProOp(&input_transpose_op, in1,
          input_transpose, input_perms.data(), input_perms.size()));
    base_ops_.push_back(input_transpose_op);
    intmd_tensors_.push_back(input_transpose);
    in1 = input_transpose;
    input_util.Update(in1);
    //input_shape = {dim_0, dim_1, n, k};
  }else {
    base_ops_.push_back(nullptr);
    intmd_tensors_.push_back(nullptr);
  } 
//

  cnmlPluginBatchMatMulV2OpParam_t bm_param;
  TF_CNML_CHECK(cnmlCreatePluginBatchMatMulV2OpParam(&bm_param, scale_0, pos_0,
        scale_1, pos_1, dim_0, dim_1, m, n, k, core_version));

  MLUBaseOp *batch_matmul_op_ptr = nullptr;

  //TODO:补齐下面的操作
  TF_STATUS_CHECK(lib::CreateBatchMatMulV2Op(......));

  base_ops_.push_back(batch_matmul_op_ptr);

  return Status::OK();
}

Status MLUBatchMatMulV2::Compute(const std::vector<void *> &inputs,
                               const std::vector<void *> &outputs,
                               cnrtQueue_t queue) {
  MLULOG(3) << "ComputeMLUBatchMatMulV2";

  int input_num = inputs.size();
  int output_num = outputs.size();

  void* real_inputs[input_num];
  void* real_outputs[output_num];


  for (int i = 0; i < input_num; ++i) {
    real_inputs[i] = inputs.at(i);
  }

  //TODO:补齐下面的操作
  for (int i = 0; i < output_num; ++i) {
    .......
  }

//
  bool* flags = (bool*)extra_;
  std::vector<void*> intmd_addrs;
  if (flags[0]) {
  // transpose for in1
    size_t input_size =
      lib::MLUTensorUtil::GetTensorDataSize(intmd_tensors_[0]);
    void* input_transpose_ = nullptr;
    cnrtMalloc(&input_transpose_, input_size);
    TF_STATUS_CHECK(lib::ComputeTransposeProOp(base_ops_[0],
          queue, real_inputs[1], input_transpose_));
    intmd_addrs.push_back(input_transpose_);
    real_inputs[1] = input_transpose_;
  }

//

  //TODO:补齐下面的操作
  TF_STATUS_CHECK(lib::ComputeBatchMatMulV2Op(......));

  TF_CNRT_CHECK(cnrtSyncQueue(queue));

  return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor
