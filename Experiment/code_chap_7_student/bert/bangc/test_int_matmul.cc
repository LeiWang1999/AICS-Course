// Copyright (c) 2019 Smarsu. All Rights Reserved.

#include <iostream>
#include <vector>
#include <cnrt.h>
#include <cnml.h>
#include <cnplugin.h>

#include "utils.h"

extern "C" {
  void int_matmul_entry();
}

std::vector<float> CpuMatmul(const std::vector<float> &left,
                             const std::vector<float> &right,
                             int dim_0,
                             int dim_1,
                             int m,
                             int n,
                             int k) {
  return Matmul(dim_0, dim_1, left, right, m, n, k);
}

std::vector<float> MluMatmul(const std::vector<float> &left_host_fp32,
                             const std::vector<float> &right_host_fp32,
                             int dim_0,
                             int dim_1,
                             int m,
                             int n,
                             int k,
                             int core_num) {
  if (k > 3968) {
    std::cout << "Only support k <= 3968 now";
    abort();
  }
  if (core_num != 4 && core_num != 16) {
    std::cout << "Only support core_num == 16 || core_num == 4 now";
    abort();
  }

  void *left_ptr = nullptr;
  void *right_ptr = nullptr;

  float left_scale = 1;
  int left_pos = 0;
  std::vector<int16_t> left_host_int16;
  float right_scale = 1;
  int right_pos = 0;
  std::vector<int16_t> right_host_int16;
  left_host_int16 = Quant<int16_t, float>(left_host_fp32, left_scale, left_pos);
  right_host_int16 = Quant<int16_t, float>(right_host_fp32, right_scale, right_pos);
  left_ptr = reinterpret_cast<void *>(left_host_int16.data());
  right_ptr = reinterpret_cast<void *>(right_host_int16.data());
  std::vector<half> left_host_half = ConvertToHalf<half>(left_host_fp32);
  std::vector<half> right_host_half = ConvertToHalf<half>(right_host_fp32);


  cnmlCoreVersion_t core_version = CNML_MLU270;
  cnmlPluginBatchMatMulV2OpParam_t param;
  cnmlCreatePluginBatchMatMulV2OpParam(&param,
          left_scale,
          left_pos,
          right_scale,
          right_pos,
          dim_0,
          dim_1,
          m,
          n,
          k,
          core_version);

  cnmlTensor_t *mlu_input_tensor = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t) * 2);
  cnmlTensor_t *mlu_output_tensor = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t));
  cnmlCreateTensor(&mlu_input_tensor[0], CNML_TENSOR, CNML_DATA_FLOAT16, dim_0, dim_1, m, k);
  cnmlCreateTensor(&mlu_input_tensor[1], CNML_TENSOR, CNML_DATA_FLOAT16, dim_0, dim_1, n, k);
  cnmlCreateTensor(&mlu_output_tensor[0], CNML_TENSOR, CNML_DATA_FLOAT16, dim_0, dim_1, m, n);

  cnmlBaseOp_t op;
  cnmlCreatePluginBatchMatMulV2Op(&op, param, mlu_input_tensor, mlu_output_tensor);
  cnmlSetOperationComputingLayout(op, CNML_NHWC);
  cnmlCompileBaseOp(op, CNML_MLU270, 1);


  // prepare input
  void ** mlu_input = (void**)malloc(sizeof(void *) * 2);
  void ** mlu_output = (void**)malloc(sizeof(void *));
  cnrtMalloc(&(mlu_input[0]), dim_0*dim_1*m*k*sizeof(half));
  cnrtMalloc(&(mlu_input[1]), dim_0*dim_1*n*k*sizeof(half));
  cnrtMalloc(&(mlu_output[0]), dim_0*dim_1*m*n*sizeof(half));
  cnrtMemcpy(mlu_input[0], left_host_half.data(), dim_0*dim_1*m*k*sizeof(half),CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(mlu_input[1], right_host_half.data(), dim_0*dim_1*n*k*sizeof(half),CNRT_MEM_TRANS_DIR_HOST2DEV);

  // forward mlu
  cnrtQueue_t queue;
  cnrtCreateQueue(&queue);
  cnmlComputePluginBatchMatMulV2OpForward(op, mlu_input, 2, mlu_output, 1, queue);
  cnrtSyncQueue(queue);
  cnrtDestroyQueue(queue);

  // get Result
  std::vector<half> dst(dim_0*dim_1*m*n);
  cnrtMemcpy(dst.data(), mlu_output[0], dim_0*dim_1*m*n*sizeof(half),CNRT_MEM_TRANS_DIR_DEV2HOST);


  return ConvertToFloat<float>(dst);
}

void TestMatmul(int dim_0,
                int dim_1,
                int m,
                int n,
                int k,
                int core_num) {
  std::cout << "TEST M: " << m << " N: " << n << " K: " << k << " core_num: " << core_num << std::endl; 

  std::vector<float> left = Rand<float>(static_cast<int64_t>(m) * k *dim_0 *dim_1, -1, 1);
  std::vector<float> right = Rand<float>(static_cast<int64_t>(n) * k *dim_0 *dim_1, -1, 1);
  std::vector<float> mlu_result = MluMatmul(left, right, dim_0, dim_1, m, n, k, core_num);
  std::vector<float> cpu_result = CpuMatmul(left, right, dim_0, dim_1, m, n, k);
  CheckResult(mlu_result.data(), cpu_result.data(), cpu_result.size(), 0.5);

  std::cout << "PASS M: " << m << " N: " << n << " K: " << k << " core_num: " << core_num << std::endl; 
}

int main(int argv, char *args[]) {
  SetDevice(0);

  if (argv == 6) {
    int dim_0 = std::atoi(args[1]);
    int dim_1 = std::atoi(args[2]);
    int m = std::atoi(args[3]);
    int n = std::atoi(args[4]);
    int k = std::atoi(args[5]);

    TestMatmul(dim_0, dim_1, m, n, k, 16);
    TestMatmul(dim_0, dim_1, m, n, k, 16);
    TestMatmul(dim_0, dim_1, m, n, k, 4);
    TestMatmul(dim_0, dim_1, m, n, k, 4);

    return 0;
  }
  
  // M: 1 N: 129 K: 3967
  std::vector<int> candidate_m = {1, 2, 9, 10, 63, 127, 128,
                                  255, 256, 512, 513};
  std::vector<int> candidate_n = {1, 2, 9, 10, 63, 127, 128,
                                  255, 256, 512, 513};
  std::vector<int> candidate_k = {1, 2, 9, 10, 63, 127, 128,
                                  255, 256, 512, 513};
  std::vector<DataType> candidate_it = {kInt16};
  std::vector<DataType> candidate_ft = {kFloat32};
  std::vector<int> candidate_core_num = {4, 16};

  for (auto m : candidate_m) {
    for (auto n : candidate_n) {
      for (auto k : candidate_k) {
        for (auto it : candidate_it) {
          for (auto ft : candidate_ft) {
            for (auto core_num : candidate_core_num) {
              if (byteof(it) * static_cast<int64_t>(m) * k > 2147483648 || 
                  byteof(it) * static_cast<int64_t>(n) * k > 2147483648 || 
                  byteof(ft) * static_cast<int64_t>(m) * n > 2147483648) {
                continue;
              }
              TestMatmul(1, 1,m, n, k, core_num);
            }
          }
        }
      }
    }
  }
}
