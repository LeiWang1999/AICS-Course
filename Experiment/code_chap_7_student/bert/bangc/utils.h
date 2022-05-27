// Copyright (c) 2019 Smarsu. All Rights Reserved.

#pragma once

#define ALIGN_UP(a, b) (((a) + (b) - 1) / (b) * (b))
#define ALIGN_DN(a, b) ((a) / (b) * (b))
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#define DIV_DN(a, b) ((a) / (b))

#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define ABS(a) (((a) > 0) ? (a) : (-(a)))

#define INIFITE 0x7F800000

enum DataType {
  kInvalid,
  kFloat32,
  kFloat16,
  kUint8,
  kInt8,
  kInt16,
  kInt32,
};

enum TopkSplitStrategy {
  kAuto,
  kSplitN,
  kSplitC,
};

enum ColorType {
  kGray,
  kRGB,
  kBGR,
  kRGBA,
};

#include <random>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <cnrt.h>
extern "C" {
#include <openblas/cblas.h>
}

#define CALL_CNRT(x) x

using half = uint16_t;

void SetDevice(int device) {
  CALL_CNRT(cnrtInit(0));
  cnrtDev_t dev;
  CALL_CNRT(cnrtGetDeviceHandle(&dev, device));
  CALL_CNRT(cnrtSetCurrentDevice(dev));
}

/* ms
 */
double time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (static_cast<double>(ts.tv_sec) * 1000 + 
          static_cast<double>(ts.tv_nsec) / 1000000);
}


size_t byteof(DataType data_type) {
  switch (data_type) {
    case kInvalid:
      return 0;

    case kFloat32:
      return 4;

    case kFloat16:
      return 2;

    case kUint8:
      return 1;

    case kInt8:
      return 1;

    case kInt16:
      return 2;
    
    case kInt32:
      return 4;

    default:
      abort();
  }
}

template <typename T>
std::vector<T> ConvertToHalf(const std::vector<float> &src) {
  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    half tmp;
    cnrtConvertFloatToHalf(&tmp, src[i]);
    dst[i] = tmp;
  }
  return std::move(dst);
}


template <typename T>
std::vector<T> ConvertToFloat(const std::vector<half> &src) {
  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    float tmp;
    cnrtConvertHalfToFloat(&tmp, src[i]);
    dst[i] = tmp;
  }
  return std::move(dst);
}


template <typename T>
std::vector<T> Rand(size_t num, T min = -1, T max = 1) {
  static std::default_random_engine e;
  std::uniform_real_distribution<T> u(min, max);

  std::vector<T> data(num);
  for (size_t i = 0; i < num; ++i) {
    data[i] = u(e);
  }
  return data;
}

template <typename T>
T abs(T v) {
  return (v >= 0) ? v : -v;
}

template <typename T, typename F>
std::vector<T> Quant(const std::vector<F> &src, float &scale, int &pos) {
  F max_v = 0;
  for (auto v : src) {
    max_v = std::max(max_v, abs(v));
  }

  scale = (std::pow(2, sizeof(T) * 8 - 1) - 1) / max_v;
  pos = std::log2(scale);
  scale = scale / std::pow(2, pos);
  pos = -pos;

  std::vector<T> dst(src.size());
  for (size_t i = 0; i < dst.size(); ++i) {
    dst[i] = src[i] * scale / std::pow(2, pos);
  }

  return std::move(dst);
}

std::vector<float> Matmul(int dim_0,
                          int dim_1,
                          const std::vector<float> &left,
                          const std::vector<float> &right,
                          int64_t m,
                          int64_t n,
                          int64_t k) {
  std::vector<float> dst(dim_0 *dim_1* m * n);
  for (int cur_dim_0 = 0; cur_dim_0 < dim_0; cur_dim_0++) {
      for (int cur_dim_1 = 0;cur_dim_1 < dim_1; cur_dim_1++) {
          int offset = (cur_dim_0*dim_1+cur_dim_1);

          const float *A = (float*)left.data()+offset*m*k;
          const float *B = (float*)right.data()+offset*n*k;
          float *C = const_cast<float *>(dst.data())+offset*m*n;
          int64_t M = m;
          int64_t K = k;
          int64_t N = n;
          const CBLAS_ORDER Order = CblasRowMajor;
          const CBLAS_TRANSPOSE TransA = CblasNoTrans;
          const CBLAS_TRANSPOSE TransB = CblasTrans;
          float alpha = 1;
          float beta = 0;
          int64_t lda = K;
          int64_t ldb = K;
          int64_t ldc = N;
          cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      }
  }

  return std::move(dst);
}


template <typename T>
void CheckResult(T *mlu, T *host, size_t size, double abort_thr = 0, bool show_result = false) {
  auto show = [&](int64_t idx) {
    if (idx < 0 || idx >= static_cast<int64_t>(size)) {
      return;
    }
    std::cout << "i: " << idx
              << " mlu: " << mlu[idx]
              << " host: " << host[idx]
              << " dis: " << abs(mlu[idx] - host[idx])
              << " dis percent: " << abs(mlu[idx] - host[idx]) / (abs(host[idx]) + 1e-12) << std::endl;
    if (abort_thr == -1) {
      return;
    }
    if (abs(mlu[idx] - host[idx]) > abort_thr && abs(mlu[idx] - host[idx]) / (abs(host[idx]) + 1e-12) > abort_thr) {
      abort();
    }
  };

  T max_dis_percent = 0;
  int64_t max_dis_percent_idx = -1;

  T max_dis = 0;
  int64_t max_dis_idx = -1;

  double sum = 0.f;
  for (size_t i = 0; i < size; ++i) {
    T dis_percent = abs(mlu[i] - host[i]) / (abs(host[i]) + 1e-12);
    if (dis_percent > max_dis_percent) {
      max_dis_percent = dis_percent;
      max_dis_percent_idx = static_cast<int64_t>(i);
    }

    T dis = abs(mlu[i] - host[i]);
    if (dis > max_dis) {
      max_dis = dis;
      max_dis_idx = static_cast<int64_t>(i);
    }

    sum += dis_percent;

    if (show_result) {
      int idx = i;
      std::cout << "i: " << idx
                << " mlu: " << mlu[idx]
                << " host: " << host[idx]
                << " dis: " << abs(mlu[idx] - host[idx])
                << " dis percent: " << abs(mlu[idx] - host[idx]) / (abs(host[idx]) + 1e-12) << std::endl;
    }
  }

  std::cout << "----------------" << std::endl;
  for (int64_t i = -4; i < 5; ++i) {
    show(i + max_dis_percent_idx);
  }
  std::cout << "----------------" << std::endl;
  for (int64_t i = -4; i < 5; ++i) {
    show(i + max_dis_idx);
  }
  std::cout << "----------------" << std::endl;

  double div = sum / size;
  std::cout << "mean abs dis percent error: " << div << std::endl;
  if (abort_thr == -1) {
    return;
  }
  std::cout << "----------------" << std::endl;
}

class Func {
 public:
  // \brief The input params should be ptr.
  template <typename... Args>
  Func(void *func, Args... rest) : func_(func) {
    CALL_CNRT(cnrtCreateQueue(&queue_));
    CALL_CNRT(cnrtCreateNotifier(&start_));
    CALL_CNRT(cnrtCreateNotifier(&end_));

    CALL_CNRT(cnrtGetKernelParamsBuffer(&params_));
    AddParam(rest...);
  }

  float Invoke(int core_num, int loop_times = 1) {
    VerifyCoreNum(core_num);

    auto dim3 = GetDim3(core_num);
    auto func_type = GetFuncType(core_num);

    float sum = 0.f;
    for (int i = 0; i < loop_times; ++i) {
      CALL_CNRT(cnrtPlaceNotifier(start_, queue_));
      CALL_CNRT(cnrtInvokeKernel_V2(func_, dim3, params_, func_type, queue_));
      CALL_CNRT(cnrtPlaceNotifier(end_, queue_));
      CALL_CNRT(cnrtSyncQueue(queue_));

      float us;
      CALL_CNRT(cnrtNotifierDuration(start_, end_, &us));
      sum += us;
    } 
    std::cout << "Hardware Time: " << (sum / loop_times) << "us" << std::endl;

    return sum / loop_times;
  }

  ~Func() {
    if (queue_) {
      CALL_CNRT(cnrtDestroyQueue(queue_));
    }
    if (params_) {
      CALL_CNRT(cnrtDestroyKernelParamsBuffer(params_));
    }
    if (start_) {
      CALL_CNRT(cnrtDestroyNotifier(&start_));
    }
    if (end_) {
      CALL_CNRT(cnrtDestroyNotifier(&end_));
    }
  }

 private:
  void AddParam() {}

  template <typename T>
  void AddParam(T *param) {
    CALL_CNRT(cnrtKernelParamsBufferAddParam(params_, param, sizeof(T)));
  }

  template <typename T, typename... Args>
  void AddParam(T *param, Args... rest) {
    CALL_CNRT(cnrtKernelParamsBufferAddParam(params_, param, sizeof(T)));
    AddParam(rest...);
  }

  void VerifyCoreNum(int core_num) {
  }

  cnrtDim3_t GetDim3(int core_num) {
    cnrtDim3_t dims;
    dims.x = core_num;
    dims.y = 1;
    dims.z = 1;
    return dims;
  } 

  cnrtFunctionType_t GetFuncType(int core_num) {
    switch (core_num) {
      case 1:
        return CNRT_FUNC_TYPE_BLOCK;
      
      case 4:
        return CNRT_FUNC_TYPE_UNION1;

      case 8:
        return CNRT_FUNC_TYPE_UNION2;
      
      case 16:
        return CNRT_FUNC_TYPE_UNION4;

      default:
        return CNRT_FUNC_TYPE_UNION1;
    }
  }

 private:
  void *func_{nullptr};
  cnrtKernelParamsBuffer_t params_{nullptr};
  cnrtQueue_t queue_{nullptr};

  cnrtNotifier_t start_{nullptr};
  cnrtNotifier_t end_{nullptr};
};


class Tensor {
 public:
  Tensor() {}

  Tensor(size_t size, DataType data_type = kFloat16, const void *data = nullptr) {
    data_type_ = data_type;
    size_ = size;
    memory_size_ = byteof(data_type) * size;
    capacity_ = memory_size_;
    CALL_CNRT(cnrtMalloc(&data_, memory_size_));

    if (data) {
      Copyin(data);
    }
  }

  Tensor operator=(Tensor &other) {
    this->size_ = other.size();
    this->capacity_ = other.capacity();
    this->memory_size_ = other.memory_size();
    this->data_ = other.data();
    this->data_type_ = other.data_type();
    
    other.count += 1;

    return *this;
  }

  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

  size_t memory_size() const { return memory_size_; }

  void *data() const { return data_; }

  DataType data_type() const { return data_type_; }

  void SetDataType(DataType data_type) {
    data_type_ = data_type;
  }

  void Resize(size_t size, DataType data_type = kInvalid) {
    size_ = size;
    memory_size_ = byteof(data_type) * size;
    data_type_ = data_type;

    if (memory_size_ > capacity_) {
      if (data_) {
        CALL_CNRT(cnrtFree(data_));
      }
      CALL_CNRT(cnrtMalloc(&data_, memory_size_));
      capacity_ = memory_size_;
    }
  }

  void **pdata() { return &data_; }

  void Copyin(const void *data) {
    CALL_CNRT(cnrtMemcpy(data_, const_cast<void *>(data), memory_size_, CNRT_MEM_TRANS_DIR_HOST2DEV));
  }

  void Copyin(const void *data, size_t size) {
    CALL_CNRT(cnrtMemcpy(data_, const_cast<void *>(data), size, CNRT_MEM_TRANS_DIR_HOST2DEV));
  }

  void Copyout(void *data) {
    CALL_CNRT(cnrtMemcpy(data, data_, memory_size_, CNRT_MEM_TRANS_DIR_DEV2HOST));
  }

  template <typename T>
  std::vector<T> Copyout() {
    std::vector<T> output_host(size_);
    Copyout(output_host.data());
    return std::move(output_host);
  }

  ~Tensor() {
    if (data_ && count <= 0) {
      CALL_CNRT(cnrtFree(data_));
    }
  }

 public:
  int count{0};

 private:
  DataType data_type_{kInvalid};
  void *data_{nullptr};
  size_t size_{0};
  size_t memory_size_{0};
  size_t capacity_{0};
};

