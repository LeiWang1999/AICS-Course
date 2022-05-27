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

#if CAMBRICON_MLU
#include "tensorflow/core/kernels/yolov3_detection_output_op_mlu.h"

namespace tensorflow {
#define REGISTER_MLU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
         //TODO:完成算子注册            
         ......
#undef REGISTER_MLU
#endif  // CAMBRICON_MLU
}
