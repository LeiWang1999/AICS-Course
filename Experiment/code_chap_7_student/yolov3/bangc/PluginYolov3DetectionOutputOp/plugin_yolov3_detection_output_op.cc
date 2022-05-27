/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "cnplugin.h"
#include "plugin_yolov3_detection_output_kernel_v1.h"

cnmlStatus_t cnmlCreatePluginYolov3DetectionOutputOpParam(
    cnmlPluginYolov3DetectionOutputOpParam_t *param,
    int batchNum,
    int inputNum,
    int classNum,
    int maskGroupNum,
    int maxBoxNum,
    int netw,
    int neth,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version,
    int* inputWs,
    int* inputHs,
    float* biases) {
  // CHECK_ENFORCE(param, "param shouldn't be nullptr");
  *param = new cnmlPluginYolov3DetectionOutputOpParam();

  (*param)->cnml_static_tensors
    = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t) * 7);
  (*param)->cpu_static_tensors
    = (cnmlCpuTensor_t *)malloc(sizeof(cnmlCpuTensor_t) * 7);

  // malloc and create const tensor for inputWs, inputHs, and biases
  cnmlCreateTensor(
      &(*param)->cnml_static_tensors[4],  // imageWs
      CNML_CONST, CNML_DATA_INT32,
      1, 64, 1, 1);

  cnmlCreateTensor(
      &(*param)->cnml_static_tensors[5],  // imageHs
      CNML_CONST, CNML_DATA_INT32,
      1, 64, 1, 1);

  cnmlCreateTensor(
      &(*param)->cnml_static_tensors[6],  // biases
      CNML_CONST, CNML_DATA_FLOAT16,
      1, 64, 1, 1);

  cnmlCreateCpuTensor(
      &(*param)->cpu_static_tensors[4],  // imageWs
      CNML_CONST, CNML_DATA_INT32,
      CNML_NHWC, 1, 64, 1, 1);

  cnmlCreateCpuTensor(
      &(*param)->cpu_static_tensors[5],  // imageHs
      CNML_CONST, CNML_DATA_INT32,
      CNML_NHWC, 1, 64, 1, 1);

  cnmlCreateCpuTensor(
      &(*param)->cpu_static_tensors[6],  // biases
      CNML_CONST, CNML_DATA_FLOAT32,
      CNML_NHWC, 1, 64, 1, 1);

  // malloc and create const tensor for fake input tensor
  for (int inputId = 0; inputId < 7 - inputNum; inputId++) {
    cnmlCreateTensor(
        &(*param)->cnml_static_tensors[inputId],  // fake inputs
        CNML_CONST, CNML_DATA_INT32,
        1, 64, 1, 1);
  }

  for (int inputId = 0; inputId < 7 - inputNum; inputId++) {
    cnmlCreateCpuTensor(
        &(*param)->cpu_static_tensors[inputId],  // fake inputs
        CNML_CONST, CNML_DATA_INT32,
        CNML_NHWC, 1, 64, 1, 1);
  }

  // scalar params
  (*param)->batchNum = batchNum;
  (*param)->inputNum = inputNum;
  (*param)->classNum = classNum;
  (*param)->maskGroupNum = maskGroupNum;
  (*param)->maxBoxNum = maxBoxNum;
  (*param)->netw = netw;
  (*param)->neth = neth;
  (*param)->confidence_thresh = confidence_thresh;
  (*param)->nms_thresh = nms_thresh;
  (*param)->core_version = core_version;

  // bind const data
  (*param)->inputWs = (int *)malloc(sizeof(int) * 64);
  (*param)->inputHs = (int *)malloc(sizeof(int) * 64);
  (*param)->biases  = (float *)malloc(sizeof(int) * 64);

  for (int inputId = 0; inputId < inputNum; inputId++) {
    (*param)->inputWs[inputId] = inputWs[inputId];
    (*param)->inputHs[inputId] = inputHs[inputId];
  }
  for (int biasId = 0; biasId < 2 * maskGroupNum * inputNum; biasId++) {
    (*param)->biases[biasId] = biases[biasId];
  }

  //If MluTensor's datatype != CpuTensor's datatype, need to cast data.
  void *cast_biases = (int16_t*)malloc(sizeof(int16_t)*(1*64*1*1));
  cnrtCastDataType((*param)->biases, CNRT_FLOAT32, cast_biases, CNRT_FLOAT16, 64, nullptr);
  (*param)->cast_data.push_back(cast_biases);

  cnmlBindConstData_V2(
      (*param)->cnml_static_tensors[0],
      (*param)->inputWs,
      false);
  cnmlBindConstData_V2(
      (*param)->cnml_static_tensors[1],
      (*param)->inputWs,
      false);
  cnmlBindConstData_V2(
      (*param)->cnml_static_tensors[2],
      (*param)->inputWs,
      false);
  cnmlBindConstData_V2(
      (*param)->cnml_static_tensors[3],
      (*param)->inputWs,
      false);
  cnmlBindConstData_V2(
      (*param)->cnml_static_tensors[4],
      (*param)->inputWs,
      false);
  cnmlBindConstData_V2(
      (*param)->cnml_static_tensors[5],
      (*param)->inputHs,
      false);
  cnmlBindConstData_V2(
      (*param)->cnml_static_tensors[6],
      cast_biases,
      false);

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlDestroyPluginYolov3DetectionOutputOpParam(
    cnmlPluginYolov3DetectionOutputOpParam_t *param) {
  // CHECK_ENFORCE(param, "param pointer shouldn't be nullptr!");
  // CHECK_ENFORCE(*param, "param is nullptr, maybe double free!");

  // destroy static tensors
  for (int i = 0; i < 7; i++) {
    cnmlDestroyTensor(
        &(*param)->cnml_static_tensors[i]);
  }
  for (int i = 0; i < 7; i++) {
    cnmlDestroyCpuTensor(
        &(*param)->cpu_static_tensors[i]);
  }
  free((*param)->cnml_static_tensors);
  free((*param)->cpu_static_tensors);
  free((*param)->inputWs);
  free((*param)->inputHs);
  free((*param)->biases);
  for (auto ptr : (*param)->cast_data) {
    free(ptr);
  }
  delete (*param);
  *param = nullptr;

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCreatePluginYolov3DetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginYolov3DetectionOutputOpParam_t param,
    cnmlTensor_t *yolov3_input_tensors,
    cnmlTensor_t *yolov3_output_tensors) {
  // CHECK_ENFORCE(op, "op shouldn't be nullptr.");
  // CHECK_ENFORCE(param, "param shouldn't be nullptr.");
  // CHECK_ENFORCE(yolov3_output_tensors, "bbox_deltas shouldn't be nullptr.");
  // CHECK_ENFORCE(yolov3_input_tensors, "scores shouldn't be nullptr");

  // read OpParam
  int pad_size = 64;
  int batchNum = param->batchNum;
  int inputNum = param->inputNum;
  int classNum = param->classNum;
  int maskGroupNum = param->maskGroupNum;
  int maxBoxNum = param->maxBoxNum;
  int netw = param->netw;
  int neth = param->neth;
  float confidence_thresh
    = param->confidence_thresh;
  float nms_thresh
    = param->nms_thresh;
  cnmlCoreVersion_t core_version
    = param->core_version;

  // convert thresh from float to half
  uint16_t confidence_threshold_half;
  uint16_t nms_threshold_half;
  cnrtConvertFloatToHalf(&confidence_threshold_half, confidence_thresh);
  cnrtConvertFloatToHalf(&nms_threshold_half, nms_thresh);

  // prepare op
  int input_num = inputNum;
  int output_num = 2;
  int static_num = 7;
  cnmlTensor_t *cnml_static_tensors
    = param->cnml_static_tensors;
  cnmlCpuTensor_t *cpu_static_tensors
    = param->cpu_static_tensors;

  // prepare bangC-kernel param
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferMarkOutput(params); // output tensor

  for (int count = 0; count < inputNum; count++) {
    cnrtKernelParamsBufferMarkInput(params); // input tensors
  }

  for (int count = 0; count < 7 - inputNum; count++) {
    cnrtKernelParamsBufferMarkStatic(params); // fake input tensors
  }

  cnrtKernelParamsBufferMarkOutput(params); // buffer tensor
  cnrtKernelParamsBufferMarkStatic(params); // w arr
  cnrtKernelParamsBufferMarkStatic(params); // h arr
  cnrtKernelParamsBufferMarkStatic(params); // bias data

  cnrtKernelParamsBufferAddParam(params, &inputNum, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &classNum, sizeof(int));

  int SPI_DISABLED = 0;
  if (SPI_DISABLED) {
    cnrtKernelParamsBufferAddParam(params, &batchNum, sizeof(int));
  } else {
    cnmlTensor_t *mlu_tensors
      = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t) * (inputNum + 2));

    mlu_tensors[0] = yolov3_output_tensors[0];
    for (int count = 0; count < inputNum; count++) {
      mlu_tensors[count + 1] = yolov3_input_tensors[count];
    }
    mlu_tensors[inputNum + 1] = yolov3_output_tensors[1];

    cnmlDimension_t *dimension
      = (cnmlDimension_t *)malloc(sizeof(cnmlDimension_t) * (inputNum + 2)) ;
    dimension[0] = cnmlDimension_t::CNML_DIM_C;
    for (int count = 0; count < inputNum + 1; count++) {
      dimension[count + 1] = cnmlDimension_t::CNML_DIM_N;
    }
    cnmlPluginOpParamsBufferMarkTensorDimension(params, mlu_tensors,
                    dimension, inputNum + 2);
    free(mlu_tensors);
    free(dimension);
  }

  // cnrtKernelParamsBufferAddParam(params, &batchNum, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &maskGroupNum, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &maxBoxNum, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &pad_size, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &netw, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &neth, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &confidence_threshold_half,
                                 sizeof(uint16_t));
  cnrtKernelParamsBufferAddParam(params, &nms_threshold_half, sizeof(uint16_t));

  // create Plugin op
  void **InterfacePtr;
  switch (core_version){
    case CNML_MLU220:
      InterfacePtr = reinterpret_cast<void **>(&yolov3Kernel_MLU220);
      break;
    default:
      InterfacePtr = reinterpret_cast<void **>(&yolov3Kernel_MLU270);
      break;
  }
  cnmlCreatePluginOp(op,
                     "yolov3_detection",
                     InterfacePtr,
                     params,
                     yolov3_input_tensors,
                     input_num,
                     yolov3_output_tensors,
                     output_num,
                     cnml_static_tensors,
                     static_num);

  cnrtDestroyKernelParamsBuffer(params);
  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlComputePluginYolov3DetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *inputs[],
    int num_inputs,
    void *outputs[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue) {
  cnmlComputePluginOpForward_V3(op,
                                inputs,
                                num_inputs,
                                outputs,
                                num_outputs,
                                compute_forw_param,
                                queue);

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCpuComputePluginYolov3DetectionOutputOpForward(
    cnmlPluginYolov3DetectionOutputOpParam_t param,
    void **inputs,
    void *outputs) {
  int batchNum = param->batchNum;
  int inputNum = param->inputNum;
  int classNum = param->classNum;
  int anchorNum = param->maskGroupNum;
  int maxBoxNum = param->maxBoxNum;
  int netw = param->netw;
  int neth = param->neth;
  float confidence_thresh = param->confidence_thresh;
  float nms_thresh = param->nms_thresh;

  int* inputWs = param->inputWs;
  int* inputHs = param->inputHs;
  float* biases = param->biases;

  int total_hw = inputHs[0] * inputWs[0]
               + inputHs[1] * inputWs[1]
               + inputHs[2] * inputWs[2];

  std::vector<float> obj_kept;
  std::vector<std::vector<float> >loc_kept;
  std::vector<std::vector<float> >prob_kept;
  int unit = 5 + classNum;
  int ch = unit * anchorNum;

  std::cout << "==================== CPU TEST ===================="
            << std::endl;
  #if DEBUG_MODE
  std::cout << "inputNum: " << inputNum << std::endl;
  std::cout << "batchNum: " << batchNum << std::endl;
  #endif

  for (int batchIdx = 0; batchIdx < batchNum; batchIdx++) {
    int outBatchSize = (maxBoxNum * 7 + 64) * batchIdx;
    int boxCount = 0;
    obj_kept.clear();
    loc_kept.clear();
    prob_kept.clear();
    int idx = 0;
    for (int i = 0; i < inputNum; i++) {
      int hw = inputWs[i] * inputHs[i];
      int batchSize = hw * ch;

      #ifdef DEBUG_MDOE
      std::cout << "inputId: " << i << " " << hw << std::endl;
      #endif

      for (int j = 0; j < anchorNum; j++) {
        for (int k = 0; k < hw; k++) {
          int x_idx   = j * unit * hw + 0 * hw + k + batchSize * batchIdx;
          int y_idx   = j * unit * hw + 1 * hw + k + batchSize * batchIdx;
          int w_idx   = j * unit * hw + 2 * hw + k + batchSize * batchIdx;
          int h_idx   = j * unit * hw + 3 * hw + k + batchSize * batchIdx;
          int obj_idx = j * unit * hw + 4 * hw + k + batchSize * batchIdx;
          float obj = 1.0 / (1 + std::exp(-((float *)inputs[i])[obj_idx]));
          if (obj > confidence_thresh) {
            obj_kept.push_back(obj);
            int x_offset  = k % inputWs[i];
            int y_offset  = k / inputHs[i];
            float w_bias  = (float)biases[i * 6 + j * 2 + 0] / netw;
            float h_bias  = (float)biases[i * 6 + j * 2 + 1] / neth;

            #ifdef DEBUG_MODE
            std::cout << "===== before =====" << std::endl;
            std::cout << ((float *)inputs[i])[x_idx] << " "
                      << ((float *)inputs[i])[y_idx] << " "
                      << ((float *)inputs[i])[w_idx] << " "
                      << ((float *)inputs[i])[h_idx] << " "
                      << std::endl;

            std::cout << i << " "
                      << j << " "
                      << k << " "
                      << x_offset << " "
                      << y_offset << " "
                      << std::endl;
            #endif
            std::vector<float> loc_tmp;
            loc_tmp.push_back(
              (x_offset + 1.0 / (1 + std::exp(-((float *)inputs[i])[x_idx]))) / inputWs[i]);
            loc_tmp.push_back(
              (y_offset + 1.0 / (1 + std::exp(-((float *)inputs[i])[y_idx]))) / inputHs[i]);
            loc_tmp.push_back(
              std::exp(((float *)inputs[i])[w_idx]) * w_bias);
            loc_tmp.push_back(
              std::exp(((float *)inputs[i])[h_idx]) * h_bias);
            loc_kept.push_back(loc_tmp);

            #ifdef DEBUG_MODE
            std::cout << "===== after =====" << std::endl;
            std::cout << loc_kept[idx][0] << " "
                      << loc_kept[idx][1] << " "
                      << loc_kept[idx][2] << " "
                      << loc_kept[idx][3] << " "
                      << idx << " "
                      << std::endl
                      << std::endl;
            #endif

            std::vector<float> prob_tmp;
            for (int entry = 0; entry < classNum; entry++) {
              int prob_idx = j * unit * hw + (entry + 5) * hw + k + batchSize * batchIdx;
              prob_tmp.push_back(
                1.0 / (1 + std::exp(-((float *)inputs[i])[prob_idx])));
            }
            prob_kept.push_back(prob_tmp);
            idx++;
          }
        }
      }
    }

    #ifdef DEBUG_MODE
    std::cout << "maskCount: " << obj_kept.size() << std::endl;
    std::cout << "===== check obj =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << obj_kept[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check x =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_kept[i][0] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check y =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_kept[i][1] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check w =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_kept[i][2] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check h =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_kept[i][3] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    #endif

    for (int i = 0; i < idx; i++) {
      loc_kept[i][0] = loc_kept[i][0] - loc_kept[i][2] / 2;
      loc_kept[i][1] = loc_kept[i][1] - loc_kept[i][3] / 2;
      loc_kept[i][2] = loc_kept[i][0] + loc_kept[i][2];
      loc_kept[i][3] = loc_kept[i][1] + loc_kept[i][3];
    }

    #if DEBUG_MODE
    std::cout << "===== check obj =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << obj_kept[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check x1 =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_kept[i][0] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check y1 =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_kept[i][1] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check x2 =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_kept[i][2] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check y2 =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_kept[i][3] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    #endif

    #if DEBUG_MODE
    std::cout << "===== check obj =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << obj_sort[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check x1 =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_sort[i][0] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check y1 =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_sort[i][1] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check x2 =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_sort[i][2] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "===== check y2 =====" << std::endl;
    for (int i = 0; i < idx; i++) {
      std::cout << loc_sort[i][3] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    #endif

    // NMS BY CLASS
    for (int classIdx = 0 ; classIdx < classNum; classIdx++) {
      std::vector<float> target;
      for (int boxIdx = 0; boxIdx < idx; boxIdx++) {
        target.push_back(obj_kept[boxIdx] * prob_kept[boxIdx][classIdx]);
      }
      for (int iter = 0; iter < idx; iter++) {
        float maxValue = *max_element(target.begin(), target.end());
        int   maxIndex = max_element(target.begin(), target.end()) - target.begin();
        if (maxValue < confidence_thresh)
          continue;
        // store and printf boxParams
        ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 0] = batchIdx;
        ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 1] = classIdx;
        ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 2] = maxValue;
        ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 3] = loc_kept[maxIndex][0];
        ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 4] = loc_kept[maxIndex][1];
        ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 5] = loc_kept[maxIndex][2];
        ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 6] = loc_kept[maxIndex][3];
        boxCount++;
        std::cout << batchIdx << " "
                  << classIdx << " "
                  << maxValue << " "
                  << loc_kept[maxIndex][0] << " "
                  << loc_kept[maxIndex][1] << " "
                  << loc_kept[maxIndex][2] << " "
                  << loc_kept[maxIndex][3] << " "
                  << std::endl
                  << std::endl;
        target[maxIndex] = -1;

        // do nms
        float x1_star = loc_kept[maxIndex][0];
        float y1_star = loc_kept[maxIndex][1];
        float x2_star = loc_kept[maxIndex][2];
        float y2_star = loc_kept[maxIndex][3];
        for (int nmsIter = 0; nmsIter < idx; nmsIter++) {
          if ((target[nmsIter] <= confidence_thresh) || (nmsIter == maxIndex))
            continue;
          // compute IOU
          float x1 = loc_kept[nmsIter][0];
          float y1 = loc_kept[nmsIter][1];
          float x2 = loc_kept[nmsIter][2];
          float y2 = loc_kept[nmsIter][3];
          float x1_max = std::max(x1_star, x1);
          float y1_max = std::max(y1_star, y1);
          float x2_min = std::min(x2_star, x2);
          float y2_min = std::min(y2_star, y2);
          float inter_area = (y2_min - y1_max + 1.0 / neth)
                           * (x2_min - x1_max + 1.0 / netw);
          if ((y2_min <= y1_max) || (x2_min <= x1_max))
            inter_area = 0.0;
          float union_area = (y2_star - y1_star + 1.0 / neth)
                           * (x2_star - x1_star + 1.0 / netw)
                           + (y2 - y1 + 1.0 / neth)
                           * (x2 - x1 + 1.0 / netw)
                           - inter_area;

          #ifdef DEBUG_MODE
          std::cout << "  " << inter_area << " "
                    << "  " << union_area << " "
                    << "  " << inter_area / union_area << " "
                    << std::endl
                    << std::endl;
          #endif

          float IOU = inter_area / union_area;
          if ((IOU >= nms_thresh)) {
          target[nmsIter] = -1;
          }
        }
      }
      for (int nmsIdx = 0; nmsIdx < idx; nmsIdx++) {
        if (target[nmsIdx] > confidence_thresh) {
          // store and printf boxParams
          ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 0] = batchIdx;
          ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 1] = classIdx;
          ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 2] = target[nmsIdx];
          ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 3] = loc_kept[nmsIdx][0];
          ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 4] = loc_kept[nmsIdx][1];
          ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 5] = loc_kept[nmsIdx][2];
          ((float *)outputs)[outBatchSize + 64 + boxCount * 7 + 6] = loc_kept[nmsIdx][3];
          boxCount++;
          std::cout << batchIdx << " "
                    << classIdx << " "
                    << target[nmsIdx] << " "
                    << loc_kept[nmsIdx][0] << " "
                    << loc_kept[nmsIdx][1] << " "
                    << loc_kept[nmsIdx][2] << " "
                    << loc_kept[nmsIdx][3] << " "
                    << std::endl
                    << std::endl;
        }
      }
    }
    ((float *)outputs)[outBatchSize] = (float)boxCount;
  }
  std::cout << "========= Num of valid box from cpu is "
            << (int)(((float *)outputs)[0]) << " ========="
            << std::endl;
  return CNML_STATUS_SUCCESS;
}
