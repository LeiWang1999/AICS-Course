/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc.
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
#include "cnml.h"
#include "cnrt.h"
#include "stdlib.h"
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <algorithm>
#include <cmath>

#ifndef CNPLUGIN_H_
#define CNPLUGIN_H_

#define CNPLUGIN_MAJOR_VERSION 0
#define CNPLUGIN_MINOR_VERSION 0
#define CNPLUGIN_PATCH_VERSION 0

using std::vector;
typedef uint16_t half;

#define CNPLUGIN_VERSION (CNPLUGIN_MAJOR_VERSION * 10000 + CNPLUGIN_MINOR_VERSION * 100 + CNPLUGIN_PATCH_VERSION)

/* ====================== */
/* enum definitions start */
/* ====================== */
/*!
 *  @enum cnmlPluginSsdCodeType_t
 *  @breif An enum.
 *
 *  ``cnmlPluginSsdCodeType_t`` is an enum holding the description of CodeType
 *  used in PluginSsdDetectionOutoutOp, including:
 *    CodeType_CORNER:      (x1, y1) + (x2, y2)
 *    CodeType_CENTER_SIZE: (xc, yc) + (w , h )
 *    CodeType_CORNER_SIZE: (x1, y1) + (w , h )
 *    where (x1, y1) represents the top-left corner,
 *          (x2, y2) represents the bottom-right corner, and
 *          (w , h ) represents the (w)idth and (h)eight.
 */
typedef enum
{
  CodeType_CORNER = 0,
  CodeType_CENTER_SIZE = 1,
  CodeType_CORNER_SIZE = 2,
} cnmlPluginSsdCodeType_t;

/*!
 *  @enum cnmlPluginColorCvt_t
 *  @brief An enum.
 *
 *  ``cnmlPluginColorCvt_t`` is an num holding the description of color
 *  conversion mode used in ``ResizeAndColorCvt`` kind of operations, including:
 *  Resize, ResizeYuvToRgba, CropAndResize, YuvToRgba. More will come.
 */
typedef enum
{
  RGBA_TO_RGBA = 0,
  YUV_TO_RGBA_NV12 = 1,
  YUV_TO_RGBA_NV21 = 2,
  YUV_TO_BGRA_NV12 = 3,
  YUV_TO_BGRA_NV21 = 4,
  YUV_TO_ARGB_NV12 = 5,
  YUV_TO_ARGB_NV21 = 6,
  YUV_TO_ABGR_NV12 = 7,
  YUV_TO_ABGR_NV21 = 8,
  GRAY_TO_GRAY = 9
} cnmlPluginColorCvt_t;

/*!
 *  @enum cnmlPluginDataType_t
 *  @brief An enum.
 *
 *  ``cnmlPluginDataType_t`` is an num holding the description of datatype
 *  conversion mode used in ``ResizeAndColorCvt`` kind of operations, including:
 *  Resize, ResizeYuvToRgba, CropAndResize, YuvToRgba. More will come.
 */
typedef enum
{
  FP16_TO_FP16 = 0,
  FP16_TO_UINT8 = 1,
  UINT8_TO_FP16 = 2,
  UINT8_TO_UINT8 = 3
} cnmlPluginDataType_t;
/* -------------------- */
/* enum definitions end */
/* -------------------- */

/* ======================== */
/* struct definitions start */
/* ======================== */
/*!
 *  @struct roiParams
 *  @brief A struct.
 *
 *  ``roiParams`` is a struct holding the description of bounding box info.
 *  CORNER_SIZE mode is used here, i.e., all bounding boxes are discribed in
 *  terms of (x1, y1) + (w , h ).
 */
typedef struct roiParams
{
  int roi_x;
  int roi_y;
  int roi_w;
  int roi_h;
} roiParams;

/*!
 *  @struct ioParams
 *  @brief A struct
 *
 *  ``ioParams`` is a struct holding the descroption of color and datatype
 *  conversion mode used in ``ResizeAndColorCvt`` kind of operations, including:
 *  Resize, ResizeYuvToRgba, CropAndResize, YuvToRgba. More will come.
 */
typedef struct ioParams {
  cnmlPluginColorCvt_t color;
  cnmlPluginDataType_t datatype;
} ioParams;

/*!
 *  @struct cnmlPluginResizeAndColorCvtParam
 *  @brief
 *
 *  ``cnmlPluginResizeAndColorCvtParam`` is a struct holding the parameters used
 *  in ``ResizeAndColotCvt`` kind of operations. In this struct, users only need
 *  to provide "user params". Others will be parsed through the ioParams chosen
 *  by users.
 */
struct cnmlPluginResizeAndColorCvtParam {
  // user params
  int s_row;
  int s_col;
  int d_row;
  int d_col;
  int roi_x;
  int roi_y;
  int roi_w;
  int roi_h;
  ioParams mode;
  int batchNum;
  cnmlCoreVersion_t core_version;

  // operation params
  int inputType;
  int outputType;
  int channelIn;
  int channelOut;
  int layerIn;
  int layerOut;
  int reverseChannel;
  int input2half;
  int output2uint;
  cnmlDataType_t inputDT_MLU;
  cnmlDataType_t inputDT_CPU;
  cnmlDataType_t outputDT_MLU;
  cnmlDataType_t outputDT_CPU;

  int input_num;
  int output_num;
  int static_num;

  // for cropfeatureandresize
  int depth;
  int box_number;
  int pad_size;

  cnmlTensor_t *cnml_static_ptr;
  cnmlCpuTensor_t *cpu_static_ptr;
  void **static_data_ptr;
};
/*! ``cnmlPluginResizeAndColorCvtParam_t`` is a pointer to a
    structure (cnmlPluginResizeAndColorCvtParam) holding the description of CV operations param.
*/
typedef cnmlPluginResizeAndColorCvtParam *cnmlPluginResizeAndColorCvtParam_t;
/* ---------------------- */
/* struct definitions end */
/* ---------------------- */

/* =============================================== */
/* cnmlPluginYolov3DetectionOutout operation start */
/* =============================================== */
/*!
 *  @struct cnmlPluginYolov3DetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginYolov3DetectionOutputOpParam is a structure describing the "param"
 *  parameter of Yolov3DetectionOutput operation.
 *  cnmlCreatePluginYolov3DetectionOutputOpParam() is used to create
 *  an instance of cnmlPluginYolov3DetectionOutputOpParam_t.
 *  cnmlDestroyPluginYolov3DetectionOutputOpParam() is used to destroy
 *  an instance of cnmlPluginYolov3DetectionOutputOpParam_t.
 */
/*struct cnmlPluginYolov3DetectionOutputOpParam
{
    cnmlTensor_t *cnml_static_tensors;
    cnmlCpuTensor_t *cpu_static_tensors;
    int batchNum;
    int inputNum;
    int classNum;
    int maskGroupNum;
    int maxBoxNum;
    int netw;
    int neth;
    float confidence_thresh;
    float nms_thresh;
    cnmlCoreVersion_t core_version;
    int *inputWs;
    int *inputHs;
    float *biases;
    vector<void*> cast_data;
};*/
/*! ``cnmlPluginYolov3DetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginYolov3DetectionOutputOpParam) holding the description of a Yolov3DetectionOutput operation param.
*/
/*typedef cnmlPluginYolov3DetectionOutputOpParam
*cnmlPluginYolov3DetectionOutputOpParam_t;
*/
/*!
 *  @brief A function.
 *
 *  This function creates a PluginYolov3DetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of input batches.
 *           No default value, a valid batchNum must be in the range of [1, inf).
 *  @param[in] inputNum
 *    Input. The number of input tensors.
 *           No default value, a valid inputNum must be in the range of [1, 7].
 *  @param[in] classNum
 *    Input. The number of input classes.
 *           No default value, a valid classNum must be in the range of [1, 4096].
 *  @param[in] maskGroupNum
 *    Input. The number of anchors used by every input tensors.
 *           No default value, a valid maskGroupNum must be in the range of [1, inf].
 *  @param[in] maxBoxNum
 *    Input. The largest possible number of output boxes.
 *           Default value is 1024, a valid maxBoxNum must be in the range of [1, inf].
 *  @param[in] netw
 *    Input. Width of input image of backbone network.
 *           No default value, a valid netw must be in the range of [1, inf).
 *  @param[in] neth
 *    Input. Height of input image of backbone network.
 *           No default value, a valid neth must be in the range of [1, inf).
 *  @param[in] confidence_thresh
 *    Input. Confidence threshold.
 *           No default value, a valid confidence_thresh must be in the range of [0, 1].
 *  @param[in] nms_thresh.
 *    Input. IOU threshold used in NMS function.
 *           No default value, a valid nms_thresh must be in the range of [0, 1].
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value, a valid core_version must be either MLU220 or MLU270.
 *  @param[in] inputWs
 *    Input. Width of every input tensor. Must have the same order as inputHs
 *           No default value, the number of valid elements must be equal with inputNum.
 *  @param[in] inputHs
 *    Input. Height of every input tensor. Must have the same order as inputWs
 *           No default value, the number of valid elements must be equal with inputNum.
 *  @param[in] biases
 *    Input. Anchors of every input tensor.
 *           No default value. The number of valid elements must be equal with 2 x inputNum x maskGroupNum.
 *           The order of data from high to low, is [N(1) H(inputNum) W(maskGroupNum) C(2)]. For example:
 *
 *           Width of anchor for mask0 input0, Height of anchor for mask0 input0,
 *
 *           Width of anchor for mask1 input0, Height of anchor for mask1 input0,
 *
 *           ...
 *
 *           Width of anchor for maskN input0, Height of anchor for maskN input0,
 *
 *           Width of anchor for mask0 input1, Height of anchor for mask0 input1,
 *
 *           ......
 *
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    The inputH/Ws ptr is nullptr or input param is invalid.
 */
/*cnmlStatus_t cnmlCreatePluginYolov3DetectionOutputOpParam(
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
    int *inputWs,
    int *inputHs,
    float *biases);
*/
/*!
 *  @brief A function.
 *
 *  This function frees the PluginYolov3DetectionOutputOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU220/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginYolov3DetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
/*cnmlStatus_t cnmlDestroyPluginYolov3DetectionOutputOpParam(
    cnmlPluginYolov3DetectionOutputOpParam_t *param);
*/
/*!
 *  @brief A function.
 *
 *  This function creates PluginYolov3DetectionOutputOp with proper param,
 *  input, and output tensors.
 *
 *  PluginYolov3DetectionOutputOp takes in feature maps and network
 *  parameters and computes valid bounding boxes based on two thresholds
 *  you have chosen.
 *
 *  **Reference:**
 *    This implementation is based on the project on ``github/pjreddie/darknet`` .
 *
 *  **Formula:** This op contains two steps:
 *
 *    1. DecodeAllBBoxes.
 *
 *       Convert input feature maps into real ojectness score and coordinates.
 *    for inputIdx in (0, inputNum - 1)
 *
 *       obj = sigmoid(obj_feature);
 *
 *       x   = (x_offset + sigmoid(x_feature)) / inputWs[inputIdx]
 *
 *       y   = (y_offset + sigmoid(y_feature)) / inputHs[inputIdx]
 *
 *       w   = (w_biases * exp(w_feature)) / netw
 *
 *       h   = (h_biases * exp(h_feature)) / neth
 *
 *       Obj, x_feature, y_feature, w_feature, h_feature are data from input feature maps.
 *
 *       x_offset, y_offset are the coordinates of the grid cell in the feature map.
 *
 *       w_offset, h_biases are the shape of the anchor box.
 *
 *    2. Non-maximum Suppression
 *       For each class of data, compute IOU score for every pair of bounding boxes.
 *
 *       If IOU score exceeds the IOU threshold, keep the box with larger score.
 *
 *       x1 = x - w / 2
 *
 *       y1 = y - y / 2
 *
 *       x2 = x + w / 2
 *
 *       y2 = y + y / 2
 *
 *       for classIdx in (0, classNum - 1)
 *
 *        conf = obj * probability[classIdx]
 *
 *        max, maxIdx = findMaxValueAndIndex(conf)
 *
 *        if (max >= confidence_thresh)
 *
 *          for boxIdx in (0, boxNum - 1)
 *
 *            iou = computeIOU(coord_maxIdx, coord_boxIdx)  // where "coords" means x1,y1,x2,y2
 *
 *            if (iou < nms_thresh)
 *
 *              keep coords and conf for boxIdx
 *
 *  **DataType:**
 *
 *    Support only half(float16) type for both input and output tensors.
 *
 *  **Performance Optimization:**
 *
 *    The performance of detection layer depends on both the data size and the value.
 *    However, this op achieves relatively better performance when
 *    all of the following conditions are met:
 *
 *    - inputH/Ws are 64-aligned(unit in number of data).
 *
 *    - (5 + classNum) is 64-aligned(unit in number of data).
 *
 *    The bigger the remainder of the value of param divided by 64, the better performance the op will achieve.
 *
 *  Supports both MLU220 and MLU270.
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYolov3DetectionOutput parameter struct pointer.
 *  @param[in]  yolov3_input_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, (5 + classNum) * numMaskGroup, inputH, inputW](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, 64 + 7 * numMaxBox, 1, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The first two numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, x1, y1, x2, y2], where
 *           (x1, y1) and (x2, y2) are the coordinates of top-left and bottom-
 *           -right points accordingly.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
/*cnmlStatus_t cnmlCreatePluginYolov3DetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginYolov3DetectionOutputOpParam_t param,
    cnmlTensor_t *yolov3_input_tensors,
    cnmlTensor_t *yolov3_output_tensors);
*/
/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov3DetectionOutputOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[out]  outputs
 *    Output. An array stores the address of all output tensors
 *  @param[in]  num_outputs
 *    Input. Number of output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
/*cnmlStatus_t cnmlComputePluginYolov3DetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);
*/
/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov3DetectionOutputOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginYolov3DetectionOutput parameter struct pointer.
 *  @param[in]  inputs
 *    Input. An array stores the address of all cpu input data
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
/*cnmlStatus_t cnmlCpuComputePluginYolov3DetectionOutputOpForward(
    cnmlPluginYolov3DetectionOutputOpParam_t param,
    void *input[],
    void *output);
*/
/* --------------------------------------------- */
/* cnmlPluginYolov3DetectionOutout operation end */
/* --------------------------------------------- */

/* =============================================== */
/* cnmlPluginOneHot operation start */
/* =============================================== */
/*!
 *  @struct cnmlPluginOneHotOpParam
 *  @brief A struct.
 *
 *  cnmlPluginOneHotOpParam is a structure describing the "param"
 *  parameter of OneHot operation.
 *  cnmlPluginOneHotOpParam() is used to create
 *  an instance of cnmlPluginOneHotOpParam_t.
 *  cnmlPluginOneHotOpParam() is used to destroy
 *  an instance of cnmlPluginOneHotOpParam_t.
 */ 
struct cnmlPluginOneHotOpParam
{
    int N;
    int H;
    int W;
    int C;
    int depth;
    float onvalue;
    float offvalue;
	int axis;
    cnmlCoreVersion_t core_version;
};
/*! ``cnmlPluginOneHotOpParam_t`` is a pointer to a
    structure (cnmlPluginOneHotOpParam) holding the description of a OneHot operation param.
*/
typedef cnmlPluginOneHotOpParam *cnmlPluginOneHotOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginOneHotOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU270.
 *  @param[in] N
 *    Input. The number of batches.
 *  @param[in] H
 *    Input. The Height of input tensors.
 *  @param[in] W
 *    Input. The number of classes.
 *  @param[in] C
 *    Input. The number of anchors for every input tensors.
 *  @param[in] depth
 *    Input. The number of classes.
 *  @param[in] onvalue
 *    Input. The locations represented by indices take value onvalue.
 *  @param[in] offvalue
 *    Input. All other locations take value offvalue.
 *  @param[in] axis
 *    Input. The new axis is created at dimension axis.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginOneHotOpParam(
    cnmlPluginOneHotOpParam_t *param,
    cnmlCoreVersion_t core_version,
    int N,
    int H,
    int W,
    int C,
    int depth,
    float onvalue,
    float offvalue,
	int axis);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginYolov3DetectionOutputOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginYolov3DetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginOneHotOpParam(
    cnmlPluginOneHotOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginYolov3DetectionOutputOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports Caffe/Pytorch on MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginOneHot parameter struct pointer.
 *  @param[in]  yolov3_input_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, (5 + classNum) * numMaskGroup, inputH, inputW](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, 64 + 7 * numMaxBox, 1, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The first two numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, x1, y1, x2, y2], where
 *           (x1, y1) and (x2, y2) are the coordinates of top-left and bottom-
 *           -right points accordingly.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginOneHotOp(
    cnmlBaseOp_t *op,
    cnmlPluginOneHotOpParam_t param,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginOneHotOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[out]  outputs
 *    Output. An array stores the address of all output tensors
 *  @param[in]  num_outputs
 *    Input. Number of output tensors
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginOneHotOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov3DetectionOutputOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginYolov3DetectionOutput parameter struct pointer.
 *  @param[in]  inputs
 *    Input. An array stores the address of all cpu input data
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginOneHotOpForward(
   cnmlPluginOneHotOpParam_t param,
   int* indeces,
   float *dst);
/* --------------------------------------------- */
/* cnmlPluginOneHot operation end */
/* --------------------------------------------- */

/* =============================================== */
/* cnmlPluginRange operation start */
/* =============================================== */
/*!
 *  @struct cnmlPluginRangeOpParam
 *  @brief A struct.
 *
 *  cnmlPluginRangeOpParam is a structure describing the "param"
 *  parameter of Range operation.
 *  cnmlCreatePluginRangeOpParam() is used to create
 *  an instance of cnmlPluginRangeOpParam_t.
 *  cnmlDestroyPluginRangeOpParam() is used to destroy
 *  an instance of cnmlPluginOneHotOpParam_t.
 */
struct cnmlPluginRangeOpParam
{
    int size;
    cnmlCoreVersion_t core_version;
};
/*! ``cnmlPluginRangeOpParam_t`` is a pointer to a
    structure (cnmlPluginRangeOpParam) holding the description of a Range operation param.
*/
typedef cnmlPluginRangeOpParam *cnmlPluginRangeOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginRangeOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220/270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220/270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @warning
 *    The sum of input tensor HW values should be less than 32768.
 */
cnmlStatus_t cnmlCreatePluginRangeOpParam(
    cnmlPluginRangeOpParam_t *param,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginRangeOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginYolov3DetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginRangeOpParam(
    cnmlPluginRangeOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginRangeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports TensorFlow on MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYolov3DetectionOutput parameter struct pointer.
 *  @param[in]  input_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [1, 1, 1, 1](NCHW).The size of array is three, with an order of
 *           [start, limit, delta].
 *           Support only FLOAT32 dataType currently.
 *  @param[in]  outputs
 *    Output. An array of four-demensional cnmlTensors with a shape of
 *           [size, 1, 1, 1](NCHW).
 *           Support only FLOAT32 dataType currently.
 *           The size is the length of result.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginRangeOp(
    cnmlBaseOp_t *op,
    cnmlPluginRangeOpParam_t param,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRangeOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[out]  outputs
 *    Output. An array stores the address of all output tensors
 *  @param[in]  num_outputs
 *    Input. Number of output tensors
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginRangeOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRangeOp on CPU.
 *
 *  @param[in]  start
 *    Input.
 *  @param[in]  limit
 *    Input.
 *  @param[in]  delta
 *    Input.
 *  @param[out]  output
 *    Output. An address of all cpu output data
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginRangeOpForward(
   float start, float limit, float delta, float *output);
/* --------------------------------------------- */
/* cnmlPluginRange operation end */
/* --------------------------------------------- */

/* ============================================ */
/* cnmlPluginSsdDetectionOutout operation start */
/* ============================================ */
/*!
 *  @struct cnmlPluginSsdDetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginSsdDetectionOutputOpParam is a structure describing the "param"
 *  parameter of SsdDetectionOutput operation.
 *  cnmlCreatePluginSsdDetectionOutputOpParam() is used to create an instance of
 *  cnmlPluginSsdDetectionOutputOpParam_t.
 *  cnmlDestroyPluginSsdDetectionOutputOpParam() is used to destroy an instance
 *  of cnmlPluginSsdDetectionOutputOpParam_t.
 */
struct cnmlPluginSsdDetectionOutputOpParam
{
    int batchNum;
    int boxNum;
    int classNum;
    int shareLocation;
    int backgroundLabelId;
    int codeType;
    int variance_encoded_in_target;
    int clip;
    int topkNum;
    int keepNum;
    int const_prior_tensor;
    int pad_size;
    int pad_size_const;
    float confidence_thresh;
    float nms_thresh;
    cnmlCoreVersion_t core_version;
};
/*! ``cnmlPluginSsdDetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginSsdDetectionOutputOpParam) holding the description of a SsdDetectionOutput operation param.
*/
typedef cnmlPluginSsdDetectionOutputOpParam
    *cnmlPluginSsdDetectionOutputOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginSsdDetectionOutputOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official caffe website.
 *
 *  **Supports Cambricon Caffe and Cambricon Pytorch on MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of input batches.
 *           No default value, a valid batchNum must be in the range of [1, inf).
 *  @param[in] boxNum
 *    Input. The number of input boxes.
 *           No default value, a valid inputNum must be in the range of [1, inf).
 *  @param[in] classNum
 *    Input. The number of input classes.
 *           No default value, a valid classNum must be in the range of [1, 4096].
 *  @param[in] shareLocation
 *    Input. The mark of whether boxes in different classes share coordinates.
 *           Default value is 1, a valid shareLocation must be either 0 or 1.
 *  @param[in] backgroundLabelId
 *    Input. The class index of background.
 *           Default value is 0, a valid backgroundLabelId must be in the range of [0, classNum).
 *  @param[in] codeType
 *    Input. The encode type of four coordinates of boxes.
 *           Default value is CodeType_CENTER_SIZE. a valid codeType must be from enum
 *           cnmlPluginSsdCodeType_t.
 *  @param[in] variance_encoded_in_target
 *    Input. The mark of whether variance infomation has been encoded in coordinates.
 *           Default value is 0, a valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] clip
 *    Input. The mark of whether coordinates are restricted in the range of [0, 1];
 *           Default value is 1, a valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] topkNum
 *    Input. The number of topk process.
 *           No default value, a valid topkNum should be in the range of [1, boxNum).
 *  @param[in] keepNum
 *    Input. The number of boxes kept in ssd_detection op.
 *           No default value, a valid keepNum should be in the range of [1, boxNum).
 *  @param[in] const_prior_tensor
 *    Input. The mark of whether prior tensor is const tensor.
 *           Default value is 0, a valid const_prior_tensor is either 0 or 1.
 *  @param[in] pad_size
 *    Input. Padding size of boxNum.
 *           Default value is 64, a valid pad_size is divisible by 64.
 *  @param[in] pad_size_const
 *    Input. Padding size of const prior tensor.
 *           Default value is 64, a valid pad_size_const is divisible by 64.
 *  @param[in] confidence_thresh
 *    Input. Confidence score threshold used in topk process.
 *           No default value, a valid nms_thresh must be in the range of [0, 1].
 *  @param[in] nms_thresh
 *    Input. IOU threshold used in NMS function.
 *           No default value, a valid nms_thresh must be in the range of [0, 1].
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value, a valid core_version must be either MLU220 or MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginSsdDetectionOutputOpParam(
    cnmlPluginSsdDetectionOutputOpParam_t *param,
    int batchNum,
    int boxNum,
    int classNum,
    int shareLocation,
    int backgroundLabelId,
    int codeType,
    int variance_encoded_in_target,
    int clip,
    int topkNum,
    int keepNum,
    int const_prior_tensor,
    int pad_size,
    int pad_size_const,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginSsdDetectionOutputParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports Cambricon Caffe and Cambricon Pytorch on MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginSsdDetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginSsdDetectionOutputOpParam(
    cnmlPluginSsdDetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginSsdDetectionOutputOp with proper param,
 *  input, and output tensors.
 *
 *  PluginSsdDetectionOutputOp takes in feature maps and network parameters
 *  and selects valid bounding boxes based on given thresholds.
 *
 *  **Reference:**
 *    This implemenation is based on the project on ``github/weiliu89``.
 *
 *  **Formula:** This op contains three steps:
 *
 *    1. TopkByClass.
 *
 *       Filter out the boxes in every class with top k scores, and filtered
 *       boxes's scores must be larger than confidence_thresh.
 *
 *       for box_conf_idx in (0, boxNum)
 *
 *         if (box_conf(box_conf_idx) >= confidence_thresh)
 *           filtered_box_conf.push(box_conf(box_conf_idx))
 *
 *       filtered_box = topk(filtered_box_conf, topkNum)
 *
 *       box_conf is box scores, box_conf_idx is the index of box_conf.
 *
 *       confidence_thresh is the thresold of confidence scores.
 *
 *       topkNum is the left box number in topk.
 *
 *       filtered_box_conf is the filtered boxes by thresh confidence_thresh.
 *
 *       filtered_box is the filtered boxes after topk.
 *
 *    2. DecodeAllBBoxes
 *
 *       Convert input feature maps into real objectness score and coordinates.
 *
 *       for inputIdx in (0, inputNum - 1)
 *
 *          obj = sigmoid(obj_feature);
 *
 *          x   = (x_offset + sigmoid(x_feature)) / inputWs[inputIdx]
 *
 *          y   = (y_offset + sigmoid(y_feature)) / inputHs[inputIdx]
 *
 *          w   = (w_biases * exp(w_feature)) / netw
 *
 *          h   = (h_biases * exp(h_feature)) / neth
 *
 *       Obj, x_feature, y_feature, w_feature, h_feature are data from input feature maps.
 *
 *       x_offset, y_offset are the coordinates of the grid cell in the feature map.
 *
 *       w_biases, h_biases are the shape of the anchor box.
 *
 *    3. Non-maximum Suppression
 *       For each class of data, compute IOU score for every pair of bounding boxes.
 *
 *       If IOU score exceeds the IOU threshold, keep the box with larger score.
 *
 *       x1 = x - w / 2
 *
 *       y1 = y - y / 2
 *
 *       x2 = x + w / 2
 *
 *       y2 = y + y / 2
 *
 *       for classIdx in (0, classNum - 1)
 *
 *        conf = obj * probability[classIdx]
 *
 *        max, maxIdx = findMaxValueAndIndex(conf)
 *
 *        if (max >= confidence_thresh)
 *
 *          for boxIdx in (0, boxNum - 1)
 *
 *            iou = computeIOU(coord_maxIdx, coord_boxIdx)  // where "coords" means x1,y1,x2,y2
 *
 *            if (iou < nms_thresh)
 *
 *              keep coords and conf for boxIdx
 *
 *    4. KeepOnlyTopKResults
 *
 *       Filter out the boxes in all classes with top k scores.
 *
 *       filtered_box = topk(filtered_box_conf, keepNum)
 *
 *       keepNum is the left box number in topk.
 *
 *       filtered_box_conf is the box confidence of all boxes after step Non-maximum Suppression.
 *
 *       filtered_box is the left box after KeepOnlyTopKResults.
 *  **DataType:**
 *
 *    Support half(float16) and float32 type for both input and output tensors.
 *
 *  **Performance Optimization:**
 *
 *    The performance of detection layer depends on both the data size and
 *    the value. This op achieves relatively better performance
 *    when all following conditions are met:
 *
 *    - topkNum is 64-aligned and less or equal to 512.
 *
 *    - keepNum is 64-aligned and less or equal to 512.
 *
 *  **Supports MLU220/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginSsdDetectionOutput parameter struct pointer.
 *  @param[in]  ssd_input_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, (5 + classNum) * numMaskGroup, inputH, inputW](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  ssd_output_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, 64 + 7 * numMaxBox, 1, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The first two numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, x1, y1, x2, y2], where
 *           (x1, y1) and (x2, y2) are the coordinates of top-left and bottom-
 *           -right points accordingly.
 *  @param[in]  ssd_static_tensors
 *    Input. An array of prior tensors when CONST_PRIOR_TENSOR is set true.
 *           Otherwise just pass nullptr.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginSsdDetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginSsdDetectionOutputOpParam_t param,
    cnmlTensor_t *ssd_input_tensors,
    cnmlTensor_t *ssd_output_tensor,
    cnmlTensor_t *ssd_static_tensor);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginSsdDetectionOutputOp on MLU.
 *
 *  **Supports Cambricon Caffe and Cambricon Pytorch on MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  num_outputs
 *    Input. Number of output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginSsdDetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginSsdDetectionOutputOp on CPU.
 *
 *  **Supports Cambricon Caffe and Cambricon Pytorch on MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A PluginSsdDetectionOutput parameter struct pointer.
 *  @param[in]  loc_data
 *    Input. An array stores the bbox location data with a shape of [N C H W].
 *  @param[in]  conf_data
 *    Input. An array stores the bbox confidence data with a shape of [N C H W].
 *  @param[in]  pri_data
 *    Input. An array stores the prior bbox location/variance data with a shape
 *    of [N C H W].
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data with a shape
 *    of 1 + [N H W 7]. The first number is the number of detected bboxes. The
 *    rest stores the bbox info with an order:
 *    [batchId, classId, score, x1, y1, x2, y2], where (x1, y1) and (x2, y2)
 *    are the coordinates of top-left and bottom-right points accordingly.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginSsdDetectionOutputOpForward(
    cnmlPluginSsdDetectionOutputOpParam_t param,
    void *loc_data,
    void *conf_data,
    void *pri_data,
    void *output);
/* ------------------------------------------ */
/* cnmlPluginSsdDetectionOutout operation end */
/* ------------------------------------------ */

/* ========================================= */
/* cnmlPluginAnchorGenerator operation start */
/* ========================================= */
/*!
 *  @struct cnmlPluginAnchorGeneratorOpParam
 *  @brief A struct.
 *
 *  cnmlPluginAnchorGeneratorOpParam is a structure describing the "param"
 *  parameter of cnmlPluginAnchorGenerator operation.
 *  cnmlCreatePluginAnchorGereratorOpParam() is used to create an instance of
 *  cnmlPluginAnchorGeneratorOpParam_t.
 *  cnmlDestroyPluginAnchorGereratorOpParam() is used to destroy an instance
 *  of cnmlPluginAnchorGeneratorOpParam_t.
 */
struct cnmlPluginAnchorGeneratorOpParam
{
    vector<float> scales;
    vector<float> aspect_ratios;
    vector<float> base_anchor_sizes;
    vector<float> anchor_strides;
    vector<float> anchor_offsets;
    vector<int> image_shape;
    bool corner_bbox;
    bool clip_window;
    bool normalize;
    bool x_before;
    int channel;
    int align_channel;
    int grid_height;
    int grid_width;
    vector<cnmlTensor_t> mlu_tensors;
    vector<cnmlCpuTensor_t> cpu_tensors;
    vector<void *> cpu_ptrs;
};
/*! ``cnmlPluginAnchorGeneratorOpParam_t`` is a pointer to a
    structure (cnmlPluginAnchorGeneratorOpParam) holding the description of a AnchorGenerator operation param.
*/
typedef struct cnmlPluginAnchorGeneratorOpParam *cnmlPluginAnchorGeneratorOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates PluginAnchorGeneratorOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  feature_map_shape_mlu_tensor
 *    Input. A cnmlTensors with a shape of [1, 2, 1, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  grid_anchors_mlu_tensor
 *    Input. A cnmlTensors with a shape of
 *           [1, len(scales) * len(aspect_ratios) * 4,
 *           featuremap_height, featuremap_width](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  param
 *    Input. A PluginAnchorGenerator parameter struct pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
cnmlStatus_t cnmlCreatePluginAnchorGeneratorOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginAnchorGeneratorOpParam_t param,
    cnmlTensor_t feature_map_shape_mlu_tensor,
    cnmlTensor_t grid_anchors_mlu_tensor
    );

/*!
 *  @brief A function.
 *
 *  This function forwards PluginAnchorGeneratorOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A PluginAnchorGenerator parameter struct pointer.
 *  @param[out]  anchors
 *    Output. The address cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
void cnmlCpuComputePluginAnchorGeneratorOpForward(
    cnmlPluginAnchorGeneratorOpParam_t param,
    float *anchors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginAnchorGeneratorOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  feature_map_shape_mlu
 *    Input. An address of input tensors
 *  @param[in]  grid_anchors_mlu
 *    Input. An address of output tensors
 *  @param[in]  forward_param
 *    Input. Which records runtime degree of data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
cnmlStatus_t cnmlComputePluginAnchorGeneratorOpForward(
    cnmlBaseOp_t op,
    void *feature_map_shape_mlu,
    void *grid_anchors_mlu,
    cnrtInvokeFuncParam_t forward_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginSsdDetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] scales
 *    Input. The scales.
 *  @param[in] aspect_ratios
 *    Input. The aspect ratios.
 *  @param[in] base_anchor_sizes
 *    Input. The base anchor sizes.
 *  @param[in] anchor_strides
 *    Input. The strides of anchor.
 *  @param[in] anchor_offsets
 *    Input. The offsets of anchor.
 *  @param[in] image_shape
 *    Input. The shape of image.
 *  @param[in] corner_bbox
 *    Input. If ture, the anchor box will be like [x1, y1, x2, y2], else [xc, yc, w, h].
 *  @param[in] clip_window
 *    Input. If true, the anchor will be limited to [0, image_shape].
 *  @param[in] normalize.
 *    Input. If true, the anchor box will be normalized to 0 - 1.
 *  @param[in] x_before
 *    Input. If true, the anchor box will be like [x1, y1, x2, y2] or [xc, yc, w, h], else [y1, x1, y2, x2] or [yc, xc, h, w].
 *  @param[in] grid_height
 *    Input. The height of the grid.
 *  @param[in] grid_width
 *    Input. The width of the grid.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginAnchorGereratorOpParam(
    cnmlPluginAnchorGeneratorOpParam_t *param_ptr,
    vector<float> scales,
    vector<float> aspect_ratios,
    vector<float> base_anchor_sizes,
    vector<float> anchor_strides,
    vector<float> anchor_offsets,
    vector<int> image_shape,
    bool corner_bbox,
    bool clip_window,
    bool normalize,
    bool x_before,
    int channel,
    int grid_height,
    int grid_width);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginAnchorGereratorParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginSsdDetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
cnmlStatus_t cnmlDestroyPluginAnchorGereratorOpParam(
    cnmlPluginAnchorGeneratorOpParam_t param);
/* --------------------------------------- */
/* cnmlPluginAnchorGenerator operation end */
/* --------------------------------------- */


/* ================================= */
/* cnmlPluginSBC operation start */
/* ================================= */

struct cnmlPluginSBCOpParam
{
    int batch_num_;
};
typedef cnmlPluginSBCOpParam* cnmlPluginSBCOpParam_t;

cnmlStatus_t cnmlCreatPluginSBCOpParam(
    cnmlPluginSBCOpParam_t *param,
    int batch_num_
);

cnmlStatus_t cnmlDestroyPluginSBCOpParam(
    cnmlPluginSBCOpParam_t *param
);

cnmlStatus_t cnmlCreatePluginSBCOp(
    cnmlBaseOp_t *op,
    cnmlTensor_t *SBC_input_tensors,
    cnmlTensor_t *SBC_output_tensors,
    int batch_num_
);

cnmlStatus_t cnmlComputePluginSBCOpForward(
    cnmlBaseOp_t op,
    void **inputs,
    int input_num,
    void **outputs,
    int output_num,
    cnrtQueue_t queue
);

/* ------------------------------- */
/* cnmlPluginSBC operation end */
/* ------------------------------- */





/* ================================= */
/* cnmlPluginRoiPool operation start */
/* ================================= */
/*!
 *  @struct cnmlPluginRoiPoolOpParam
 *  @brief A struct.
 *
 *  cnmlPluginRoiPoolOpParam is a structure describing the "param"
 *  parameter of RoiPool operation.
 *  cnmlCreatePluginRoiPoolOpParam() is used to create
 *  an instance of cnmlPluginRoiPoolOpParam_t.
 *  cnmlDestroyPluginRoiPoolOpParam() is used to destroy
 *  an instance of cnmlPluginRoiPoolOpParam_t.
 */
struct cnmlPluginRoiPoolOpParam
{
    int channels;
    int height;
    int width;
    int pooled_height;
    int pooled_width;
    int rois_num;
    int roi_cols;
    int batch_size;
    float spatial_scale;
    int int8_mode;
    cnmlCoreVersion_t coreVersion;
};
/*! ``cnmlPluginRoiPoolOpParam_t`` is a pointer to a
    structure (cnmlPluginRoiPoolOpParam) holding the description of a RoiPool operation param.
*/
typedef cnmlPluginRoiPoolOpParam *cnmlPluginRoiPoolOpParam_t;

/*!
 *  @brief cnmlCreatePluginRoiPoolOpParam.
 *
 *  This function creates a RoiPoolOp param object with the pointer
 *  and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] channels
 *    Input. The number of channels.
 *  @param[in] height
 *    Input. The number of height of bottom date.
 *  @param[in] width
 *    Input. The number of width of bottom_data.
 *  @param[in] pooled_height
 *    Input. The number of height after pooling.
 *  @param[in] pooled_width
 *    Input. The number of width after pooling.
 *  @param[in] rois_num
 *    Input. The number of rois.
 *  @param[in] roi_cols
 *    Input. The size of one roi.
 *  @param[in] batch_size
 *    Input. The number of batches.
 *  @param[in] spatial_scale
 *    Input. The scaling ratio.
 *  @param[in] int8_mode
 *    Input. Whether the datatype of input is int8.
 *  @param[in] coreVersion
 *    Input. Supported core version, including MLU100/270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 */
cnmlStatus_t cnmlCreatePluginRoiPoolOpParam(
    cnmlPluginRoiPoolOpParam_t *param,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int rois_num,
    int roi_cols,
    int batch_size,
    float spatial_scale,
    int int8_mode,
    cnmlCoreVersion_t coreVersion);

/*!
 *  @brief cnmlDestroyPluginRoiPoolOpParam.
 *
 *  This function frees the PluginRoiPoolOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in] param
 *    Input. A pointer to the address of the struct of computation parameters
 *           for PluginRoiPool operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginRoiPoolOpParam(
    cnmlPluginRoiPoolOpParam_t *param);

/*!
 *  @brief cnmlCreatePluginRoiPoolOp.
 *
 *  This function creates PluginRoiPoolOp with proper param, input,
 *  and output tensors.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginRoiPool parameter struct pointer.
 *  @param[in] roiPool_input_tensors
 *    Input. An array of two four-demensional cnmlTensors. One is with the shape
 *           of [batch_size, channels, height, width](NCHW), and the other
 *           is with the shape of [batch_size, rois_num, 1, roi_cols](NCHW).
 *           Support FLOAT32 and FLOAT16 datatype.
 *  @param[in] roiPool_output_tensors
 *    Input. An array of a four-demensional cnmlTensor with a shape of
 *           [batch_size * rois_num, channels, pooled_height, pooled_width](NCHW).
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The operator pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlCreatePluginRoiPoolOp(
    cnmlBaseOp_t *op,
    cnmlPluginRoiPoolOpParam_t param,
    cnmlTensor_t *roiPool_input_tensors,
    cnmlTensor_t *roiPool_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRoiPoolOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensor.
 *  @param[in] input_num
 *    Input. The number of input tensors.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensor.
 *  @param[in] output_num
 *    Input. The number of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The operator pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlComputePluginRoiPoolOpForward(
    cnmlBaseOp_t op,
    void **inputs,
    int input_num,
    void **outputs,
    int output_num,
    cnrtQueue_t queue);

/*!
 *  @brief cnmlCpuComputePluginRoiPoolOpForward.
 *
 *  This function forwards PluginRoiPoolOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in] param
 *    Input. A PluginRoiPool parameter struct pointer.
 *  @param[in] input_data
 *    Input. Cpu input bottom data.
 *  @param[in] input_rois
 *    Input. Cpu input bottom rois.
 *  @param[out] output_data
 *    Output. Cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The param pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlCpuComputePluginRoiPoolOpForward(
    cnmlPluginRoiPoolOpParam_t param,
    float *input_data,
    float *input_rois,
    float *output_data);
/* ------------------------------- */
/* cnmlPluginRoiPool operation end */
/* ------------------------------- */








/* ================================== */
/* cnmlPluginProposal operation start */
/* ================================== */
/*!
 *  @struct cnmlPluginProposalOpParam
 *  @brief A struct.
 *
 *  cnmlPluginProposalOpParam is a structure describing the "param"
 *  parameter of Proposal operation, used to create proposal operation.
 */
struct cnmlPluginProposalOpParam
{
    cnmlTensor_t *cnml_static_tensors;
    cnmlCpuTensor_t *cpu_static_tensors;
    int batch_size;
    int height;
    int width;
    int anchor_num;
    int nms_num;
    int top_thresh;
    float im_min_h;
    float im_min_w;
    float nms_scale;
    float stride;
    float nms_thresh;
    float im_h;
    float im_w;
    float scale;
    int fix8;
    cnmlCoreVersion_t core_version;
    float *anchor;
};
/*! ``cnmlPluginProposalOpParam_t`` is a pointer to a
    structure (cnmlPluginProposalOpParam) holding the description of a Proposal operation param.
*/
typedef cnmlPluginProposalOpParam *cnmlPluginProposalOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a ProposalOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batch_size
 *    Input. Batch_size of input images for network.
 *  @param[in] height
 *    Input. Height of the feature map.
 *  @param[in] width
 *    Input. Width of the feature map.
 *  @param[in] anchor_num
 *    Input. Number of anchors of every point in the feature map.
 *  @param[in] nms_num
 *    Input. Number of boxes to be select in nms process.
 *  @param[in] top_thresh
 *    Input. Number of boxes selected in TopK process.
 *  @param[in] im_min_h
 *    Input. The minimum size of height for boxes selected.
 *  @param[in] im_min_w
 *    Input. The minimum size of width for boxes selected.
 *  @param[in] nms_scale
 *    Input. The scaling rate of boxes when computing areas of box in nms process.
 *  @param[in] stride
 *    Input. Stride in computing anchor. Unused.
 *  @param[in] nms_thresh
 *    Input. Threshold of IOU in nms process.
 *  @param[in] im_h
 *    Input. Heigth of input images for network.
 *  @param[in] im_w
 *    Input. Width of input images for network.
 *  @param[in] scale
 *    Input. The scaling rate of the size of input images.
 *  @param[in] fix8
 *    Input. Type of input. Uenused.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU100/220/270.
 *  @param[in] anchor
 *    Input. The anchor of boxes' coordinates.
 *   @retval CNML_STATUS_SUCCESS
 *     The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginProposalOpParam(
    cnmlPluginProposalOpParam_t *param,
    int batch_size,
    int height,
    int width,
    int anchor_num,
    int nms_num,
    int top_thresh,
    float im_min_h,
    float im_min_w,
    float nms_scale,
    float stride,
    float nms_thresh,
    float im_h,
    float im_w,
    float scale,
    int fix8,
    cnmlCoreVersion_t core_version,
    float *anchor);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginProposalParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginProposal operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginProposalOpParam(
    cnmlPluginProposalOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginProposalOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginProposal parameter struct pointer.
 *  @param[in] proposal_input_tensors
 *    Input. This pointer contains two array of four-demensional cnmlTensors,
 *           first tensor's shape is [barch_size, 4, 1,
 *           anchor_num * width * height](NHWC), second tensor's shape is
 *           [batchNum, 2, 1, anchor_num * width * height](NHWC).
 *           Support only FLOAT16 dataType currently.
 *  @param[in] proposal_output_tensors
 *    Input. This pointer contains an array of four_demensional cnmlTemsors
 *           with a shape of [batch_size, 5, 1, nms_num](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginProposalOp(
    cnmlBaseOp_t *op,
    cnmlPluginProposalOpParam_t param,
    cnmlTensor_t *proposal_input_tensors,
    cnmlTensor_t *proposal_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginProposalOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  num_outputs
 *    Input. Number of output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *           data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginProposalOpForward(
    cnmlBaseOp_t op,
    void *inputs[],
    int num_inputs,
    void *outputs[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginProposalOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A PluginProposal parameter struct pointer.
 *  @param[in]  inputs
 *    Input. An array stores the address of all cpu input data
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginProposalOpForward(
    cnmlPluginProposalOpParam_t param,
    void *input[],
    void *output);
/* ----------------------------------------------- */
/* cnmlPluginCommonProposalC20Outout operation end */
/* ----------------------------------------------- */

/* ======================================== */
/* cnmlPluginFasterrcnnPost operation start */
/* ======================================== */
/*!
 *  @struct cnmlPluginFasterrcnnPostOpParam
 *  @brief A struct.
 *
 *  cnmlCreatePluginFasterrcnnPostOpParam() is used to create
 *  an instance of cnmlPluginFasterrcnnPostOpParam_t.
 *  cnmlDestroyPluginFasterrcnnPostOpParam() is used to destroy
 *  an instance of cnmlPluginFasterrcnnPostOpParam_t.
 */
struct cnmlPluginFasterrcnnPostOpParam;
/*! ``cnmlPluginFasterrcnnPostOpParam_t`` is a pointer to a
    structure (cnmlPluginFasterrcnnPostOpParam) holding the description of a FasterrcnnPost operation param.
*/
typedef struct cnmlPluginFasterrcnnPostOpParam
*cnmlPluginFasterrcnnPostOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a FasterrcnnPostOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**  Faster RCNN model
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in] num_proposal_mlu_tensor.
 *    Input. the number of proposed boxes in each batch;
 *    a four-dimension cnmlTensor_t which shape is [batchNum,1,1, 1](HCHW);
 *  @param[in] box_encoding_mlu_tensor.
 *    Input. The box_encoding from second stage neural network;
 *    a four-dimension cnmlTensor_t which shape is [batchNum,box_align_size, class_align_size, 4](HCHW);
 *  @param[in] class_predictions_mlu_tensor.
 *    Input. The class predicting logit from second stage network;
 *    a four-dimension cnmlTensor_t which shape is [batchNum,max_num_proposals, 1, class_align_size](HCHW);
 *  @param[in] box_proposal_mlu_tensor.
 *    Input. The predicting boxes from region proposal network;
 *    a four-dimension cnmlTensor_t which shape is [batchNum, box_align_size, 1, 4](HCHW);
 *  @param[in] detection_result_tensor.
 *    Input. Bounding boxes params, classIdx, score, x1, y1, x2, y2, and etc;
 *    a four-dimension cnmlTensor_t which shape is [batchNum, output_align_size, 1, 6](HCHW);
 *  @param[in]  param
 *    Input. A PluginFasterrcnnPost parameter struct pointer.
 */
cnmlStatus_t cnmlCreatePluginFasterrcnnPostOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginFasterrcnnPostOpParam_t param,
    cnmlTensor_t num_proposal_mlu_tensor,
    cnmlTensor_t box_encoding_mlu_tensor,
    cnmlTensor_t class_predictions_mlu_tensor,
    cnmlTensor_t box_proposal_mlu_tensor,
    cnmlTensor_t detection_result_tensor);

/*!
 *  @brief A function.
 *
 *  This function creates PluginFasterrcnnPostOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270**  Faster RCNN model
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] box_encoding_count
 *    Input.  count of box_encoding
 *  @param[in] box_proposal_count
 *    Input. count of box_proposal
 *  @param[in] class_logit_count
 *    Input. count of class_logit
 *  @param[in] align_class_logit_count
 *    Input. count of align_class_logit
 *  @param[in] batch_size
 *    Input. Batch size of this neural network.
 *  @param[in] num_classes
 *    Input. Number of objects's category.
 *  @param[in] score_thresh
 *    Input. The minimal threshold for marking a box as an object.
 *  @param[in] iou_thresh
 *    Input. The minimal threshold for marking a box as a duplicate.
 *  @param[in] max_size_per_class
 *    Input. The number of boxes kept in each class.
 *  @param[in] max_total_size
 *    Input. The total number of boxes kept finally.
 *  @param[in] max_num_proposals
 *    Input. The maximum number of region proposals kept in the first stage
 *  @param[in] scale_x
 *    Input. The box decoded factor of coordinate of center point
 *  @param[in] scale_y
 *    Input. The box decoded factor of height and width of box
 *  @param[in] int8mode
 *    Input. whether this op is in int8 mode
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 */
cnmlStatus_t cnmlCreatePluginFasterrcnnPostOpParam(
    cnmlPluginFasterrcnnPostOpParam_t *param,
    int box_encoding_count,
    int box_proposal_count,
    int class_logit_count,
    int align_class_logit_count,
    int batch_size,
    int num_classes,
    float score_threshold,
    float iou_threshold,
    int max_size_per_class,
    int max_total_size,
    int max_num_proposals,
    float scale_x,
    float scale_y,
    int int8mode);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginFasterrcnnPostOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginFasterrcnnPost operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginFasterrcnnPostOpParam(
    cnmlPluginFasterrcnnPostOpParam_t *param_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginFasterrcnnPostOp on MLU.
 *
 *  **Supports MLU100/MLU270** Faster RCNN model
 *
 *  @param[in] op
 *    Input. A pointer to the base operator address.
 *  @param[in] num_proposal_mlu
 *    Input. mlu pointer of num_proposal
 *  @param[in] box_encoding_mlu
 *    Input. mlu pointer of box_encoding
 *  @param[in] class_predictions_mlu
 *    Input. mlu pointer of  class_predictions
 *  @param[in] box_proposal_mlu
 *    Input. mlu pointer of box_proposal
 *  @param[out] detection_result
 *    Output. mlu pointer of output
 *  @param[in]  forward_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    - op is a null pointer.
 *    - The pointer content pointed by op is already freed.
 */
cnmlStatus_t cnmlComputePluginFasterrcnnPostOpForward(
    cnmlBaseOp_t op,
    void *num_proposal_mlu,
    void *box_encoding_mlu,
    void *class_predictions_mlu,
    void *box_proposal_mlu,
    void *detection_result,
    cnrtInvokeFuncParam_t *forward_param,
    cnrtQueue_t queue);


/*!
 *  @brief A function.
 *  *  This function forwards PluginFasterrcnnPostOp on CPU.
 *
 *  @param[in]  num_proposal_cpu_tensor
 *    Input. cpu tensor of  num_proposal input
 *  @param[in]  num_proposal
 *    Input. cpu pointer of num_proposal input
 *  @param[in]  box_encoding_cpu_tensor
 *    Input.  cpu tensor of  box_encoding input
 *  @param[in]  box_encoding
 *    Input. cpu pointer of box_encoding input
 *  @param[in]  box_proposal_cpu_tensor
 *    Input.  cpu tensor of  box_proposal input
 *  @param[in]  box_proposal
 *    Input. cpu pointer of box_proposal input
 *  @param[in]  class_predictions_cpu_tensor
 *    Input.  cpu tensor of  class_predictions input
 *  @param[in]  class_predictions
 *    Input. cpu pointer of class_predictions input
 *  @param[in]  true_image_shape_cpu_tensor
 *    Input.  cpu tensor of  true_image_shape input
 *  @param[in]  true_image_shape
 *    Input. cpu pointer of true_image_shape input
 *  @param[in]  output_result_cpu_tensor
 *    Input.  cpu tensor of  output_result input
 *  @param[out]  output_result_cpu
 *    Output. cpu pointer of output_result output
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */

cnmlStatus_t cnmlCpuComputePluginFasterrcnnPostOpForward(
    cnmlCpuTensor_t num_proposal_cpu_tensor,
    const float *num_proposal,
    cnmlCpuTensor_t box_encoding_cpu_tensor,
    const float *box_encoding,
    cnmlCpuTensor_t box_proposal_cpu_tensor,
    const float *box_proposal,
    cnmlCpuTensor_t class_predictions_cpu_tensor,
    const float *class_predictions,
    cnmlCpuTensor_t true_image_shape_cpu_tensor,
    const float *true_image_shape,
    cnmlCpuTensor_t output_result_cpu_tensor,
    float *output_result_cpu);
/* -------------------------------------- */
/* cnmlPluginFasterrcnnPost operation end */
/* -------------------------------------- */

/* ================================= */
/* cnmlPluginSsdPost operation start */
/* ================================= */
/*!
 *  @struct cnmlPluginSsdPostOpParam
 *  @brief A struct.
 *
 *  cnmlPluginSsdPost Param is a structure describing the "param"
 *  parameter of SSD postprocess operation, used to create ssd postprocess
 *  operation.
 *
 *  cnmlCreatePluginSsdPostOpParam() is used to create
 *  an instance of cnmlPluginSsdPostOpParam_t.
 *  cnmlDestroyPluginSsdPostOpParam() is used to destroy
 *  an instance of cnmlPluginSsdPostOpParam_t.
 *
 */
struct cnmlPluginSsdPostOpParam;
/*! ``cnmlPluginSsdPostOpParam_t`` is a pointer to a
    structure (cnmlPluginSsdPostOpParam) holding the description of a SsdPost operation param.
*/
typedef struct cnmlPluginSsdPostOpParam *cnmlPluginSsdPostOpParam_t;
/*!
 *  @brief A function.
 *
 *  This function creates PluginSsdPostOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270** SSD series model
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginSsdPost parameter struct pointer.
 *  @param[in] detection_result_tensor
 *    Input[in]. A four-demensional cnmlTensors which shape is
 *        [batchNum,max_toal_size_align,1,6](NCHW);
 *  @param[in] image_shape_mlu_tensor
 *    Input[in]. A four-demensional cnmlTensors which shape is
 *        [batchNum,3,1,1](NCHW);
 *  @param[in] box_encoding_mlu_tensor
 *    Input[in]. A four-demensional cnmlTensors which shape is
 *        [batchNum,max_num_proposals_align,1,4](NCHW)
 *  @param[in] anchor_mlu_tensor
 *    Input[in]. A four-demensional cnmlTensors which shape is
 *        [batchNum,max_num_proposals_align,1,4](NCHW)
 *  @param[in] class_predictions_mlu_tensor
 *    Input[in]. A four-demensional cnmlTensors which shape is
 *        [batchNum,num_classes_align,1,max_num_proposals](NCHW)
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginSsdPostOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginSsdPostOpParam_t param,
    cnmlTensor_t detection_result_tensor,
    cnmlTensor_t image_shape_mlu_tensor,
    cnmlTensor_t box_encoding_mlu_tensor,
    cnmlTensor_t anchor_mlu_tensor,
    cnmlTensor_t class_predictions_mlu_tensor);


/*!
 *  @brief A function.
 *
 *  This function forwards PluginSsdPostOp on MLU.
 *
 *  **Supports MLU100/MLU270** SSD series model
 *  @param[int]  op
 *    int. A pointer to the base operator address.
 *  @param[in] image_shape_mlu.
 *    Input. mlu pointer of image_shape input
 *  @param[in] box_encoding_mlu.
 *    Input. mlu pointer of box_encoding input
 *  @param[in] anchor_mlu.
 *    Input. mlu pointer of anchor input
 *  @param[in] class_predictions_mlu.
 *    Input. mlu pointer of class_predictions input
 *  @param[out] detection_result.
 *    Output. mlu pointer of output
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - op is a null pointer.
 *    - The pointer content pointed by op is already freed.
 */

cnmlStatus_t cnmlComputePluginSsdPostOpForward(
    cnmlBaseOp_t op,
    void *image_shape_mlu,
    void *box_encoding_mlu,
    void *anchor_mlu,
    void *class_predictions_mlu,
    void *detection_result,
    cnrtInvokeFuncParam_t forward_param,
    cnrtQueue_t queue);


/*!
 *  @brief A function.
 *
 *  This function creates a SsdPostOp param object with
 *  the pointer and parameters provided by user.
 *  **Supports MLU100/MLU270** SSD series model
 *  @param[out]  param
 *    Output. A PluginSsdPost parameter struct pointer.
 *  @param[in] box_encoding_count.
 *    Input. count of box_encoding
 *  @param[in] image_shape_count.
 *    Input. count of image_shape
 *  @param[in] class_predictions_count.
 *    Input. count of class_prediction
 *  @param[in] anchor_count.
 *    Input. count of anchor
 *  @param[in] batch_size
 *    Input. Batch size of this neural network.
 *  @param[in] num_classes
 *    Input. Number of objects's category.
 *  @param[in] score_thresh
 *    Input. The minimal threshold for marking a box as an object.
 *  @param[in] iou_thresh
 *    Input. The minimal threshold for marking a box as a duplicate.
 *  @param[in] max_size_per_class
 *    Input. The number of boxes kept in each class.
 *  @param[in] max_total_size
 *    Input. The total number of boxes kept finally.
 *  @param[in] anchor_num
 *    Input. The maximum number of anchor_num
 *  @param[in] scale_x
 *    Input. The box decoded factor of coordinate of center point
 *  @param[in] scale_y
 *    Input. The box decoded factor of height and width of box
 *  @param[in] int8mode
 *    Input. whether this op is in int8 mode
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
cnmlStatus_t cnmlCreatePluginSsdPostOpParam(
    cnmlPluginSsdPostOpParam_t *param_ptr,
    int box_encoding_count,
    int image_shape_count,
    int class_predictions_count,
    int anchor_count,
    int batch_size,
    int num_classes,
    float score_threshold,
    float iou_threshold,
    int max_size_per_class,
    int max_total_size,
    int max_num_proposals,
    float scale_x,
    float scale_y,
    int int8mode);



/*!
 *  @brief A function.
 *
 *  This function frees the PluginSsdPostParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270** SSD series model
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginProposal operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
cnmlStatus_t cnmlDestroyPluginSsdPostOpParam(
    cnmlPluginSsdPostOpParam_t *param_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginSsdPostOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginSsdPost parameter struct pointer.
 *  @param[in] boxes_cpu_tensor.
 *    Input. cpu tensor of boxes input
 *  @param[in] boxes.
 *    Input. cpu pointer of boxes input
 *  @param[in] scores_cpu_tensor.
 *    Input. cpu tensor of scores input
 *  @param[in] scores.
 *    Input. cpu pointer of scores input
 *  @param[in] anchors_cpu_tensor.
 *    Input. cpu tensor of anchors input
 *  @param[in] anchors.
 *    Input. cpu pointer of anchors input
 *  @param[in] output_cpu_tensor.
 *    Input. cpu tensor of output
 *  @param[out] detection_output_cpu.
 *    Output. cpu pointer of output
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
cnmlStatus_t cnmlCpuComputePluginSsdPostOpForward(
    cnmlPluginSsdPostOpParam_t param,
    const float *boxes,
    const float *scores,
    const float *anchors,
    float *detection_output_cpu);
/* ------------------------------- */
/* cnmlPluginSsdPost operation end */
/* ------------------------------- */

/* ================================== */
/* cnmlPluginAddpadOp operation start */
/* ================================== */
/*!
 *  @struct cnmlPluginAddpadOpParam
 *  @brief A struct.
 *
 *  cnmlPluginAddpadOpParam is a structure describing the "param"
 *  parameter of cnmlPluginAddpadOp operation.
 *  cnmlCreatePluginAddpadOpParam() is used to create an instance of
 *  cnmlPluginAddpadOpParam_t.
 *  cnmlDestroyPluginAddpadOpParam() is used to destroy an instance
 *  of cnmlPluginAddpadOpParam_t.
 */
struct cnmlPluginAddpadOpParam;
/*! ``cnmlPluginAddpadOpParam_t`` is a pointer to a
    structure (cnmlPluginAddpadOpParam) holding the description of a Addpad operation param.
*/
typedef cnmlPluginAddpadOpParam * cnmlPluginAddpadOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a cnmlPluginAddpadOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batch_size
 *    Input. The number of batches.
 *  @param[in] src_h
 *    Input. Height of input image.
 *  @param[in] src_w
 *    Input. Width of input image.
 *  @param[in] dst_h
 *    Input. Height of output image.
 *  @param[in] dst_w
 *    Input. Width of output image.
 *  @param[in] type_uint8
 *    Input. input data type is uint8_t or not.
 *  @param[in] type_yuv
 *    Input. input image is yuv 420SP or not
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginAddpadOpParam(
    cnmlPluginAddpadOpParam_t *param_ptr,
    int batch_size,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w,
    int type_uint8,
    int type_yuv);

/*!
 *  @brief A function.
 *
 *  This function frees the cnmlPluginAddpadOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for cnmlPluginAddpadOp operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginAddpadOpParam(
    cnmlPluginAddpadOpParam_t param);

/*!
 *  @brief A function.
 *
 *  This function forwards cnmlPluginAddpadOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  mlu_input_ptr
 *    Input. mlu address of input image pointer
 *  @param[in]  mlu_padValue_ptr
 *    Input. mlu address of pad value pointer
 *  @param[in]  mlu_dst_ptr
 *    Input. mlu address of output image pointer
 *  @param[in]  compute_forward_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginAddpadOpForward(
    cnmlBaseOp_t op,
    void *mlu_input_ptr,
    void *mlu_padValue_ptr,
    void *mlu_dst_ptr,
    cnrtInvokeFuncParam_t compute_forward_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards cnmlPluginAddpadOp on CPU.
 *
 *  @param[in]  param
 *    Input. A cnmlPluginAddpadOp parameter struct pointer.
 *  @param[in]  src_cpu_ptr
 *    Input. cpu address of input image
 *  @param[in]  padValue_cpu_ptr
 *    Input. cpu address of pad value
 *  @param[in]  dst_cpu_ptr
 *    Input. cpu address of output image
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginAddpadOpForwad(
    cnmlPluginAddpadOpParam_t param,
    uint8_t *src_cpu_ptr,
    uint8_t *padValue_cpu_ptr,
    uint8_t *dst_cpu_ptr
);

/*!
 *  @brief A function.
 *
 *  This function creates cnmlPluginAddpadOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A cnmlPluginAddpadOp parameter struct pointer.
 *  @param[in]  dst_tensor
 *    Input. A four-demensional cnmlTensors with a shape of
 *           [batchNum, 1 or 4, src_h, src_w](NCHW).
 *           Support only UINT8 dataType currently.
 *  @param[in]  src_tensor
 *    Input. A four-demensional cnmlTensors with a shape of
 *           [batchNum, 1 or 4, dst_h, dst_w](NCHW).
 *           Support only UINT8 dataType currently.
 *  @param[in]  dst_tensor
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [1, 3, 1, 1](NCHW).
 *           Support only UINT8 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginAddpadOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginAddpadOpParam_t param,
    cnmlTensor_t dst_tensor,
    cnmlTensor_t src_tensor,
    cnmlTensor_t value_tensor);

/* -------------------------------- */
/* cnmlPluginAddpadOp operation end */
/* -------------------------------- */

/* ======================================== */
/* cnmlPluginPostProcessRpn operation start */
/* ======================================== */
/*!
 *  @struct cnmlPluginPostProcessRpnOpParam
 *  @brief A struct.
 *
 *  cnmlPluginPostProcessRpnOpParam is a structure describing the "param"
 *  parameter of cnmlPluginPostProcessRpnOp operation.
 *  cnmlCreatePluginPostProcessRpnOpParam() is used to create an instance of
 *  cnmlPluginPostProcessRpnOpParam_t.
 *  cnmlDestroyPluginPostProcessRpnOpParam() is used to destroy an instance
 *  of cnmlPluginPostProcessRpnOpParam_t.
 */
struct cnmlPluginPostProcessRpnOpParam;
/*! ``cnmlPluginPostProcessRpnOpParam_t`` is a pointer to a
    structure (cnmlPluginPostProcessRpnOpParam) holding the description of a PostProcessRpn operation param.
*/
typedef cnmlPluginPostProcessRpnOpParam * cnmlPluginPostProcessRpnOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a cnmlPluginPostProcessRpnOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  batch_size
 *    Input. Size of batch.
 *  @param[in]  num_anchors
 *    Input. Number of anchors.
 *  @param[in]  max_nms_out
 *    Input. The max number of outputs of nms.
 *  @param[in]  iou_thresh
 *    Input. The thresh of iou when computing nms.
 *  @param[in]  im_height
 *    Input. The height of image.
 *  @param[in]  im_width
 *    Input. The width of image.
 *  @param[in]  scaled_xy
 *    Input. Coefficient used in bounding-box regression.
 *  @param[in]  scaled_wh
 *    Input. Coefficient used in bounding-box regression.
  *  @param[in]  min_nms_score
 *    Input. min score for bbox going to nms.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginPostProcessRpnOpParam(
    cnmlPluginPostProcessRpnOpParam_t *param_ptr,
    int batch_size,
    int num_anchors,
    int max_nms_out,
    float iou_thresh_,
    float im_height,
    float im_width,
    float scale_xy,
    float scale_wh,
    float min_nms_score);

/*!
 *  @brief A function.
 *
 *  This function frees the cnmlPluginPostProcessRpnOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for cnmlPluginPostProcessRpnOp operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginPostProcessRpnOpParam(
    cnmlPluginPostProcessRpnOpParam_t param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginPostProcessRpnOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op_ptr
 *    Output. A pointer to the base operator address.
 *  @param[in]  rpn_box_encodings_batch
 *    Input. A tensor describes the bbox data.
 *  @param[in]  rpn_objectness_predictions_with_background_batch
 *    Input. A tensor describes the scores data.
 *  @param[in]  acnhors
 *    Input. A tensor describes the anchors.
 *  @param[in]  tmp_tensor
 *    Input. A tensor used to malloc temporary memory on mlu.
 *  @param[in]  proposal_box
 *    Input. The output tensor.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginPostProcessRpnOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginPostProcessRpnOpParam_t param,
    cnmlTensor_t rpn_box_encodings_batch,
    cnmlTensor_t rpn_objectness_predictions_with_background_batch,
    cnmlTensor_t anchors,
    cnmlTensor_t tmp_tensor,
    cnmlTensor_t proposal_box);


/*!
 *  @brief A function.
 *
 *  This function forwards PluginPostProcessRpnOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  rpn_box_encodings_batch_mlu_ptr
 *    Input. A pointer to the bbox data on mlu
 *  @param[in]  rpn_objectness_predictions_with_background_batch_mlu_ptr
 *    Input. A pointer to the scores data on mlu
 *  @param[in]  anchors_mlu_ptr
 *    Input. A pointer to the anchors on mlu
 *  @param[in]  tmp_tensor_mlu_ptr
 *    Input. A pointer to the temporary memory on mlu
 *  @param[out]  proposal_box_mlu_ptr
 *    Output. A pointer to the output on mlu
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginPostProcessRpnForward(
    cnmlBaseOp_t op,
    void *rpn_box_encodings_batch_mlu_ptr,
    void *rpn_objectness_predictions_with_background_batch_mlu_ptr,
    void *anchors_mlu_ptr,
    void *tmp_tensor_mlu_ptr,
    void *proposal_box_mlu_ptr,
    cnrtInvokeFuncParam_t forward_param,
    cnrtQueue_t queue);
/* -------------------------------------- */
/* cnmlPluginPostProcessRpn operation end */
/* -------------------------------------- */

/* ========================================= */
/* cnmlPluginResizeYuvToRgba operation start */
/* ========================================= */
/*!
 *  @brief A function.
 *
 *  This function create a PluginResizeYuvToRgbaOp param onject with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordiate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordiate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToRgbaOpParam(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  int roi_x,
  int roi_y,
  int roi_w,
  int roi_h,
  ioParams mode,
  int batchNum,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginResizeYuvToRgbaOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginResizeYuvToRgba operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginResizeYuvToRgbaOpParam(
    cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginResizeYuvToRgbaOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports Caffe/Pytorch on MLU100/MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors
 *    - tensor shapes does not meet reuqirements of YUV420SP
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToRgbaOp(
  cnmlBaseOp_t *op,
  cnmlPluginResizeAndColorCvtParam_t param,
  cnmlTensor_t *cnml_input_ptr,
  cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginResizeYuvToRgbaOpForward(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    void **input_addrs,
    void **output_addrs,
    cnrtInvokeFuncParam_t compute_forward_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image
 *  @param[in] src
 *    Input. The pointer of src image
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordiate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordiate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginResizeYuvToRgbaOpForward(
    unsigned char *dst,
    unsigned char *src,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    ioParams mode);
/* --------------------------------------- */
/* cnmlPluginResizeYuvToRgba operation end */
/* --------------------------------------- */

/* ========================================= */
/* cnmlPluginResizeConvert16B16C operation start */
/* ========================================= */
/*!
 *  @brief A function.
 *
 *  This function create a PluginResizeConvert16B16COp param onject with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The batch number of src image.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordiate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordiate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeConvert16B16COpParam(
    cnmlPluginResizeAndColorCvtParam_t *param,
    int batchNum,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    ioParams mode,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginResizeConvert16B16COpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginResizeConvert16B16C operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginResizeConvert16B16COpParam(
    cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginResizeConvert16B16COp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports Caffe on MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] cnml_rank_ptr
 *    Input. An array of 16 src image input address
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors
 *    - tensor shapes does not meet reuqirements of YUV420SP
 */
cnmlStatus_t cnmlCreatePluginResizeConvert16B16COp(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t *cnml_rank_ptr,
    cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeConvert16B16COp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  rank_input_addrs
 *    Input. An array stores the address of 16 input tensors
 *  @param[in]  output_addrs_addrs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginResizeConvert16B6COpForward(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    void **rank_input_addrs,
    void **output_addrs_cpu,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t stream);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeConvert16B16COp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image
 *  @param[in] src
 *    Input. The pointer of src image
 *  @param[in] batch_num
 *    Input. The batch number of src image.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordiate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordiate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginResizeConvert16B16COpForward(
    unsigned char* dst,
    unsigned char* src,
    int batch_num,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    ioParams mode);
/* --------------------------------------- */
/* cnmlPluginResizeConvert16B16C operation end */
/* --------------------------------------- */

/* ================================ */
/* cnmlPluginResize operation start */
/* ================================ */
/*!
 *  @brief A function.
 *
 *  This function create a PluginResizeOp param onject with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t *param,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    ioParams mode,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginResizeOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginResize operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginResizeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  traditional bi-linear interpolation method in OpenCV.
 *
 *  **Supports Caffe/Pytorch on MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in]  dst
 *    Input. A four-dimensional tensor for dst image
 *  @param[in]  src
 *    Input. A four-dimensional tensor for src image
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginResizeOp(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t dst,
    cnmlTensor_t src);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] src_tensor
 *    Input. A four-dimensional tensor for src image
 *  @param[in] dst_tensor
 *    Input. A four-dimensional tensor for dst image
 *  @param[in] src_mlu_ptr
 *    Input. Address of input tensor
 *  @param[in] dst_mlu_ptr
 *    Input. Address of output tensor
 *  @param[in] compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginResizeOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t src_tensor,
    cnmlTensor_t dst_tensor,
    void *src_mlu_ptr,
    void *dst_mlu_ptr,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] src_cpu_ptr
 *    Input. An array stores the address of all input tensors
 *  @param[in] dst_cpu_ptr
 *    Input. An array stores the address of all output tensors
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - addrs malloc failed
 */
cnmlStatus_t cnmlCpuComputePluginResizeOpForward(
    cnmlPluginResizeAndColorCvtParam_t param,
    uint8_t *src_cpu_ptr,
    uint8_t *dst_cpu_ptr);
/* ------------------------------ */
/* cnmlPluginResize operation end */
/* ------------------------------ */

/* =================================================== */
/* cnmlPluginFasterRcnnDetectionOutout operation start */
/* =================================================== */
/*!
 *  @struct cnmlPluginFasterRcnnDetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginFasterRcnnDetectionOutputOpParam is a structure describing the "param"
 *  parameter of FasterRcnnDetectionOutput operation, used to create conv operation.
 *  cnmlCreateConvOpParam() is used to create an instance of cnmlConvOpParam_t.
 *  cnmlDestroyConvOpParam() is used to destroy an instance of cnmlConvOpParam_t. */

struct cnmlPluginFasterRcnnDetectionOutputOpParam
{
    int batch_num;
    int box_num;
    int num_class;
    int im_h;
    int im_w;
    float scale;
    float nms_thresh;
    float score_thresh;
    bool fix8;
    cnmlCoreVersion_t core_version;
};
/*! ``cnmlPluginFasterRcnnDetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginFasterRcnnDetectionOutputOpParam) holding the description of a FasterRcnnDetectionOutput operation param.
*/
typedef cnmlPluginFasterRcnnDetectionOutputOpParam
*cnmlPluginFasterRcnnDetectionOutputOpParam_t;

/*i
 *  @brief A function.
 *
 *  This function creates a PluginFasterRcnnDetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batch_num
 *    Input. The number of batches.
 *  @param[in] box_num
 *    Input. The number of input box.
 *  @param[in] num_class
 *    Input. The number of classes.
 *  @param[in] im_h
 *    Input. Height of input image of backbone network.
 *  @param[in] im_w
 *    Input. Width of input image of backbone network.
 *  @param[in] score_thresh
 *    Input. Score threshold.
 *  @param[in] nms_thresh
 *    Input. Enumerant IOU threshold used in NMS function.
 *  @param[in] fix8
 *    Input. Precision(fix8=1->INT8; fix8=1->FLOAT/HALF).
 *  @param[in] core_version
 *    Input. Supported core version, including MLU100/220/270.
 *  @param[in] scale
 *    Input. The scaling of images.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */

cnmlStatus_t cnmlCreatePluginFasterRcnnDetectionOutputOpParam(
    cnmlPluginFasterRcnnDetectionOutputOpParam_t *param,
    int batch_num,
    int box_num,
    int num_class,
    int im_h,
    int im_w,
    bool fix8,
    cnmlCoreVersion_t core_version,
    float scale,
    float nms_thresh,
    float score_thresh);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginFasterRcnnDetectionOutputParam struct,
 *  pointed by the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *           for PluginFasterRcnnDetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginFasterRcnnDetectionOutputOpParam(
    cnmlPluginFasterRcnnDetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginFasterRcnnDetectionOutputOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginFasterRcnnDetectionOutput parameter struct pointer.
 *  @param[in]  bbox_pred
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [num_class * 4, box_num, 1, 1](NCHW).
 *           Support FLOAT16/FLOAT32 dataType currently.
 *  @param[in]  scores_pred
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [num_class, box_num, 1, 1](NCHW).
 *           Support FLOAT16/FLOAT32 dataType currently.
 *  @param[in]  rois_pred
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [1, box_num, 1, 5](NCHW).
 *           Support FLOAT16/FLOAT32 dataType currently.
 *  @param[out]  new_box
 *    Output. An array of four-demensional cnmlTensors with a shape of
 *           [1, box_num * num_class, 1, 6](NCHW).
 *           Support FLOAT16/FLOAT32 dataType currently.
 *  @param[out]  tmp
 *    Output. An array of four-demensional cnmlTensors with a shape of
 *           [1, 64, 1, 1](NCHW).
 *           Support FLOAT16/FLOAT32 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 */

cnmlStatus_t cnmlCreatePluginFasterRcnnDetectionOutputOp(
    cnmlBaseOp_t *op_ptr,
    cnmlTensor_t bbox_pred,
    cnmlTensor_t scores_pred,
    cnmlTensor_t rois_pred,
    cnmlTensor_t new_box,
    cnmlTensor_t tmp,
    cnmlPluginFasterRcnnDetectionOutputOpParam_t param);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginFasterRcnnDetectionOutputOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  num_outputs
 *    Input. Number of output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 */

cnmlStatus_t cnmlComputePluginFasterRcnnDetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *inputs[],
    int num_inputs,
    void *outputs[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginFasterRcnnDetectionOutputOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A PluginFasterRcnnDetectionOutput parameter struct pointer.
 *  @param[in]  cls_boxes
 *    Input. An array stores the address of all cpu input box data
 *  @param[in]  cls_scores
 *    Input. An array stores the address of all cpu input score data
 *  @param[in]  ori_rois
 *    Input. An array stores the address of all cpu input rois data
 *  @param[out]  all_decoded_boxes_cpu
 *    Output. An array stores the address of all cpu output data
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 */
cnmlStatus_t cnmlCpuComputePluginFasterRcnnDetectionOutputOpForward(
    cnmlPluginFasterRcnnDetectionOutputOpParam_t param,
    float *all_decoded_boxes_cpu,
    float *cls_boxes,
    float *cls_scores,
    float *ori_rois);
/* ------------------------------------------------- */
/* cnmlPluginFasterRcnnDetectionOutout operation end */
/* ------------------------------------------------- */

/* ========================================= */
/* cnmlPluginPsRoipoolOp operation start */
/* ========================================= */
/*!
 *  @struct cnmlPluginPsRoiPoolOpParam
 *  @brief A struct.
 *
 *  cnmlPluginPsRoiPoolOpParam is a structure describing the "param"
 *  parameter of PsRoiPool operation, used to create conv operation.
 *  cnmlCreateConvOpParam() is used to create an instance of cnmlConvOpParam_t.
 *  cnmlDestroyConvOpParam() is used to destroy an instance of cnmlConvOpParam_t.
 */
struct cnmlPluginPsRoiPoolOpParam {
  int batchNum;
  int int8;
  int outputdim;
  int group_size;
  int height;
  int width;
  int pooled_height;
  int pooled_width;
  int num_rois;
  int rois_offset;
  float spatial_scale;
  cnmlCoreVersion_t core_version;
};
/*! ``cnmlPluginPsRoiPoolOpParam_t`` is a pointer to a
    structure (cnmlPluginPsRoiPoolOpParam) holding the description of a PsRoiPool operation param.
*/
typedef cnmlPluginPsRoiPoolOpParam *cnmlPluginPsRoiPoolOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginPsroiPoolOpParam param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of batches.
 *  @param[in] input8
 *    Input. Int input tensor is or is not int8.
 *  @param[in] outputdim
 *    Input. The size of outputdim.
 *  @param[in] group_size
 *    Input. The size of group.
 *  @param[in] height
 *    Input. The height of feature map.
 *  @param[in] width
 *    Input. The width of feature map.
 *  @param[in] pooled_height
 *    Input. The height of output feature map .
 *  @param[in] pooled_width
 *    Input. The width of output feature map.
 *  @param[in] nums_rois.
 *    Input  The num of rois.
 *  @param[in] rois_offset.
 *    Input  The len of per roi.
 *  @param[in] spatial_scale.
 *    Input  Spatial scale.
 *  @param[in] core_version
 *    Input. Supported core version, including 220/270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginPsRoiPoolOpParam(
    cnmlPluginPsRoiPoolOpParam_t *param,
  int batchNum,
  int int8,
  int outputdim,
  int group_size,
  int height,
  int width,
  int pooled_height,
  int pooled_width,
  int num_rois,
  int rois_offset,
  float spatial_scale,
  cnmlCoreVersion_t core_version);


/*!
 *  @brief A function.
 *
 *  This function frees the PluginPsRoiPoolParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginPsRoiPool operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginPsRoiPoolOpParam(
    cnmlPluginPsRoiPoolOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginPsRoiPoolOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginPsRoiPool parameter struct pointer.
 *  @param[in]  psroipool_input_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, outputdim * group_size * group_size, height, width](NCHW).The other
 *           four-demensional cnmlTensors width a shape of [batch_num,num_rois,rois_offset,1]
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum * num_rois, outputdim, pooled_height, pooled_width](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - base op pointer is nullptr
 *    - param is nullptr or not initialized
 *    - input/output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginPsRoiPoolOp(
    cnmlBaseOp_t *op,
    cnmlPluginPsRoiPoolOpParam_t param,
    cnmlTensor_t *psroipool_input_tensors,
    cnmlTensor_t *psroipool_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginPsRoiPoolOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  num_outputs
 *    Input. Number of output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - op is nullptr or not initialized
 *    - input/output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlComputePluginPsroipoolOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/* --------------------------------- */
/* cnmlPluginPsRoiPool operation end */
/* --------------------------------- */

/* =================================== */
/* cnmlPluginYuv2RgbOp operation start */
/* =================================== */
/*!
 *  @brief A function.
 *
 *  This function creates a PluginYuvToRgbOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. Rows of input image of backbone network.
 *  @param[in] s_col
 *    Input. Cols of input image of backbone network.
 *  @param[in] mode
 *    Input. The model for convert.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU100/220/270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginYuvToRgbOpParam(
    cnmlPluginResizeAndColorCvtParam_t *param,
    int s_row,
    int s_col,
    ioParams mode,
    cnmlCoreVersion_t core_version,
    bool is_variable = false);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginYuvToRgbOp struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginYuvToRgbOp  operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginYuvToRgbOpParam(cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginYuvToRgbOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYuvToRgbOp parameter struct pointer.
 *  @param[in]  yuv2rgb_input_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, channelIn, rows, cols](NCHW).
 *           Support FLOAT16 or UINT8 dataType currently.
 *  @param[in]  outputs
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, channel, rows, cols](NCHW).
 *           Support FLOAT16 or UINT8 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 */
cnmlStatus_t cnmlCreatePluginYuvToRgbOp(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYuvToRgbOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYuvToRgbOp parameter struct pointer.
 *  @param[in]  inputs_addrs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  outputs_addrs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 */
cnmlStatus_t cnmlComputePluginYuvToRgbOpForward(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    void **input_addrs,
    void **output_addrs,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYuvToRgbOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] dst
 *    Output. The result of yuv2rgb on cpu.
 *  @param[in] src
 *    Input. The YUV image data.
 *  @param[in] s_row
 *    Input. Rows of input image of backbone network.
 *  @param[in] s_col
 *    Input. Cols of input image of backbone network.
 *  @param[in] mode
 *    Input. The model for convert.
 */
void cnmlCpuComputePluginYuvToRgbOpForward(
    unsigned char* dst,
    unsigned char* src,
    int s_row,
    int s_col,
    ioParams mode);
/* --------------------------------- */
/* cnmlPluginYuv2RgbOp operation end */
/* --------------------------------- */

/* ======================================= */
/* cnmlPluginCropAndResize operation start */
/* ======================================= */
/*!
 *  @brief A function.
 *
 *  This function create a PluginCropAndResizeOp param onject with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginCropAndResizeOpParam(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  ioParams mode,
  int batchNum,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginCropAndResizeOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginCropAndResize operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginCropAndResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginCropAndResizeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  traditional bi-linear interpolation method on OpenCV.
 *
 *  **Supports Caffe/Pytorch on MLU100/MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] dst
 *    Input. A four-dimensional tensor for dst image
 *  @param[in] src
 *    Input. A four-dimensional tensor for src image
 *  @param[in] cropParams
 *    Input. A four-dimensional tensor for all cropParams, i.e. roiParams
 *  @param[in] roiNums
 *    Input. A four-dimensional tensor for the number of rois of each images
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors
 *    - shapes of cropParams and roiNums are not consistent
 */
cnmlStatus_t cnmlCreatePluginCropAndResizeOp(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t dst,
    cnmlTensor_t src,
    cnmlTensor_t cropParams,
    cnmlTensor_t roiNums);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] src_mlu_tensor
 *    Input. A four-dimensional tensor for src image
 *  @param[in] src_addr
 *    Input. Address of input tensor
 *  @param[in] cropParams_mlu_tensor
 *    Input. A four-dimensional tensor for cropParams
 *  @param[in] cropParams_addr
 *    Input. Address of cropParams tensor
 *  @param[in] roiNums_mlu_tensor
 *    Input. A four-dimensional tensor for roiNums
 *  @param[in] roiNums_addr
 *    Input. Address of roiNums tensor
 *  @param[in] dst_tensor
 *    Input. A four-dimensional tensor for dst image
 *  @param[Out] dst_addr
 *    Output. Address of output tensor
 *  @param[in] compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginCropAndResizeOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t src_mlu_tensor,
    void *src_addr,
    cnmlTensor_t cropParams_mlu_tensor,
    void *cropParams_addr,
    cnmlTensor_t roiNums_mlu_tensor,
    void *roiNums_addr,
    cnmlTensor_t dst_mlu_tensor,
    void *dst_addr,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginCropAndResizeOp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image
 *  @param[in] src
 *    Input. The pointer of src image
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordiate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordiate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Input/output pointer is nullptr.
 *    - Param is not consistent with input and output.
 */
cnmlStatus_t cnmlCpuComputePluginCropAndResizeOpForward(
    unsigned char* dst,
    unsigned char* src,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h);
/* ------------------------------------- */
/* cnmlPluginCropAndResize operation end */
/* ------------------------------------- */

/* ======================================= */
/* cnmlPluginCropFeatureAndResize operation start */
/* ======================================= */
/*!
 *  @brief A function.
 *
 *  This function create a PluginCropFeatureAndResizeOp param onject with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] depth
 *    Input. The depth/channel of src image.
 *  @param[in] box_number
 *    Input. detect number of bbox.
 *  @param[in] pad_size
 *    Input. pad_size.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginCropFeatureAndResizeOpParam(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  int batchNum,
  int depth,
  int box_number,
  int pad_size,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginCropFeatureAndResizeOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginCropFeatureAndResize operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginCropFeatureAndResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginCropFeatureAndResizeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  traditional bi-linear interpolation method on OpenCV.
 *
 *  **Supports Caffe/Pytorch on MLU100/MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] input_cnml_tensors
 *    Input. A four-dimensional tensor for dst image
 *  @param[in] output_cnml_tensors
 *    Input. A four-dimensional tensor for src image
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors
 *    - shapes of cropParams and roiNums are not consistent
 */
cnmlStatus_t cnmlCreatePluginCropFeatureAndResizeOp(
    cnmlBaseOp_t* op,
    cnmlPluginResizeAndColorCvtParam_t* param,
    cnmlTensor_t* input_cnml_tensors, // src
    cnmlTensor_t* output_cnml_tensors);  // dst

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_addr
 *    Input. Address of input tensor
 *  @param[Out] output_addr
 *    Output. Address of output tensor
 *  @param[in] compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginCropFeatureAndResizeOpForward(
    cnmlBaseOp_t op,
    void* input_addr[],
    void* output_addr[],
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginCropFeatureAndResizeOp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image
 *  @param[in] src
 *    Input. The pointer of src image
 *  @param[in] boxes
 *    Input. The pointer to detect bbox.
 *  @param[in] box_index
 *    Input. The pointer to index of bbox.
 *  @param[in] new_box
 *    Input. The pointer to output.
 *  @param[in] batchNum
 *    Input. batch size.
 *  @param[in] depth
 *    Input. The channel of input feature.
 *  @param[in] image_height
 *    Input. The height of input feature.
 *  @param[in] image_width
 *    Input. The width of input feature.
 *  @param[in] crop_height
 *    Input. The height of resize output.
 *  @param[in] crop_width
 *    Input. The width of resize output.
 *  @param[in] box_number
 *    Input. The number of detect bbox.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Input/output pointer is nullptr.
 *    - Param is not consistent with input and output.
 */
cnmlStatus_t cnmlCpuComputePluginCropFeatureAndResizeOpForward(
    float* src,
    float* boxes,
    float* box_index,
    float* new_box,
    int batchNum,
    int depth,
    int image_height,
    int image_width,
    int crop_height,
    int crop_width,
    int box_number);
/* ------------------------------------- */
/* cnmlPluginCropFeatureAndResize operation end */
/* ------------------------------------- */


/* ======================================= */
/* cnmlPluginNonMaxSuppression operation start */
/* ======================================= */
/*!
 * @struct cnmlPluginNonMaxSuppressionOpParam
 * @brief A struct.
 *
 * cnmlPluginNonMaxSuppressionOpParam is a structure describing thr "param"
 * parameter of NonMaxSuppression operation.
 * cnmlCreatePluginNonMaxSuppressionOpParam() is used to create
 * an instance of cnmlPluginNonMaxSuppressionOpParam_t.
 * cnmlDestoryPluginNonMaxSuppressionOpParam() is used to destory
 * an instance of cnmlPluginNonMaxSuppressionOpParam_t.
 */
struct cnmlPluginNonMaxSuppressionOpParam
{
  cnmlTensor_t *cnml_static_tensors;
  void* *cpu_static_init;
  int len;
  int max_num;
  float iou_threshold;
  float score_threshold;
  cnmlCoreVersion_t core_version;
  float *input;
};
/*! ``cnmlPluginNonMaxSuppressionOpParam_t`` is a pointer to a
    structure (cnmlPluginNonMaxSuppressionOpParam) holding the description of a NonMaxSuppression operation param.
*/
typedef cnmlPluginNonMaxSuppressionOpParam
*cnmlPluginNonMaxSuppressionOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function create a PluginNonMaxSuppressionOp param onject with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] len
 *    Input. The number of input boxes.
 *  @param[in] max_num
 *    Input. The max number of output boxes.
 *  @param[in] iou_threshold
 *    Input. The threshold of iou to do nms.
 *  @param[in] score_threshold
 *    Input. The threshold of score to do nms.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginNonMaxSuppressionOpParam(
  cnmlPluginNonMaxSuppressionOpParam_t *param,
  int len,
  int max_num,
  float iou_threshold,
  float score_threshold,
  cnmlCoreVersion_t core_version=CNML_MLU270);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginNonMaxSuppressionOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginNonMaxSuppression operator.
 *  @param[in]  static_num
 *    Input. Number of static tensors
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginNonMaxSuppressionOpParam(
    cnmlPluginNonMaxSuppressionOpParam_t *param,
    int static_num);

/*!
 *  @brief A function.
 *
 *  This function creates PluginNonMaxSuppressionOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports Tensorflow on MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginNonMaxSuppression parameter struct pointer.
 *  @param[in] nms_input_tensors
 *    Input. This pointer contains two array of four-demensional cnmlTensors,
 *           first tensor's shape is [4, len, 1, 1], second tensor's shape is [1, len, 1, 1].
 *  @param[in] input_num
 *    Input. Number of input tensors
 *  @param[out] nms_output_tensors
 *    Output. This pointer contains an array of four-demensional cnmlTensor,
 *           the tensor's shape is [1, max_num, 1, 1].
 *  @param[in] output_num
 *    Input. Number of output tensors
 *  @param[in] static_num
 *    Input. Number of static tensors
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors
 *    - shapes of cropParams and roiNums are not consistent
 */
cnmlStatus_t cnmlCreatePluginNonMaxSuppressionOp(
    cnmlBaseOp_t *op,
    cnmlPluginNonMaxSuppressionOpParam_t param,
    cnmlTensor_t *nms_input_tensors,
    int input_num,
    cnmlTensor_t *nms_output_tensors,
    int output_num,
    int static_num);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginNonMaxSuppressionOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in] num_inputs
 *    Input. Number of input tensors
 *  @param[out] output_tensors
 *    Output. Void
 *  @param[in] outputs
 *    Input. A array stores the address of all output tensors
 *  @param[in] num_outputs
 *    Input. Number of output tensors
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @param[in] extra
 *    Input. A pointer contains other input params
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginNonMaxSuppressionOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t input_tensors[],
    void *inputs[],
    int num_inputs,
    cnmlTensor_t output_tensors[],
    void *outputs[],
    int num_outputs,
    cnrtQueue_t queue,
    void *extra);

/* ------------------------------------- */
/* cnmlPluginNonMaxSuppression operation end */
/* ------------------------------------- */

/* ================================= */
/*  cnmlPluginNms operation start    */
/* ================================= */
/*!
 *  @struct cnmlPluginNmsOpParam
 *  @brief A struct.
 *
 *  cnmlPluginNmsOpParam is a structure describing the "param"
 *  parameter of cnmlPluginNmsOpParam operation.
 *  cnmlCreatePlugincnmlPluginNmsOpParam() is used to create an instance of
 *  cnmlPluginNmsParam_t.
 *  cnmlDestroyPlugincnmlPluginNmsOpParam() is used to destroy an instance
 *  of cnmlPluginNmsParam_t.
 */
struct cnmlPluginNmsOpParam
{
  int n;
  int channels;
  int height;
  float overlap_Thresh;
  float valid_Thresh;
  int topk;
  int coord_start;
  int score_index;
  int id_index;
  int background_id;
  bool force_suppress;
  int in_format;
  int out_format;
  int dtype_flag;
  cnmlCoreVersion_t coreVersion;
};
/*! ``cnmlPluginNmsOpParam_t`` is a pointer to a
    structure (cnmlPluginNmsOpParam) holding the description of a NmsOp operation param.
*/
typedef cnmlPluginNmsOpParam *cnmlPluginNmsOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginNmsOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official MXNet website.
 *
 *  **Supports MXNet on MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] n
 *    Input. The number of batches.
 *  @param[in] channels
 *    Input. The number of boxes.
 *  @param[in] h
 *    Input. The number of items.
 *  @param[float] overlap_Thresh
 *    Input. Overlapping(IoU) threshold to suppress object with smaller score.
 *  @param[float] valid_Thresh
 *    Input. Filter input boxes to those whose scores greater than valid_thresh.
 *  @param[in] topk
 *    Input. Apply nms to topk boxes with descending scores.
 *  @param[in] coord_start
 *    Input. Start index of the consecutive 4 coordinates.
 *  @param[in] score_index
 *    Input. Index of the scores/confidence of boxes.
 *  @param[in] id_index.
 *    Input. Index of the class categories.
 *  @param[in] background_id
 *    Input. The id of background.
 *  @param[bool] force_suppress
 *    Input. if set 0 and id_index is provided, nms will only apply to boxes belongs to the same category
 *  @param[in] in_format
 *    Input. The input box encoding type.1 indicate "center" and 0 indicate "corner".
 *  @param[in] out_format
 *    Input. The output box encoding type.1 indicate "center" and 0 indicate "corner".
 *  @param[in] dtype_flag
 *    Input. The data type of input. 0:float32, 1:float64, 2:float16
 *  @param[cnmlCoreVersion_t] coreVersion
 *    Input. The core version of MLU.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginNmsOpParam(
  cnmlPluginNmsOpParam_t *param,
  int n,
  int channels,
  int height,
  float overlap_Thresh,
  float valid_Thresh,
  int topk,
  int coord_start,
  int score_index,
  int id_index,
  int background_id,
  bool force_suppress,
  int in_format,
  int out_format,
  int dtype_flag=2,
  cnmlCoreVersion_t coreVersion=CNML_MLU270);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginNmsOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginNms operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginNmsOpParam(
    cnmlPluginNmsOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginNmsOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official MXNet NMS op.
 *
 *  **Supports MXNet on MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginNmsOp parameter struct pointer.
 *  @param[in]  nms_input_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, inputC, inputH, inputW](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Output. An array of four-demensional cnmlTensors with a shape of
 *           [batchsize, anchor_num, 4, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The size is the length of result.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginNmsOp(
  cnmlBaseOp_t *op,
  cnmlPluginNmsOpParam_t param,
  cnmlTensor_t *nms_input_tensors,
  cnmlTensor_t *nms_output_tensors
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginNmsOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in] num_inputs
 *    Input. Number of input tensors
 *  @param[out] output_tensors
 *    Output. Void
 *  @param[in] outputs
 *    Input. A array stores the address of all output tensors
 *  @param[in] num_outputs
 *    Input. Number of output tensors
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginNmsOpForward(
  cnmlBaseOp_t op,
  void **inputs,
  int input_num,
  void **outputs,
  int output_num,
  cnrtQueue_t queue
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginNmsOp on CPU.
 *
 *  **Supports MLU270**
 *
 *  @param[in]  param
 *    Input. A PluginProposal parameter struct pointer.
 *  @param[in]  inputs
 *    Input. Adress of cpu input data
 *  @param[out]  outputs
 *    Output. Adress of cpu output data
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginNmsOpForward(
    cnmlPluginNmsOpParam_t param,
    float *input,
    float *output);

/* ================================= */
/*  cnmlPluginNms operation end      */
/* ================================= */


/* ================================= */
/*  cnmlPluginInitOp operation start    */
/* ================================= */
/*!
 *  @struct cnmlPluginInitOpParam
 *  @brief A struct.
 *
 *  cnmlPluginInitOpParam is a structure describing the "param"
 *  parameter of cnmlPluginInitOpParam operation.
 *  cnmlCreatePlugincnmlPluginInitOpParam() is used to create an instance of
 *  cnmlPluginInitParam_t.
 *  cnmlDestroyPlugincnmlPluginInitOpParam() is used to destroy an instance
 *  of cnmlPluginInitParam_t.
 */
struct cnmlPluginInitOpParam
{
  int size;
  float value;
  int dtype_flag;
  cnmlCoreVersion_t coreVersion;
};
/*! ``cnmlPluginInitOpParam_t`` is a pointer to a
    structure (cnmlPluginInitOpParam) holding the description of a InitOp operation param.
*/
typedef cnmlPluginInitOpParam *cnmlPluginInitOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginInitOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official MXNet website.
 *
 *  **Supports MXNet on MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] size
 *    Input. The size of need initialized.
 *  @param[float] value
 *    Input. The value of should be initialized.
 *  @param[in] dtype_flag
 *    Input. The data type of input. 0:float32, 1:float64, 2:float16
 *  @param[cnmlCoreVersion_t] coreVersion
 *    Input. The core version of MLU.
 */
cnmlStatus_t cnmlCreatePluginInitOpParam(
  cnmlPluginInitOpParam_t *param,
  int size,
  float value,
  int dtype_flag=2,
  cnmlCoreVersion_t coreVersion=CNML_MLU270);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginInitOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginInit operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginInitOpParam(
    cnmlPluginInitOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginInitOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official MXNet Init op.
 *
 *  **Supports MXNet on MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginInitOp parameter struct pointer.
 *  @param[in]  Init_input_tensors
 *    Input. An array of multi-demensional cnmlTensors .
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Output. An array of multi-demensional cnmlTensors.
 *           Support only FLOAT16 dataType currently.
 *           The size is the length of result.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginInitOp(
  cnmlBaseOp_t *op,
  cnmlPluginInitOpParam_t param,
  cnmlTensor_t *Init_input_tensors,
  cnmlTensor_t *Init_output_tensors
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginInitOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in] num_inputs
 *    Input. Number of input tensors
 *  @param[out] output_tensors
 *    Output. Void
 *  @param[in] outputs
 *    Input. A array stores the address of all output tensors
 *  @param[in] num_outputs
 *    Input. Number of output tensors
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginInitOpForward(
  cnmlBaseOp_t op,
  void **inputs,
  int input_num,
  void **outputs,
  int output_num,
  cnrtQueue_t queue
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginInitOp on CPU.
 *
 *  @param[in] param
 *    Input. Param of Init operator, cnmlPluginInitOpParam
 *  @param[in] output
 *    Input. An address of output tensor
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
cnmlStatus_t cnmlCpuComputePluginInitOpForward(
  cnmlPluginInitOpParam_t param,
  float *output
);

/* ================================= */
/*  cnmlPluginInit operation end      */
/* ================================= */

/* ================================= */
/*  cnmlPluginArange operation start    */
/* ================================= */
/* cnmlPluginArangeOpParam
 *  @brief A struct.
 *
 *  cnmlPluginArangeOpParam is a structure describing the "param"
 *  parameter of cnmlPluginArangeOpParam operation.
 *  cnmlCreatePlugincnmlPluginArangeOpParam() is used to create an instance of
 *  cnmlPluginArangeParam_t.
 *  cnmlDestroyPlugincnmlPluginArangeOpParam() is used to destroy an instance
 *  of cnmlPluginArangeParam_t.
 */
struct cnmlPluginArangeOpParam
{
  float start;
  float stop;
  float step;
  int repeat;
  int size;
  int dtype_flag;
  cnmlCoreVersion_t coreVersion;
};
/*! ``cnmlPluginArangeParam_t`` is a pointer to a
    structure (cnmlPluginArangeParam) holding the description of a ArangeOp operation param.
*/
typedef cnmlPluginArangeOpParam *cnmlPluginArangeParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginArangeOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official MXNet website.
 *
 *  **Supports MXNet on MLU270**
 * cnmlPluginArangeParam_t *param,
  float start,
  float stop,
  float step,
  int repeat,
  int size,
  int dtype_flag,
  cnmlCoreVersion_t coreVersion
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[float] start
 *    Input. Start of interval.
 *  @param[float] stop
 *    Input. End of interval.
 *  @param[float] step
 *    Input. Spacing between values.
 *  @param[int] repeat
 *    Input. The repeating time of all elements.
 *  @param[int] size
 *    Input. intput shape size .
 *  @param[int] dtype_flag
 *    Input. The data type of input. only support float16 so far
 *  @param[cnmlCoreVersion_t] coreVersion
 *    Input. The core version of MLU.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginArangeOpParam(
  cnmlPluginArangeParam_t *param,
  float start,
  float stop,
  float step,
  int repeat,
  int size,
  int dtype_flag,
  cnmlCoreVersion_t coreVersion);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginArangeOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginNms operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginArangeOpParam(
    cnmlPluginArangeParam_t *param);
/*!
 *  @brief A function.
 *
 *  This function creates PluginArangeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official MXNet NMS op.
 *
 *  **Supports MXNet on MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginArangeOp parameter struct pointer.
 *  @param[in]  arange_input_tensors
 *    Input. Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Output.Support only FLOAT16 dataType currently.
 *           The size is the length of result.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginArangeOp(
  cnmlBaseOp_t *op,
  cnmlPluginArangeParam_t param,
  cnmlTensor_t *arange_input_tensors,
  cnmlTensor_t *arange_output_tensors
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginArangeOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in] num_inputs
 *    Input. Number of input tensors
 *  @param[out] output_tensors
 *    Output. Void
 *  @param[in] outputs
 *    Input. A array stores the address of all output tensors
 *  @param[in] num_outputs
 *    Input. Number of output tensors
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginArangeOpForward(
  cnmlBaseOp_t op,
  void **inputs,
  int input_num,
  void **outputs,
  int output_num,
  cnrtQueue_t queue
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginArangeOp on CPU.
 *
 *  @param[in] param
 *    Input. A pointer of cnmlPluginArangeParam, which
 *    support params needed by this operator.
 *  @param[in] output
 *    Input. An address of output tensor
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 */
cnmlStatus_t cnmlCpuComputePluginArangeOpForward(
  cnmlPluginArangeParam_t param,
  float* output
);

/* ======================================= */
/* cnmlPluginArange operation end */
/* ======================================= */

/* ======================================= */
/* cnmlPluginYolov2DetectionOutput operation start */
/* ======================================= */
/*!
 *  @struct cnmlPluginYolov2DetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginYolov2DetectionOutputOpParam is a structure describing the "param"
 *  parameter of Yolov2DetectionOutput operation.
 *  cnmlCreatePluginYolov2DetectionOutputOpParam() is used to create
 *  an instance of cnmlPluginYolov2DetectionOutputOpParam_t.
 *  cnmlDestroyPluginYolov2DetectionOutputOpParam() is used to destroy
 *  an instance of cnmlPluginYolov2DetectionOutputOpParam_t.
 */
struct cnmlPluginYolov2DetectionOutputOpParam {
  cnmlTensor_t *cnml_static_tensors;
  cnmlCpuTensor_t *cpu_static_tensors;
  int width;
  int height;
  int classNum;
  int anchorNum;
  int coords;
  int paramNum;
  int batchNum;
  int int8_mode;
  float confidence_thresh;
  float nms_thresh;
  cnmlCoreVersion_t core_version;
  float* biases;
};
/*! ``cnmlPluginYolov2DetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginYolov2DetectionOutputOpParam) holding the description of a Yolov2DetectionOutput operation param.
*/
typedef cnmlPluginYolov2DetectionOutputOpParam *cnmlPluginYolov2DetectionOutputOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginYolov2DetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] width
 *    Input. width.
 *  @param[in] height
 *    Input. height.
 *  @param[in] classNum
 *    Input. The number of classes.
 *  @param[in] anchorNum
 *    Input. The number of anchors.
 *  @param[in] coords
 *    Input. The number of anchor coordinates.
 *  @param[in] batchNum
 *    Input. The number of batch.
 *  @param[in] int8_mode
 *    Input. If the net run in int8 mode.
 *  @param[in] confidence_thresh
 *    Input. Confidence threshold.
 *  @param[in] nms_thresh.
 *    Enumerant IOU threshold used in NMS function.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU100/220/270.
 *  @param[in] biases
 *    Input. The biase data.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @warning
 *    The sum of input tensor HW values should be less than 32768.
 */
cnmlStatus_t cnmlCreatePluginYolov2DetectionOutputOpParam(
    cnmlPluginYolov2DetectionOutputOpParam_t *param,
    int width,
    int height,
    int classNum,
    int anchorNum,
    int coords,
    int paramNum,
    int batchNum,
    int int8_mode,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version,
    float* biases
    );

/*!
 *  @brief A function.
 *
 *  This function frees the PluginYolov2DetectionOutputOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginYolov2DetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginYolov2DetectionOutputOpParam(
    cnmlPluginYolov2DetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginYolov2DetectionOutputOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports Caffe/Pytorch on MLU100/MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYolov2DetectionOutput parameter struct pointer.
 *  @param[in]  yolov2_input_tensors
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, anchornum * width * height, 1, (paramnum + 5)](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Input. An array of four-demensional cnmlTensors with a shape of
 *           [batchNum, 1, 7, 256](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The first two numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, xc, yc, w, h], where
 *           (xc, yc) is the coordinates of center of the box, w is the width of
 *           the bos, h is the height of the box.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input / output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginYolov2DetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginYolov2DetectionOutputOpParam_t param,
    cnmlTensor_t *yolov2_input_tensors,
    cnmlTensor_t *yolov2_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov2DetectionOutputOp on MLU.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[out]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[out]  outputs
 *    Output. An array stores the address of all output tensors
 *  @param[in]  num_outputs
 *    Input. Number of output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginYolov2DetectionOutputOpForward_V2(
    cnmlBaseOp_t op,
    void *inputs[],
    int num_inputs,
    void *outputs[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t stream);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov2DetectionOutputOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginYolov2DetectionOutput parameter struct pointer.
 *  @param[in]  inputs
 *    Input. An array stores the address of all cpu input data
 *  @param[in]  biases_ori
 *    Input. An array stores the address of bias input data
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input / output addrs is nullptr ot malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginYolov2DetectionOutputOpForward(
    cnmlPluginYolov2DetectionOutputOpParam_t param,
    void *inputs,
    void *biases_ori,
    void *outputs);

/* ------------------------------------- */
/* cnmlPluginYolov2DetectionOutput operation end */
/* ------------------------------------- */

/* ======================================= */
/* cnmlPluginBertPre operation start */
/* ======================================= */
/*!
 *  @struct cnmlPluginBertPreParam
 *  @brief A struct.
 *
 *  cnmlPluginBertPreParam is a structure describing the "param"
 *  parameter of BertPre operation.
 *  cnmlCreatePluginBertPreOpParam() is used to create
 *  an instance of cnmlPluginBertPreOpParam_t.
 *  cnmlDestroyPluginBertPreOpParam() is used to destroy
 *  an instance of cnmlPluginBertPreOpParam_t.
 */
struct cnmlPluginBertPreParam {
  int vocab_size;
  int segment_size;
  int position_size;
  int batch_num;
  int seq_len;
  int hidden_size;

  cnmlCoreVersion_t core_version;

  cnmlTensor_t *cnml_static_ptr;
  void **static_data_ptr;
};

/*! ``cnmlPluginBertPreParam_t`` is a pointer to a
    structure (cnmlPluginBertPreParam) holding the description of a BertPre operation param.
*/
typedef cnmlPluginBertPreParam *cnmlPluginBertPreParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginBertPreOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  word_table_ptr
 *    Input. An array stores the word table
 *  @param[in]  segment_table_ptr
 *    Input. An array stores the segment table
 *  @param[in]  position_table_ptr
 *    Input. An array stores the position table
 *  @param[in]  layernorm_gamma_ptr
 *    Input. An array stores the layernorm gamma
 *  @param[in]  layernorm_bata_ptr
 *    Input. An array stores the layernorm bata
 *  @param[in]  vocab_size
 *    Input. The size of vacab embedding table
 *  @param[in]  segment_size
 *    Input. The size of segment embedding table
 *  @param[in]  position_size
 *    Input. The size of positin embedding table
 *  @param[in]  batch_num
 *    Input. The number of batch
 *  @param[in]  seq_len
 *    Input. The length of sequnce
 *  @param[in]  hidden_size
 *    Input. The size of embedding vector
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginBertPreOpParam(
    cnmlPluginBertPreParam_t *param,
    cnmlCoreVersion_t core_version,
    float* word_table_ptr,
    float* segment_table_ptr,
    float* position_table_ptr,
    float* layernorm_gamma_ptr,
    float* layernorm_beta_ptr,
    int vocab_size,
    int segment_size,
    int position_size,
    int batch_num,
    int seq_len,
    int hidden_size);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginBertPreOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  vocab_size
 *    Input. The size of vacab embedding table
 *  @param[in]  segment_size
 *    Input. The size of segment embedding table
 *  @param[in]  position_size
 *    Input. The size of positin embedding table
 *  @param[in]  batch_num
 *    Input. The number of batch
 *  @param[in]  seq_len
 *    Input. The length of sequnce
 *  @param[in]  hidden_size
 *    Input. The size of embedding vector
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginBertPreOpParam_V2(
    cnmlPluginBertPreParam_t *param,
    cnmlCoreVersion_t core_version,
    int vocab_size,
    int segment_size,
    int position_size,
    int batch_num,
    int seq_len,
    int hidden_size);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginBertPreParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBertPre operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBertPreOpParam(
    cnmlPluginBertPreParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginBertPreParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU100/MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBertPre operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBertPreOpParam_V2(
    cnmlPluginBertPreParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginBertPreOp with proper param,
 *
 *  **Supports MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertPre parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for input
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for output
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors
 */
cnmlStatus_t cnmlCreatePluginBertPreOp(
    cnmlBaseOp_t *op,
    cnmlPluginBertPreParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function creates PluginBertPreOp with proper param,
 *
 *  **Supports MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertPre parameter struct pointer.
 *  @param[in] cnml_static_ptr
 *    Input. An array of four-dimensional cnmlTensors for consts
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for input
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for output
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors
 */
cnmlStatus_t cnmlCreatePluginBertPreOp_V2(
  cnmlBaseOp_t *op,
  cnmlPluginBertPreParam_t param,
  cnmlTensor_t* cnml_static_ptr,
  cnmlTensor_t *cnml_input_ptr,
  cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertPreOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertPre parameter struct pointer.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginBertPreOpForward(
    cnmlBaseOp_t op,
    cnmlPluginBertPreParam_t param,
    void **input_addrs,
    void **output_addrs,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertPreOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginBertPreOpForward_V2(
  cnmlBaseOp_t op,
  cnmlTensor_t* cnml_input_ptr,
  void **input_addrs,
  cnmlTensor_t* cnml_output_ptr,
  void **output_addrs,
  cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertPreOp on CPU.
 *
 *  @param[out]  embedding_result
 *    Input. An array stores the result of embedding result
 *  @param[out]  attention_index_result
 *    Input. An array stores the result of attention mask index
 *  @param[in]  word_embedding_table
 *    Input. An array stores the word embedding table
 *  @param[in]  vocab_size
 *    Input. The size of vacab embedding table
 *  @param[in]  segment_embedding_table
 *    Input. An array stores the segment embedding table
 *  @param[in]  segment_size
 *    Input. The size of segment embedding table
 *  @param[in]  position_embedding_table
 *    Input. An array stores the position embedding table
 *  @param[in]  position_size
 *    Input. The size of positin embedding table
 *  @param[in]  layernorm_gamma
 *    Input. An array stores the layernorm gamma params
 *  @param[in]  layernorm_beta
 *    Input. An array stores the layernorm beta params
 *  @param[in]  input_ids
 *    Input. the word ids input
 *  @param[in]  token_type_ids
 *    Input. the token type ids input
 *  @param[in]  attention_mask
 *    Input. the attention mask input
 *  @param[in]  batch_num
 *    Input. The number of batch
 *  @param[in]  seq_len
 *    Input. The length of sequnce
 *  @param[in]  hidden_size
 *    Input. The size of embedding vector
 */
void cnmlCpuComputePluginBertPreOpForward(float* embedding_result,
    int* attention_index_result,
    const float* word_embedding_table,
    int vocab_size,
    const float* segment_embedding_table,
    int segment_size,
    const float* position_embedding_table,
    int position_size,
    const float* layernorm_gamma,
    const float* layernorm_beta,
    const int* input_ids,
    const int* token_type_ids,
    const uint16_t* attention_mask,
    int batch_num,
    int seq_len,
    int hidden_size);
/* ------------------------------------- */
/* cnmlPluginBertPre operation end */
/* ------------------------------------- */

/* ======================================= */
/* cnmlPluginBertTransformer operation start */
/* ======================================= */
cnmlStatus_t cnmlCreatePluginBertTransformerOp(
    cnmlBaseOp_t *op,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors,
    cnmlTensor_t *cnml_static_tensors,
    int static_tensors_num);

cnmlStatus_t cnmlComputePluginBertTransformerOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t* input_tensors, // default as nullptr
    void** inputs,
    cnmlTensor_t* output_tensors, // default as nullptr
    void** outputs,
    cnrtQueue_t queue,
    void *extra);

/* ------------------------------------- */
/* cnmlPluginBertTransformer operation end */
/* ------------------------------------- */

/* ======================================= */
/* cnmlPluginBertSquad operation start */
/* ======================================= */
cnmlStatus_t cnmlCreatePluginBertSquadOp(
    cnmlBaseOp_t *op,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors,
    cnmlTensor_t *cnml_static_tensors,
    int static_tensors_num,
    int batch_size,
    int seq_num);

cnmlStatus_t cnmlComputePluginBertSquadOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t* input_tensors, // default as nullptr
    void** inputs,
    cnmlTensor_t* output_tensors, // default as nullptr
    void** outputs,
    cnrtQueue_t queue,
    void *extra);

/* ------------------------------------- */
/* cnmlPluginBertSquad operation end */
/* ------------------------------------- */

/* ======================================= */
/* cnmlPluginBertBaseEncoder operation start */
/* ======================================= */
/*!
 *  @struct cnmlPluginBertBaseEncoderParam
 *  @brief A struct.
 *
 *  cnmlPluginBertBaseEncoderParam is a structure describing the "param"
 *  parameter of BertBaseEncoder operation.
 *  cnmlCreatePluginBertBaseEncoderOpParam() is used to create
 *  an instance of cnmlPluginBertBaseEncoderOpParam_t.
 *  cnmlDestroyPluginBertBaseEncoderOpParam() is used to destroy
 *  an instance of cnmlPluginBertBaseEncoderOpParam_t.
 */
struct cnmlPluginBertBaseEncoderParam {
  int batch_size;
  int seq_len;
  int head_num;
  int head_size;

  int input_num;
  int output_num;
  int static_num;

  int kernel_id;

  cnmlCoreVersion_t core_version;
};

/*! ``cnmlPluginBertBaseEncoderParam_t`` is a pointer to a
    structure (cnmlPluginBertBaseEncoderParam) holding the description of a BertBaseEncoder operation param.
*/
typedef cnmlPluginBertBaseEncoderParam *cnmlPluginBertBaseEncoderParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginBertBaseEncoderOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  batch_size
 *    Input. The number of batch
 *  @param[in]  seq_len
 *    Input. The length of sequnce
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginBertBaseEncoderOpParam(
    cnmlPluginBertBaseEncoderParam_t *param,
    cnmlCoreVersion_t core_version,
    int batch_size,
    int seq_len);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginBertBaseEncoderParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBertBaseEncoder operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBertBaseEncoderOpParam(
    cnmlPluginBertBaseEncoderParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginBertBaseEncoderOp with proper param,
 *
 *  **Supports MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertBaseEncoder parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for inputs
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for outputs
 *  @param[in] cnml_static_ptr
 *    Input. An array of four-dimensional cnmlTensors for consts
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors
 *    - tensor shapes does not meet reuqirements
 */
cnmlStatus_t cnmlCreatePluginBertBaseEncoderOp(
    cnmlBaseOp_t *op,
    cnmlPluginBertBaseEncoderParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    cnmlTensor_t *cnml_output_ptr,
    cnmlTensor_t *cnml_static_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertBaseEncoderOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertBaseEncoder parameter struct pointer.
 *  @param[in]  cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for inputs
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors
 *  @param[in]  cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for outputs
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginBertBaseEncoderOpForward(
    cnmlBaseOp_t op,
    cnmlPluginBertBaseEncoderParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    void **input_addrs,
    cnmlTensor_t *cnml_output_ptr,
    void **output_addrs,
    cnrtQueue_t queue);
/* ------------------------------------- */
/* cnmlPluginBertBaseEncoder operation end */
/* ------------------------------------- */

#endif
