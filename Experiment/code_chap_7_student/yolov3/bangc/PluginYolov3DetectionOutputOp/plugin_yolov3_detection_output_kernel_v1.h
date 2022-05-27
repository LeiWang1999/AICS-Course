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
#ifndef __YOLOV3_KERNEL_H__
#define __YOLOV3_KERNEL_H__

#ifdef __cplusplus
extern "C" {
#endif
  void yolov3Kernel_MLU270(uint16_t *predicts,
                           void *input0,
                           void *input1,
                           void *input2,
                           void *input3,
                           void *input4,
                           void *input5,
                           void *input6,
                           void *buffer_gdram,
                           int *h_arr,
                           int *w_arr,
                           uint16_t *biases,
                           int num_inputs,
                           int num_classes,
                           int batchNum,
                           int num_mask_groups,
                           int num_max_boxes,
                           int PAD_SIZE,
                           int netw,
                           int neth,
                           uint16_t confidence_thresh,
                           uint16_t nms_thresh);

  void yolov3Kernel_MLU220(uint16_t *predicts,
                           void *input0,
                           void *input1,
                           void *input2,
                           void *input3,
                           void *input4,
                           void *input5,
                           void *input6,
                           void *buffer_gdram,
                           int *h_arr,
                           int *w_arr,
                           uint16_t *biases,
                           int num_inputs,
                           int num_classes,
                           int batchNum,
                           int num_mask_groups,
                           int num_max_boxes,
                           int PAD_SIZE,
                           int netw,
                           int neth,
                           uint16_t confidence_thresh,
                           uint16_t nms_thresh);

#ifdef __cplusplus
}
#endif

#endif  // __YOLOV3_KERNEL_H__
