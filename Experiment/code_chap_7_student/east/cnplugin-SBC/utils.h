/*************************************************************************
 * Copyright (C) [2018] by Cambricon, Inc.
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

#ifndef __UTILS_H
#define __UTILS_H

#include <iostream>
#include "cnrt.h"

typedef unsigned short half;

inline void cnrtConvertFloatToHalfArray(uint16_t* x, const float* y, int len) {
  for (int i = 0; i < len; i++) {
    cnrtConvertFloatToHalf(x + i, y[i]);
  }
}

inline void cnrtConvertHalfToFloatArray(float* x, const uint16_t* y, int len) {
  for (int i = 0; i < len; i++) {
    cnrtConvertHalfToFloat(x + i, y[i]);
  }
}

inline void cnrtConvertFloatToHalfArray(uint16_t* x, float* y, int len) {
  std::cout<<"yy"<<std::endl;
  for (int i = 0; i < len; i++) {
    cnrtConvertFloatToHalf(x + i, y[i]);
  }
}

inline void cnrtConvertHalfToFloatArray(float* x, uint16_t* y, int len) {
  for (int i = 0; i < len; i++) {
    cnrtConvertHalfToFloat(x + i, y[i]);
  }
}

inline void cnrtMallocAndMemcpy(int* mlu_a, int* a, int len) {
  cnrtMalloc((void**)&mlu_a, len * sizeof(int));
  cnrtMemcpy(mlu_a, a, len * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);
}

inline void cnrtMemcpyFloatToHalf(half* mlu_a, const float* a, const int len) {
  half* half_a = (half*)malloc(len * sizeof(half));
  cnrtConvertFloatToHalfArray(half_a, a, len);
  cnrtMemcpy((void*)mlu_a, (void*)half_a, len * sizeof(half),
             CNRT_MEM_TRANS_DIR_HOST2DEV);
  free(half_a);
}

inline void cnrtMemcpyHalfToFloat(float* a, const half* mlu_a, const int len) {
  half* half_a = (half*)malloc(len * sizeof(half));
  cnrtMemcpy((void*)half_a, (void*)mlu_a, len * sizeof(half),
             CNRT_MEM_TRANS_DIR_DEV2HOST);
  cnrtConvertHalfToFloatArray(a, half_a, len);
  free(half_a);
}

#endif /*__UTILS_H*/
