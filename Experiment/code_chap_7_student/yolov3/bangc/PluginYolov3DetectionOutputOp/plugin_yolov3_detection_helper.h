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
#define PAD_UP(x, y) (x / y + (int)(x % y > 0)) * y
#define PAD_DN(x, y) (x / y) * y
#define T half
#define SRAM_BUFFER_SIZE (2 * 1024 * 1024 - 64 * 4)
#if __BANG_ARCH_ > 220
  #define NRAM_BUFFER_SIZE 224 * 1024   // BUFFER_SIZE for MLU270
#elif __BANG_ARCH__ >= 200
  #define NRAM_BUFFER_SIZE 224 * 1024   // BUFFER_SIZE for MLU220
#else
  #define NRAM_BUFFER_SIZE 448 * 1024   // BUFFER_SIZE for MLU100
#endif
#define C_PAD_SIZE 64 // PAD_SIZE because of c_inst constrain
#define LINESIZE 256
#define TEMP_DST_STRIDE 2048
#define OUTPUT_BUFFER_SIZE 256
#define LINESIZE 256

/*!
 *  @brief a stride loading function
 *
 *  This function perform stride loading
 *  from GDRAM to NRAM for data with sizeof 2B
 *
 *  @param[out] dst
 *    Output. The dst ptr on NRAM.
 *  @param[in] src
 *    Input. The src ptr on GDRAM.
 *  @param[in] bytes
 *    Input. The number of bytes copied in single iteration.
 *  @param[in] dstStride
 *    Input. The dst stride, unit is number of data, not bytes.
 *  @param[in] srcStride
 *    Input. The src stride, unit is number of data, not bytes.
 *  @param[in] segNum
 *    Input. The number of itration.
 */
__mlu_func__ void strideLoad(
  half* dst,
  half* src,
  int bytes,
  int dstStride,
  int srcStride,
  int segNum) {
  for (int seg = 0; seg < segNum; seg++) {
    __memcpy(dst + seg * dstStride,
             src + seg * srcStride,
             bytes, GDRAM2NRAM);
  }
}

/*!
 *  @brief a stride storing function
 *
 *  This function perform stride storing
 *  from NRAM to GDRAM for data with sizeof 2B
 *
 *  @param[out] dst
 *    Output. The dst ptr on GDRAM.
 *  @param[in] src
 *    Input. The src ptr on NRAM.
 *  @param[in] bytes
 *    Input. The number of bytes copied in single iteration.
 *  @param[in] dstStride
 *    Input. The dst stride, unit is number of data, not bytes.
 *  @param[in] srcStride
 *    Input. The src stride, unit is number of data, not bytes.
 *  @param[in] segNum
 *    Input. The number of itration.
 */
__mlu_func__ void strideStore(
  half* dst,
  half* src,
  int32_t bytes,
  int32_t dstStride,
  int32_t srcStride,
  int32_t segNum) {
  for (int seg = 0; seg < segNum; seg++) {
    __memcpy(dst + seg * dstStride,
             src + seg * srcStride,
             bytes, NRAM2GDRAM);
  }
}

/*!
 *  @brief generateOffset
 *
 *  This function generates x/yOffset used in xywh calculation
 *  For a certain h, w:
 *    xOffset: (0 1 2 ... w-1) (0 1 2 ... w-1)
 *    yOffset: (0 0 0 ... 0) (1 1 1 ... 1) ...
 *
 *  @param[out] xOffset
 *    Output. The xOffset used in x-coord calculation.
 *  @param[out] yOffset
 *    Output. The yOffset used in y-coord calculation.
 *  @param[in] h
 *    Input. The (H)eight of input tensor
 *  @param[in] w
 *    Input. The (W)idth of input tensor
 */
__mlu_func__ void generateOffset(
  half* xOffset,
  half* yOffset,
  int h,
  int w) {
  for (int hIdx = 0; hIdx < h; hIdx++) {
    for (int wIdx = 0; wIdx < w; wIdx++) {
      xOffset[hIdx * w + wIdx] = wIdx;
      yOffset[hIdx * w + wIdx] = hIdx;
    }
  }
}

/*!
 *  @brief loadEntry.
 *  This function uses "strideLoad" to load xywh data
 *  from gdram and preform the "decoing" process to
 *  generate true xywh coordinates according to:
 *    x = (col + sigmoid(tx)) / l
 *    y = (row + sigmoid(ty)) / l
 *    w = (bias * exp(tw)) / netw
 *    h = (bias * exp(th)) / neth
 *
 *  @param[out] dst
 *    Output. The result after decoding process.
 *  @param[in] src
 *    Input. The src buffer used to store data from gdram.
 *  @param[in] inputs
 *    Input. Input ptrs on gdram.
 *  @param[in] hw_arr
 *    Input. h/w_arr for the corresponding input tensor.
 *  @param[in] biases
 *    Input. Biases of anchors used in h/w calculation.
 *  @param[in] boxMask
 *    Input. The mask used to select qualified box.
 *  @param[in] offsetPtr
 *    Input. The offset generated by "generateOffset" func.
 *  @param[in] temp
 *    Input. The temp buffer.
 *  @param[in] segNum
 *    Input. The number of 256-segment for a input tensor.
 *  @param[in] remain
 *    Input. The number of remain-segment for a input tensor.
 *  @param[in] remainPad
 *    Input. remainPad = PAD_UP(remain, PAD_SIZE).
 *  @param[in] batchNum
 *    Input. The number of batch.
 *  @param[in] dealNum
 *    Input. The actual number of data that are dealing with.
 *  @param[in] num_mask_groups
 *    Input. Num of anchors, assuming same for all input tensors.
 *  @param[in] segSize
 *    Input. Size of each segment.
 *  @param[in] remSize
 *    Input. Size of ramain part.
 *  @param[in] srcOffset
 *    Input. The offset in src ptr.
 *  @param[in] dstOffset
 *    Input. The offset in dst ptr.
 *  @param[in] batchId
 *    Input. The index of batch.
 *  @param[in] inputId
 *    Input. The index of input.
 *  @param[in] entry
 *    Input. x:0 | y:1 | w:2 | h:3.
 *  @param[in] maskCountPad
 *    Input. The number of qualified box after padding.
 *  @param[in] nethw
 *    Input. The size of network input image.
 *  @param[in] nethw
 *    Input. The new size of network input image after correction.
 */
__mlu_func__ void loadEntry(
  half* dst,
  half* src,
  void** inputs,
  int* hw_arr,
  half* biases,
  half* boxMask,
  half* offsetPtr,
  half* temp,
  int hw,
  int segNum,
  int remain,
  int remainPad,
  int batchNum,
  int dealNum,
  int num_mask_groups,
  int segSize,
  int remSize,
  int srcOffset,
  int dstOffset,
  int batchId,
  int inputId,
  int entry,
  int maskCountPad,
  int nethw,
  int newhw) {
  PRINTF_SCALAR("===== Loading entry: %d =====\n", entry);

  half* dataPtr = src;
  int biasIdx = entry % 2;
  int dstStride = PAD_UP(remainPad, 64);
  int batchSize = remSize / remainPad * hw * num_mask_groups;
  int srcStride = remSize / remainPad * hw;

  if (segNum > 0) {
    for (int segId = 0; segId < segNum; segId++) {
          int offset_src = batchId * batchSize + segId * LINESIZE;
          int offset_dst = segId * num_mask_groups * LINESIZE;
          strideLoad(
            src + offset_dst,
            ((half*)inputs[inputId]) + offset_src + entry * hw,
            LINESIZE * sizeof(half),
            LINESIZE,
            srcStride,
            num_mask_groups);
    }
  }
  if (remain > 0) {
        int offset_src = batchId * batchSize + segNum * LINESIZE;
        int offset_dst = segNum * num_mask_groups * LINESIZE;
        strideLoad(
          src + offset_dst,
          ((half*)inputs[inputId]) + offset_src + entry * hw,
          remain * sizeof(half),
          dstStride,
          srcStride,
          num_mask_groups);
  }

  PRINTF_VECTOR("----- before sigmoid -----\n",
                "%hf ", src, dealNum);
  if (entry < 2) {
    // sigmoid(t)
    __bang_active_sigmoid(src, src, dealNum);

    PRINTF_VECTOR("----- after sigmoid -----\n",
                  "%hf ", src, dealNum);

    // i + sigmoid(t)
    for (int segIdx = 0; segIdx < segNum; segIdx++) {
      for (int maskIdx = 0; maskIdx < num_mask_groups; maskIdx++) {
        __bang_add(dataPtr, dataPtr, offsetPtr, LINESIZE);
        dataPtr += LINESIZE;
      }
      offsetPtr += LINESIZE;
    }
    for (int maskIdx = 0; maskIdx < num_mask_groups; maskIdx++) {
      __bang_add(dataPtr, dataPtr, offsetPtr, dstStride);
      dataPtr += dstStride;
    }
    PRINTF_VECTOR("----- after offseting -----\n",
                  "%hf ", src, dealNum);

    // (i + sigmoid(tx)) / l
    __nramset_half(temp, 64, (half)1.0 / (half)hw_arr[inputId]);
    __bang_cycle_mul(src, src, temp, dealNum, 64);

    #ifdef CORRECT_ENABLED
    // correct yolobox: (x - (netw - new_w) / 2 / netw) / (new_w / netw)
    __nramset_half(temp, 64, ((half)nethw - newhw) / 2.0 / (half)nethw);
    __bang_cycle_sub(src, src, temp, dealNum, 64);
    __nramset_half(temp, 64, (half)nethw / newhw);
    __bang_cycle_mul(src, src, temp, dealNum, 64);
    #endif
  } else {
    PRINTF_VECTOR("----- before exp -----\n",
                  "%hf ", src, dealNum);
    // exp(t)
    __bang_active_exp(src, src, dealNum);
    PRINTF_VECTOR("----- after exp -----\n",
                  "%hf ", src, dealNum);

    // bias * exp(t) / net
    for (int segIdx = 0; segIdx < segNum; segIdx++) {
      for (int maskIdx = 0; maskIdx < num_mask_groups; maskIdx++) {
        half biasValue = biases[6 * inputId + 2 * maskIdx + biasIdx] / nethw;
        __nramset_half(temp, 64, biasValue);
        __bang_cycle_mul(dataPtr, dataPtr, temp, LINESIZE, 64);
        dataPtr += LINESIZE;
      }
    }
    for (int maskIdx = 0; maskIdx < num_mask_groups; maskIdx++) {
      half biasValue = biases[6 * inputId + 2 * maskIdx + biasIdx] / nethw;
      __nramset_half(temp, 64, biasValue);
      __bang_cycle_mul(dataPtr, dataPtr, temp, PAD_UP(remain, 64), 64);
      dataPtr += remain;
    }
    #ifdef CORRECT_ENABLED
    // correct yolobox: w *= netw / new_w
    __nramset_half(temp, 64, (half)nethw / newhw);
    __bang_printf("nethw / newhw: %hf\n", (half)nethw / newhw);
    __bang_cycle_mul(src, src, temp, dealNum, 64);
    #endif
  }
  __bang_collect(dst + dstOffset, src, boxMask + srcOffset, dealNum);
}

__mlu_func__ int DecodeAllBBoxesFullW(
  T* dst,
  T* src,
  T* srcTrans,
  T* src_gdram,
  T* temp,
  T* biases,
  T* offset_w,
  T* offset_h,
  int inputIdx,
  int anchorIdx,
  int h,
  int w,
  int num_entries,
  int entryPad,
  int limit,
  int segNum,
  int remain,
  int dealNum,
  int num_inputs,
  int num_classes,
  int num_mask_groups,
  int netw,
  int neth,
  mluMemcpyDirection_t dir) {
  int boxCount  = 0;
  int segCount  = 0;
  int segSize   = limit * w * num_entries * num_mask_groups;
  int dataSize  = num_entries * sizeof(T);
  int dstStride = entryPad * sizeof(T);
  int srcStride = num_entries * num_mask_groups * sizeof(T);
  int segment = limit * w - 1;
  for (int segIdx = 0; segIdx < segNum; segIdx++) {
    __memcpy(src,
             src_gdram + segIdx * segSize,
             dataSize,
             GDRAM2NRAM,
             dstStride,
             srcStride,
             segment);

    __bang_transpose(srcTrans, src, dealNum, entryPad);
    for (int i = limit * w; i < dealNum; i++) {
      srcTrans[i + 4 * dealNum] = -999;
    }

    // x & y
    __bang_active_sigmoid(srcTrans,
                          srcTrans,
                          2 * dealNum);
    __bang_add(srcTrans,
               srcTrans,
               offset_w,
               dealNum);
    __bang_add(srcTrans + dealNum,
               srcTrans + dealNum,
               offset_h,
               dealNum);
    __bang_mul_const(srcTrans,
                     srcTrans,
                     1.0 / w,
                     dealNum);
    __bang_mul_const(srcTrans + dealNum,
                     srcTrans + dealNum,
                     1.0 / h,
                     dealNum);
    #if T == half
    __nramset_half(temp + 64, 64, limit);
    #else
    __nramset_float(temp + 64, 64, limit);
    #endif
    __bang_cycle_add(offset_h, offset_h, temp + 64, dealNum, C_PAD_SIZE);

    // w & h
    T biasW = biases[2 * num_mask_groups * inputIdx +
                     2 * anchorIdx + 0] / netw;
    T biasH = biases[2 * num_mask_groups * inputIdx +
                     2 * anchorIdx + 1] / neth;
    __bang_active_exp(srcTrans + 2 * dealNum,
                      srcTrans + 2 * dealNum,
                      2 * dealNum);
    __bang_mul_const(srcTrans + dealNum * 2,
                     srcTrans + dealNum * 2,
                     biasW,
                     dealNum);
    __bang_mul_const(srcTrans + dealNum * 3,
                     srcTrans + dealNum * 3,
                     biasH,
                     dealNum);

    __bang_active_sigmoid(srcTrans + 4 * dealNum,
                          srcTrans + 4 * dealNum,
                          (num_entries - 4) * dealNum);
    __bang_cycle_mul(srcTrans + 5 * dealNum,
                     srcTrans + 5 * dealNum,
                     srcTrans + 4 * dealNum,
                     (num_entries - 5) * dealNum,
                     dealNum);
    __bang_write_zero(src, dealNum * num_entries);
    __bang_cycle_gt(src,
                    srcTrans + 4 * dealNum,
                    temp,
                    dealNum,
                    C_PAD_SIZE);
    __bang_cycle_add(src + dealNum, src + dealNum, src, dealNum * (num_entries - 1), dealNum);
    __bang_count((uint32_t*)temp + 16 * sizeof(T), src, dealNum);
    segCount = ((uint32_t*)temp)[16* sizeof(T)];
    // PRINTF_SCALAR("segCount: %d\n", segCount);
    // PRINTF_SCALAR("boxCount: %d\n", boxCount);
    if (segCount > 0) {
      __bang_collect(src, srcTrans, src, dealNum * num_entries);
      // PRINTF_VECTOR("----- obj -----", "%hf ", srcTrans + 4 * segCount, segCount);
      // PRINTF_VECTOR("----- X -----", "%hf ", srcTrans + 5 * segCount, segCount);
      // PRINTF_VECTOR("----- Y -----", "%hf ", srcTrans + 1 * segCount, segCount);
      // PRINTF_VECTOR("----- W -----", "%hf ", srcTrans + 2 * segCount, segCount);
      // PRINTF_VECTOR("----- H -----", "%hf ", srcTrans + 3 * segCount, segCount);
      __memcpy(dst + boxCount,
               src,
               segCount * sizeof(T),
               dir,
               TEMP_DST_STRIDE * sizeof(T),
               segCount * sizeof(T),
               num_entries - 1);
      boxCount += segCount;
    }
  }
  return boxCount;
}

__mlu_func__ int DecodeAllBBoxesPartW(
  T* dst,
  T* src,
  T* srcTrans,
  T* src_gdram,
  T* temp,
  T* biases,
  T* offset_w,
  T* offset_h,
  int inputIdx,
  int anchorIdx,
  int h,
  int w,
  int num_entries,
  int entryPad,
  int limit,
  int segNum,
  int remain,
  int dealNum,
  int num_inputs,
  int num_classes,
  int num_mask_groups,
  int netw,
  int neth,
  mluMemcpyDirection_t dir) {
  int boxCount  = 0;
  int segCount  = 0;
  int segSize   = limit * num_entries * num_mask_groups;
  int dataSize  = num_entries * sizeof(T);
  int dstStride = entryPad * sizeof(T);
  int srcStride = num_entries * num_mask_groups * sizeof(T);
  int segment = limit - 1;
  for (int segIdx = 0; segIdx < segNum; segIdx++) {
    __memcpy(src,
             src_gdram + segIdx * segSize,
             dataSize,
             GDRAM2NRAM,
             dstStride,
             srcStride,
             segment);

    __bang_transpose(srcTrans, src, dealNum, entryPad);
    for (int i = limit; i < dealNum; i++) {
      srcTrans[i + 4 * dealNum] = -999;
    }
    // PRINTF_SCALAR("===== obj check before sigmoid =====\n");
    // for (int i = 0; i < dealNum; i++) {
    //   printf("%hf ", srcTrans[i + 4 * dealNum]);
    // }
    // printf("\n\n");

    // x & y
    __bang_active_sigmoid(srcTrans,
                          srcTrans,
                          2 * dealNum);
    __bang_add(srcTrans,
               srcTrans,
               offset_w,
               dealNum);
    __bang_add(srcTrans + dealNum,
               srcTrans + dealNum,
               offset_h,
               dealNum);
    __bang_mul_const(srcTrans,
                     srcTrans,
                     1.0 / w,
                     dealNum);
    __bang_mul_const(srcTrans + dealNum,
                     srcTrans + dealNum,
                     1.0 / h,
                     dealNum);
    #if T == half
    __nramset_half(temp + 64, 64, limit);
    #else
    __nramset_float(temp + 64, 64, limit);
    #endif
    __bang_cycle_add(offset_w, offset_w, temp + 64, dealNum, C_PAD_SIZE);

    // w & h
    T biasW = biases[2 * num_mask_groups * inputIdx +
                     2 * anchorIdx + 0] / netw;
    T biasH = biases[2 * num_mask_groups * inputIdx +
                     2 * anchorIdx + 1] / neth;
    __bang_active_exp(srcTrans + 2 * dealNum,
                      srcTrans + 2 * dealNum,
                      2 * dealNum);
    __bang_mul_const(srcTrans + dealNum * 2,
                     srcTrans + dealNum * 2,
                     biasW,
                     dealNum);
    __bang_mul_const(srcTrans + dealNum * 3,
                     srcTrans + dealNum * 3,
                     biasH,
                     dealNum);

    __bang_active_sigmoid(srcTrans + 4 * dealNum,
                          srcTrans + 4 * dealNum,
                          (num_entries - 4) * dealNum);
    __bang_cycle_mul(srcTrans + 5 * dealNum,
                     srcTrans + 5 * dealNum,
                     srcTrans + 4 * dealNum,
                     (num_entries - 5) * dealNum,
                     dealNum);
    __bang_write_zero(src, dealNum * num_entries);
    __bang_cycle_gt(src,
                    srcTrans + 4 * dealNum,
                    temp,
                    dealNum,
                    C_PAD_SIZE);
    __bang_cycle_add(src + dealNum, src + dealNum, src, dealNum * (num_entries - 1), dealNum);
    __bang_count((uint32_t*)temp + 16 * sizeof(T), src, dealNum);
    segCount = ((uint32_t*)temp)[16* sizeof(T)];
    PRINTF_SCALAR("segCount: %d\n", segCount);
    PRINTF_SCALAR("boxCount: %d\n", boxCount);
    if (segCount > 0) {
      __bang_collect(src, srcTrans, src, dealNum * num_entries);
      // PRINTF_VECTOR("----- obj -----", "%hf ", srcTrans + 4 * segCount, segCount);
      // PRINTF_VECTOR("----- X -----", "%hf ", srcTrans + 5 * segCount, segCount);
      // PRINTF_VECTOR("----- Y -----", "%hf ", srcTrans + 1 * segCount, segCount);
      // PRINTF_VECTOR("----- W -----", "%hf ", srcTrans + 2 * segCount, segCount);
      // PRINTF_VECTOR("----- H -----", "%hf ", srcTrans + 3 * segCount, segCount);
      __memcpy(dst + boxCount,
               src,
               segCount * sizeof(T),
               dir,
               TEMP_DST_STRIDE * sizeof(T),
               segCount * sizeof(T),
               num_entries - 1);
      boxCount += segCount;
    }
  }
  return boxCount;
}
