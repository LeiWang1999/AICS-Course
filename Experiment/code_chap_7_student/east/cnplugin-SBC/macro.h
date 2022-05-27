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

#define USE_NRAM 0
#define USE_SVINST 0
#define USE_MULTICORE 0
#define NUM_MULTICORE 16

#define CHANNELS 3
#define HEIGHT 672
#define WIDTH 1280
#define BATCH_SIZE 1
#define HW HEIGHT*WIDTH

#define DATA_COUNT ((CHANNELS) * (WIDTH) * (HEIGHT))
#define NUM_PER_LOOP (NUM_MULTICORE)
#define ALIGN_SIZE 64
#define CWH_PLUS (((DATA_COUNT - 1) / ALIGN_SIZE + 1) * ALIGN_SIZE)
#define WH_PLUS ((((HEIGHT*WIDTH) - 1) / ALIGN_SIZE + 1) * ALIGN_SIZE)

#define HW_SPLIT ((((HEIGHT*WIDTH/16) - 1) / ALIGN_SIZE + 1) * ALIGN_SIZE)
#define HWC_SPLIT ((((HEIGHT*WIDTH/16) - 1) / ALIGN_SIZE + 1) * ALIGN_SIZE)*CHANNELS



