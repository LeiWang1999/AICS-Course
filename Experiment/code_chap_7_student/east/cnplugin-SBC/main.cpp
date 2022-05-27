#include <float.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <iostream>

#include "macro.h"
#include "cnrt.h"
#include "utils.h"

#define CHANNELS 3
#define HEIGHT 672
#define WIDTH 1280
#define BATCH_SIZE 1
#define DATA_COUNT ((CHANNELS) * (WIDTH) * (HEIGHT))

using namespace std;
typedef unsigned short half;

extern "C" {
    void SBCKernel(half* input_data_,half* output_data_ ,int batch_num_, int core_num_);
}

int main() {

    const int data_count = DATA_COUNT*BATCH_SIZE;
    int batch_num_ = BATCH_SIZE;
    int core_num_ = NUM_MULTICORE;
    const int channels_ = CHANNELS;
    const int height_ = HEIGHT;
    const int width_ = WIDTH;

    //开辟CPU 内存
    float* data = (float*)malloc(data_count * sizeof(float));
    
    //读取数据文件
    FILE* f_data = fopen("data.txt","r");

    if (f_data == NULL) {
        printf("Open file fail!\n");
        return 0;
    }

    int f1 = 0;
    for (int i = 0; i < data_count; i++) {
        f1 = fscanf(f_data, "%f\n", &data[i]);
    }

    fclose(f_data);
    if(f1 < 0) return 0;

    //初始化设备
    cnrtInit(0);
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev);
    cnrtQueue_t pQueue;
    cnrtCreateQueue(&pQueue);
    cnrtDim3_t dim;
    cnrtFunctionType_t c;
    
    //选择 UNION 模式
    switch (core_num_) {
    case 1:
        c = CNRT_FUNC_TYPE_BLOCK;
        printf("task type = BLOCK\n");
        break;
    case 4:
        c = CNRT_FUNC_TYPE_UNION1;
        printf("task type = UNION1\n");
        break;
    case 16:
        c = CNRT_FUNC_TYPE_UNION4;
        printf("task type = UNION4\n");
        break;
    default:
        exit(-1);
    }

    dim.x = core_num_;
    dim.y = 1;
    dim.z = 1;

    vector<float> input_data;
    vector<float> output_data;

    half *data_mlu, *out_data;

    //float2half
    CNRT_CHECK(cnrtMalloc((void**)&data_mlu, data_count * sizeof(half)));
    CNRT_CHECK(cnrtMalloc((void**)&out_data, data_count * sizeof(half)));

    cnrtMemcpyFloatToHalf(data_mlu, data, data_count);

    // Passing param
    cnrtKernelParamsBuffer_t params;
    cnrtGetKernelParamsBuffer(&params);
    cnrtKernelParamsBufferAddParam(params, &data_mlu, sizeof(half*));
    cnrtKernelParamsBufferAddParam(params, &out_data, sizeof(half*));
    cnrtKernelParamsBufferAddParam(params, &batch_num_, sizeof(int));

    // create cnrt Notifier
    cnrtRet_t ret;
    cnrtNotifier_t  Notifier_start;
    cnrtNotifier_t  Notifier_end;
    cnrtCreateNotifier(&Notifier_start);
    cnrtCreateNotifier(&Notifier_end);
    float timeTotal = 0.0;
    struct timeval start;
    struct timeval end;

    gettimeofday(&start, NULL);

    // hardware time
    cnrtPlaceNotifier(Notifier_start, pQueue);
    CNRT_CHECK(cnrtInvokeKernel_V2((void*)&SBCKernel, dim, params, c, pQueue));
    cnrtPlaceNotifier(Notifier_end, pQueue);
    CNRT_CHECK(cnrtSyncQueue(pQueue));

    gettimeofday(&end, NULL);
    float time_use = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec))/1000.0;
    printf("time use: %.3f ms\n", time_use);
 
    float* output_tmp = (float*)malloc(data_count * sizeof(float));
    cnrtMemcpyHalfToFloat(output_tmp, out_data, data_count);

    // save data
    FILE* mluOutputFile = fopen("./mluoutput.txt", "w");
    for (int i = 0; i < data_count; i++) {
        fprintf(mluOutputFile, "%f\n", output_tmp[i]);
    }
    fclose(mluOutputFile);

    //free
    CNRT_CHECK(cnrtFree(data_mlu));
    CNRT_CHECK(cnrtFree(out_data));
    CNRT_CHECK(cnrtDestroyQueue(pQueue));
    CNRT_CHECK(cnrtDestroyKernelParamsBuffer(params));
    cnrtDestroyNotifier(&Notifier_start);
    cnrtDestroyNotifier(&Notifier_end);
    cnrtDestroy();

    free(data);
    free(output_tmp);

}