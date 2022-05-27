// #include "inference.h"
// #include "cnrt.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include "stdlib.h"
// #include <sys/time.h>
// #include <time.h>

// namespace StyleTransfer{

// Inference :: Inference(std::string offline_model){
//     offline_model_ = offline_model;
// }

// void Inference :: run(DataTransfer* DataT){
    
//     cnrtInit(0);

//     // prepare model name
//     char fname[100] = "../../models/offline_models/";
//     // The name parameter represents the name of the offline model file.
//     // It is also the name of a function in the offline model file.
//     strcat(fname, DataT->model_name.c_str());
//     strcat(fname, ".cambricon");

//     // load model
//     cnrtModel_t model;
//     cnrtLoadModel(&model, fname);

//     cnrtDev_t dev;
//     cnrtGetDeviceHandle(&dev, 0);
//     cnrtSetCurrentDevice(dev);

//     // get model total memory
//     int64_t totalMem;
//     cnrtGetModelMemUsed(model, &totalMem);
//     printf("total memory used: %ld Bytes\n", totalMem);
//     // get model parallelism
//     int model_parallelism;
//     cnrtQueryModelParallelism(model, &model_parallelism);
//     printf("model parallelism: %d.\n", model_parallelism);

//     // load extract function
//     cnrtFunction_t function;
//     cnrtCreateFunction(&function);
//     cnrtExtractFunction(&function, model, "subnet0");

//     int inputNum, outputNum;
//     int64_t *inputSizeS, *outputSizeS;
//     cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
//     cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);

//     // prepare data on cpu
//     void **inputCpuPtrS = (void **)malloc(inputNum * sizeof(void *));
//     void **outputCpuPtrS = (void **)malloc(outputNum * sizeof(void *));

//     // allocate I/O data memory on MLU
//     void **inputMluPtrS = (void **)malloc(inputNum * sizeof(void *));
//     void **outputMluPtrS = (void **)malloc(outputNum * sizeof(void *));

//     // prepare input buffer
//     uint16_t *input_half;
//     for (int i = 0; i < inputNum; i++) {
//           // converts data format when using new interface model
//         inputCpuPtrS[i] = malloc(inputSizeS[i]);
//         input_half = (uint16_t *)malloc(inputSizeS[i] * 1);
//         int length = inputSizeS[i] / 2;
//         for (int j = 0; j < length; j++)
//             cnrtConvertFloatToHalf(input_half + j, DataT->input_data[j]);
//         cnrtReshapeNCHWToNHWC(inputCpuPtrS[i], input_half, 1, 256, 256, 3, cnrtDataType_t(0x12));
//           // malloc mlu memory
//         cnrtMalloc(&(inputMluPtrS[i]), inputSizeS[i]);
//         cnrtMemcpy(inputMluPtrS[i], inputCpuPtrS[i], inputSizeS[i], CNRT_MEM_TRANS_DIR_HOST2DEV);
//     }
    

//     // prepare output buffer
//     float *output_temp;
//     for (int i = 0; i < outputNum; i++)
//         output_temp = new float[256 * 256 * 3];

//     // prepare parameters for cnrtInvokeRuntimeContext
//     void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
//     for (int i = 0; i < inputNum; ++i) {
//           param[i] = inputMluPtrS[i];
//     }
//     for (int i = 0; i < outputNum; ++i) {
//           param[inputNum + i] = outputMluPtrS[i];
//     }

//     // setup runtime ctx
//     cnrtRuntimeContext_t ctx;
//     cnrtCreateRuntimeContext(&ctx, function, NULL);

//     // bind device
//     cnrtSetRuntimeContextDeviceId(ctx, 0);
//     cnrtInitRuntimeContext(ctx, NULL);
//     // compute offline
//     cnrtQueue_t queue;
//     cnrtRuntimeContextCreateQueue(ctx, &queue);
//     // invoke
//     cnrtInvokeRuntimeContext(ctx, param, queue, NULL);
//     // sync
//     cnrtSyncQueue(queue);

//     for (int i = 0; i < outputNum; i++) {
//         cnrtMemcpy(outputCpuPtrS[i], outputMluPtrS[i], outputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2HOST);
//         // convert to float
//         int length = outputSizeS[i] / 2;
//         printf("%d\n", length);
//         uint16_t *outputCpu = ((uint16_t **)outputCpuPtrS)[0];
//         DataT->output_data = new float[256 * 256 * 3];
//         printf("1\n");
//         for (int j = 0; j < length; j++)
//         {
//             cnrtConvertHalfToFloat(output_temp+j, outputCpu[j]);
//         }
//         printf("2\n");
//         cnrtReshapeNHWCToNCHW(DataT->output_data, output_temp, 1, 256, 256, 3, cnrtDataType_t(0x13));
//     }
//     // copy mlu result to cpu
//     for (int i = 0; i < outputNum; i++) {
//           cnrtMemcpy(outputCpuPtrS[i], outputMluPtrS[i], outputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2HOST);
//     }

//     // free memory space
//     for (int i = 0; i < inputNum; i++) {
//           free(inputCpuPtrS[i]);
//           cnrtFree(inputMluPtrS[i]);
//     }
//     for (int i = 0; i < outputNum; i++) {
//           free(outputCpuPtrS[i]);
//           cnrtFree(outputMluPtrS[i]);
//     }
//     free(inputCpuPtrS);
//     free(outputCpuPtrS);
//     free(param);
//     free(input_half);
//     delete output_temp;
//     cnrtDestroyQueue(queue);
//     cnrtDestroyRuntimeContext(ctx);
//     cnrtDestroyFunction(function);
//     cnrtUnloadModel(model);
//     cnrtDestroy();

// }

// } // namespace StyleTransfer


#include "inference.h"
#include "cnrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdlib.h"
#include <sys/time.h>
#include <time.h>

namespace StyleTransfer{

typedef unsigned short half;


Inference :: Inference(std::string offline_model){
    offline_model_ = offline_model;
}

void Inference :: run(DataTransfer* DataT){
    cnrtInit(0);
    // load model
    cnrtModel_t model;
    cnrtLoadModel(&model, offline_model_.c_str());

    // set current device
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev);
    
    float* input_data = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
    float* output_data = reinterpret_cast<float*>(malloc(256*256*3*sizeof(float)));
    int t = 256*256;
    for(int i=0;i<t;i++)
        for(int j=0;j<3;j++)
            input_data[i*3+j] = DataT->input_data[t*j+i];      

    int number = 0;
    cnrtGetFunctionNumber(model, &number);


    // load extract function
    cnrtFunction_t function;
    if (CNRT_RET_SUCCESS != cnrtCreateFunction(&function)) {
      printf("cnrtCreateFunction Failed!\n");
      exit(-1);
    }
    
    if (CNRT_RET_SUCCESS != cnrtExtractFunction(&function, model, "subnet0")) {
      printf("cnrtExtractFunction Failed!\n");
      exit(-1);
    }
    

    int inputNum, outputNum;
    int64_t *inputSizeS, *outputSizeS;
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);  // prepare data on cpu


    DataT->output_data = reinterpret_cast<float*>(malloc(256 * 256 * 3 * sizeof(float)));
    half* input_half = (half*)malloc(256 * 256 * 3 * sizeof(half));
    half* output_half = (half*)malloc(256 * 256 * 3 * sizeof(half));
    for (int i=0;i<256*256*3;i++)
        cnrtConvertFloatToHalf(input_half+i,input_data[i]);
    for (int i=0;i<256*256*3;i++)
        cnrtConvertFloatToHalf(output_half+i,DataT->output_data[i]);

  
  

    // allocate I/O data memory on MLU
    void *mlu_input, *mlu_output;

    // prepare input buffer
    if (CNRT_RET_SUCCESS != cnrtMalloc(&(mlu_input), inputSizeS[0])) {
      printf("cnrtMalloc Failed!\n");
      exit(-1);
    }
    if (CNRT_RET_SUCCESS != cnrtMalloc(&(mlu_output), outputSizeS[0])) {
      printf("cnrtMalloc output Failed!\n");
      exit(-1);
    }
    if (CNRT_RET_SUCCESS != cnrtMemcpy(mlu_input, input_half, 256 * 256 * 3 * sizeof(half), CNRT_MEM_TRANS_DIR_HOST2DEV)) {
      printf("cnrtMemcpy Failed!\n");
      exit(-1);
    }
    

    // setup runtime ctx
    cnrtRuntimeContext_t ctx;
    cnrtCreateRuntimeContext(&ctx, function, NULL);

    // bind device
    cnrtSetRuntimeContextDeviceId(ctx, 0);
    cnrtInitRuntimeContext(ctx, NULL);
    
    void *param[2];
    param[0] = mlu_input;
    param[1] = mlu_output;
    // compute offline
    cnrtQueue_t queue;
    cnrtRuntimeContextCreateQueue(ctx, &queue);
    cnrtInvokeRuntimeContext(ctx, (void**)param, queue, nullptr);
    cnrtSyncQueue(queue);

    
    if (CNRT_RET_SUCCESS != cnrtMemcpy(output_half, mlu_output, 256 * 256 * 3 * sizeof(half), CNRT_MEM_TRANS_DIR_DEV2HOST)) {
      printf("cnrtMemcpy output Failed!\n");
      exit(-1);
    }
    for (int i=0;i<256*256*3;i++)
        cnrtConvertHalfToFloat(output_data+i,output_half[i]);


    for(int i=0;i<t;i++)
        for(int j=0;j<3;j++)
            DataT->output_data[t*j+i] = output_data[i*3+j];

    // free memory spac
    if (CNRT_RET_SUCCESS != cnrtFree(mlu_input)) {
      printf("cnrtFree Failed!\n");
      exit(-1);
    }
    if (CNRT_RET_SUCCESS != cnrtFree(mlu_output)) {
      printf("cnrtFree output Failed!\n");
      exit(-1);
    }

    if (CNRT_RET_SUCCESS != cnrtDestroyQueue(queue)) {
      printf("cnrtDestroyQueue Failed!\n");
      exit(-1);
    }

    cnrtDestroy();
    //free(param);
    free(input_half);
    free(output_half);
}

} // namespace StyleTransfer