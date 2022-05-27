7.3 实验编译与运行
    1、bangc算子填写：补齐bangc/PluginBatchMatMulV2Op/plugin_batch_matmul_v2_kernel.mlu和plugin_batch_matmul_v2_kernel.h
    2、集成到cnplugin: 补齐cnplugin.h和plugin_batch_matmul_v2_op.cc，将整个PluginBatchMatMulV2Op文件夹复制到env/Cambricon-CNPlugin-MLU270/pluginops,重新编译cnplugin
    3、bangc算子测试：执行compile.sh，生成matmultest，进行测试
    4、集成到tensorflow:补齐tf-implementation/tf-add-batchmatmulv2/目录下的各个文件，按照readme.txt提示拷入对应目录，重新编译tensorflow
    5、在线推理：执行run_aicse.sh
    6、单batch推理：在squad_output_dir_128_small目录下，执行python inference_pb_demo.py  frozen_model_int16.pb 

提交文件：
stu_upload/
├── cnplugin.h
├── libcnplugin.so
├── plugin_batch_matmul_v2_kernel.h
├── plugin_batch_matmul_v2_kernel.mlu
├── plugin_batch_matmul_v2_op.cc
└── tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl
