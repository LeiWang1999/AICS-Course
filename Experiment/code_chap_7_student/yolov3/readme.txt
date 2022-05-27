7.1 实验编译与运行
   1、bangc算子填写：补齐bangc/PluginYolov3DetectionOutputOp/nms_detection.h
   2、集成到cnplugin:cnplugin.h和.cc文件已经写好，只需要将整个PluginYolov3DetectionOutputOp文件夹复制到env/Cambricon-CNPlugin-MLU270/pluginops,重新编译cnplugin,编译步骤同5.1操作
   3、集成到tensorflow:补齐tf-implementation/tf-1.14-detectionoutput/目录下的各个文件，按照readme.txt提示拷入对应目录，重新编译tensorflow
   4、在线推理：获取/opt/Cambricon-Test/models/yolov3/yolov3_int8_bang_shape_new.pb,使用tools/目录下的工具将原有pb转换成pbtxt，添加后处理大算子节点，再重新转换成pb，随后在yolov3-bcl/demo/run_evaluate.sh中修改新的模型位置，然后执行run_aicse.sh进行在线推理


提交文件：
stu_upload/
├── libcnplugin.so
├── nms_detection.h
├── tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl
└── yolov3_int8_bang_shape_new.pb    // 转换后的pb模型
