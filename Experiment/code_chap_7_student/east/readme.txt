7.2 实验编译与运行
    1、bangc算子填写：补齐cnplugin-SBC/spilt_sub_concat_kernel.hh和spilt_sub_concat_kernel.mlu。执行make进行编译，生成可执行文件test，进行测试。
    2、集成到cnplugin: 补齐cnplugin.h和plugin_sbc_op.cc，将整个cnplugin-SBC文件夹复制到env/Cambricon-CNPlugin-MLU270/pluginops,重新编译cnplugin,编译步骤同5.1
    3、框架算子测试：补齐tf-SBC/目录下的各个文件，按照readme.txt提示拷入对应目录，重新编译tensorflow
    4、在线推理：获取/opt/Cambricon-Test/models/east/east_int8.pb，使用tools/目录下的工具将原有pb转换成pbtxt，添加合并算子节点后，再重新转换成pb，随后修改east/EAST/run.sh中模型的位置，然后执行run_aicse.sh进行在线推理


提交文件：
stu_upload/
├── cnplugin.h
├── east_int8_sbc.pb       // 转换后的pb
├── libcnplugin.so
├── plugin_sbc_op.cc
├── spilt_sub_concat_kernel.h
├── spilt_sub_concat_kernel.mlu
└── tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl
