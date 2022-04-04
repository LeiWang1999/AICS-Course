补全 stu_upload 中的 evaluate_cpu.py、evaluate_mlu.py 文件，执行 main_exp_4_1.py 进行测试。

执行 run_cpu.sh 单独执行 cpu 模式。在进行 pb 模型生成步骤时，需要补全 evaluate_cpu.py 的代码，然后执行 run_cpu.sh 生成 pb 模型，模型文件保存在 models 目录中。

执行 run_mlu.sh 可以单独执行 mlu 模式。

需要提交的文件为 evaluate_cpu.py、evaluate_mlu.py、vgg19_int8.pb，将以上文件压缩为 zip 包提交。