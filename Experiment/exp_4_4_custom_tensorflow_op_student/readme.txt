udnie.pb 已经提供在 models/pb_models 目录下。

算子添加实验请在 env/tensorflow-v1.10/ 中完成。
在 tf-implementation 目录下有 cwise_op_power_difference.cc 的例子, 补全后将 cwise_op_power_difference.* 复制到 tensorflow/core/kernels/ 目录下.
完成算子添加实验后将编译好的whl安装包放入stu_upload中。建议进行实验前先将原始的whl文件备份。
whl文件：env/tensorflow-v1.10/virtualenv_mlu/tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl

补全 stu_upload 目录下的 power_diff_numpy.py、power_difference_test_cpu.py、transform_cpu.py，
并将 udnie_power_diff.pb 和 udnie_power_diff_numpy.pb 复制到 stu_upload 目录下，执行 run_exp_4_4.py 进行测试。

执行 run_style_transfer.sh 可以单独进行风格迁移测试。

需要提交的文件为 power_diff_numpy.py、power_difference_test_cpu.py、transform_cpu.py、tensorflow_mlu-1.14.0-cp27-cp27mu-linux_x86_64.whl、
udnie_power_diff_numpy.pb、udnie_power_diff.pb，将以上文件压缩为 zip 包提交。

注意：
1. 要把env/tensorflow-v1.10/build_tensorflow−v1.10_mlu.sh中的jobs_num改为16，否则会因为任务数过多而报莫名其妙的错误。
2. 千万千万要执行source build_tensorflow−v1.10_mlu.sh后再在env/tensorflow-v1.10/中执行source env.sh，否则source build_tensorflow−v1.10_mlu.sh会失败。