export MLU_OP_PRECISION=float16
export MLU_STATIC_NODE_FUSION=true

export CNRT_PRINT_INFO=OFF
export CNRT_GET_HARDWARE_TIME=OFF

python stu_upload/evaluate_mlu.py