# export CNRT_PRINT_INFO=ON
# export CNRT_GET_HARDWARE_TIME=ON

rm out/mlu/*

python ./stu_upload/evaluate_mlu.py --model pb_models/udnie_int8.pb --in-path data/train2014_small/ --out-path out/mlu/
