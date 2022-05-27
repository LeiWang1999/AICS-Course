#!/bin/bash
set -e
#export TF_CPP_MIN_VLOG_LEVEL=1
#export MLU_VISIBLE_DEVICES=""
export  MLU_IS_CONSTANT=true
usage () {
    echo "Usage:"
    echo "${0}    "
    echo "              core_num: 1/4/16"
    echo "              core_version: MLU100/MLU270"
    echo "              precision: float32/int8"
    echo "              mode: origin/sbc"
    echo "              image_num: 1/2/...."
    echo "              batch_size: 1/4/16"
    echo "${0} 1 MLU270 int8 1 1"
}

if [ $# -lt 5 ]; then
    usage
    exit 1
fi


core_num=$1
core_version=$2
precision=$3
mode=$4
number=$5
batch_size=$6

rm -rf results/*

if [[ "${precision}" =~ "int8" ]]; then
    MODEL_PATH="${AICSE_MODELS_MODEL_HOME}/east/east_int8.pb"
else
    MODEL_PATH="./cpu_pb/east.pb"
fi

# may change the path of the model
if [[ "${mode}" =~ "sbc" ]]; then
   MODEL_PATH="../models/east_int8_sbc.pb"
fi
echo $MODEL_PATH
DATASET_HOME=${AICSE_MODELS_DATA_HOME}
DATASET_PATH="${DATASET_HOME}/east/ICDAR_2015"
GT_PATH="${DATASET_HOME}/east/ICDAR_2015/gt.zip"

echo
echo "=== Host Demo: MLU run EAST ==="
echo
python -u eval.py \
    --test_data_path=${DATASET_PATH} \
    --gpu_list=0 \
    --checkpoint_path=${MODEL_PATH} \
    --output_dir=./results \
    --core_num=${core_num}  \
    --core_version=${core_version} \
    --precision=${precision} \
    --number=${number} \
    --batch_size=${batch_size} 2>&1 | tee log_$mode
cd results
zip -r submit.zip *.txt
cd ../
python script.py -g=${GT_PATH} -s=results/submit.zip 2>&1 | tee log_results_$mode
