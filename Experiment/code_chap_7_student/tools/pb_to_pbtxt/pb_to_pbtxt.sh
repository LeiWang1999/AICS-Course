#!/bin/bash
set -e

usage () {
    echo "Usage:"
    echo "      input_pb               [necessary]  input pb to be converted"
    echo "      output_pbtxt           [necessary]  output pbtxt generated"
}

input_pb=$1
output_pbtxt=$2

if [ $# -lt 2 ]; then
    usage
    exit 1
fi

python pb_to_pbtxt.py ${input_pb} ${output_pbtxt}
sed -i '/tensor_content/d' ${output_pbtxt}
