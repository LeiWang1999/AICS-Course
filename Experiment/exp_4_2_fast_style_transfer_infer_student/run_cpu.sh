rm out/cpu/*

python ./stu_upload/evaluate_cpu.py --model pb_models/udnie.pb --in-path data/train2014_small/  --out-path out/cpu/
