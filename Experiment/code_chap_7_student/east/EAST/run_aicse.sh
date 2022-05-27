./run_cpu.sh
export MLU_VISIBLE_DEVICES=0
./run.sh 1 MLU270 int8 origin 500 1
./run.sh 1 MLU270 int8 sbc 500 1
#export MLU_VISIBLE_DEVICES=
#./run.sh 1 MLU270 float32 origin 500 1
