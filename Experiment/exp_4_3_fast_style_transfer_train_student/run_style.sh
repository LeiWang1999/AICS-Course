export MLU_VISIBLE_DEVICES=''

python style.py --checkpoint-dir ckp_temp \
                --style examples/style/rain_princess.jpg \
                --train-path data/train2014_small \
                --content-weight 1.5e1 \
                --checkpoint-iterations 100 \
                --epochs 2 \
                --batch-size 4 \
                --type 0
