EXPREIMENT=$1
GPUS=$2
CKPT=$3

CUDA_VISIBLE_DEVICES=$GPUS python transformer.py \
    ../results/$EXPREIMENT/model.json \
    ../results/$EXPREIMENT/path.json \
    --eval_set 'val' \
    --is_train \
    --resume_file $CKPT 
