EXPREIMENT=$1
GPUS=$2
CKPT=$3

CUDA_VISIBLE_DEVICES=$GPUS python transformer.py \
    $EXPREIMENT/model.json \
    $EXPREIMENT/path.json \
    --eval_set 'tst' \
    --resume_file $CKPT