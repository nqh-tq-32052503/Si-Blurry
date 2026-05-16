#!/bin/bash

NOTE="MVP_CIFAR_0" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

MODE="mvp"
DATASET="cub200" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
MEM_SIZE=0 
ONLINE_ITER=3
MODEL_NAME="mvp" 
EVAL_PERIOD=1000
BATCHSIZE=64
LR=5e-3 
OPT_NAME="adam" 
SCHED_NAME="default"

python main.py --mode $MODE \
    --dataset $DATASET \
    --data_dir ./data/CUB200 \
    --n_tasks $N_TASKS --m $M --n $N \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 --rnd_NM \
    --use_mask --use_contrastiv --use_afs --use_gsf
