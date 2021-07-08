#!/bin/bash

# CHECKPOINT_PATH=data/checkpoints/cogview-ali-32-1024-3202-02-09-28/
# CHECKPOINT_PATH=data/checkpoints/cogview-zijian-32-1024-3202-13-14-47/
CHECKPOINT_PATH=/root/cogview2/data/checkpoints/coco_1103-29-09-36
NLAYERS=24
NHIDDEN=1024
NATT=16
MAXSEQLEN=1089
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MPSIZE=1

#SAMPLING ARGS
TEMP=0.8
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=200
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

MASTER_PORT=${MASTER_PORT} kernprof -l -v preprocess_imgtokens.py \
       --deepspeed_config ${config_json} \
       --world-rank 5 \
       --experiment-name cogview-ali-16-1024-16 \
       --img-tokenizer-num-tokens 8192 \
       --dataset-type TokenizedDataset \
       --model-parallel-size 1 \
       --num-layers 16 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --save $main_dir/data/checkpoints \
       --train-iters 50000 \
       --resume-dataloader \
       --train-data ./data/small_ali.lmdb \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .01 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --max-position-embeddings 1089 \
       --max-memory-length 0 \
       --fp16 \

