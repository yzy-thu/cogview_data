#!/bin/bash

# CHECKPOINT_PATH=data/checkpoints/cogview-ali-32-1024-3202-02-09-28/
# CHECKPOINT_PATH=data/checkpoints/cogview-zijian-32-1024-3202-13-14-47/
CHECKPOINT_PATH=/root/mnt/dingming/cogview-normmax
NLAYERS=48
NHIDDEN=2560
NATT=40
MAXSEQLEN=1089
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MPSIZE=1

#SAMPLING ARGS
TEMP=1.03
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=200
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

MASTER_PORT=${MASTER_PORT} python generate_samples.py \
       --deepspeed \
       --model-parallel-size $MPSIZE \
       --deepspeed_config ${config_json} \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 1089 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --img-tokenizer-path pretrained/vqvae/vqvae_hard_biggerset_011.pt \
       --query-window 64 \
       --key-window-times 4 \
       --num-pivot 256 \
       --is-sparse 0

