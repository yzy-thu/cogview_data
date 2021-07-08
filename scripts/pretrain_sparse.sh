#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=2

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 NCCL_NET_GDR_LEVEL=2"
# HOST_FILE_PATH="hostfile"
OPTIONS_NCCL=""
HOST_FILE_PATH="hostfile_single"


config_json="$script_dir/ds_config.json"
gpt_options=" \
       --experiment-name cogview-testmp-48-2560-80 \
       --img-tokenizer-num-tokens 8192 \
       --dataset-type TokenizedDataset \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 2560 \
       --num-attention-heads 80 \
       --save $main_dir/data/checkpoints \
       --train-iters 80000 \
       --resume-dataloader \
       --train-data ./data/zijian_new.lmdb \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .01 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --max-position-embeddings 1089 \
       --max-memory-length 0 \
       --query-window 64 \
       --key-window-times 4 \
       --num-pivot 256 \
       --txt-loss-scale 1 \
"
    #    --checkpoint-num-layers 2 \
    #    --fp16 \



gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
