#! /bin/bash

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MP_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

cd $main_dir
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 NCCL_NET_GDR_LEVEL=2"
OPTIONS_NCCL="NCCL_DEBUG=info"


config_json="$script_dir/ds_config_zero.json"
gpt_options=" \
       --experiment-name cogview-hwc-64-2560-40 \
       --img-tokenizer-num-tokens 8192 \
       --dataset-type TokenizedDataset \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 64 \
       --hidden-size 2560 \
       --num-attention-heads 40 \
       --save $main_dir/data/checkpoints \
       --train-iters 80000 \
       --resume-dataloader \
       --train-data ./data/merge_all_32.lmdb \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .01 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --max-position-embeddings 1089 \
       --max-memory-length 0 \
       --fp16 \
       --save-interval 5000 \
"
    #    --query-window 128 \
    #    --key-window-times 6 \
    #    --num-pivot 768 \
    #    --is-sparse \



gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"

echo $(date +%T);

run_cmd="${OPTIONS_NCCL} python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
