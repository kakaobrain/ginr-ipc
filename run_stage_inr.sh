#!/bin/bash

NPROC_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=master
OPTS=""
EPOCH_TEST=-1
SEED=0

for i in "$@"; do
case $i in
    -nn=*|--n-nodes=*)
    NNODES="${i#*=}"
    shift
    ;;
    -np=*|--nproc-per-node=*)
    NPROC_PER_NODE="${i#*=}"
    shift
    ;;
    -nr=*|--node-rank=*)
    NODE_RANK="${i#*=}"
    shift
    ;;
    -m=*)
    MODEL_CONFIG="${i#*=}"
    shift
    ;;
    -r=*)
    OUTPUT="${i#*=}"
    shift
    ;;
    -s=*)
    SEED="${i#*=}"
    shift
    ;;
    -e=*)
    EPOCH_TEST="${i#*=}"
    shift
    ;;
    -p=*)
    POSTFIX="${i#*=}"
    shift
    ;;
    --master=*)
    MASTER_ADDR="${i#*=}"
    shift
    ;;
    --eval)
    OPTS+=" --eval"
    shift
    ;;
    --resume)
    OPTS+=" --resume"
    shift
    ;;
    --resume-lr-reduction)
    OPTS+=" --resume-lr-reduction"
    shift
    ;;
    --fp16_compress)
    OPTS+=" --fp16_compress"
    shift
    ;;
    *)
    OPTS+=" ${i}"
    shift
    ;;
esac
done

if [ "$NNODES" -gt 1 ]; then
    MASTER_ADDR=$MASTER_ADDR
    MASTER_ADDR+=.$TASK_GROUP_NAME
    sleep 60
else
    MASTER_ADDR=localhost
fi

echo $MASTER_ADDR

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
                                   --nnodes=$NNODES \
                                   --master_addr=$MASTER_ADDR \
                                   --master_port=2901 \
                                   --node_rank=$NODE_RANK \
                                   ./src/main_stage_inr.py \
                                   -m=$MODEL_CONFIG \
                                   -r=$OUTPUT \
                                   -e=$EPOCH_TEST \
                                   -p=$POSTFIX \
                                   --nproc_per_node=$NPROC_PER_NODE \
                                   --nnodes=$NNODES \
                                   --node_rank=$NODE_RANK \
                                   --seed=$SEED \
                                   $OPTS
