#!/bin/bash

export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-3}
export MASTER_ADDR=${MASTER_ADDR:-worker-1}
export MASTER_PORT=${MASTER_PORT:-12345}
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-0}

deepspeed --hostfile=hostfile.txt \
    --no_ssh \
    --num_gpus=1 \
    --num_nodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    inference.py