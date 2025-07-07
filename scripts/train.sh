#!/bin/bash

model=x

#CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=4 \
python train.py --use-amp --seed=0 \
    -c configs/deim_dfine/dfine_hgnetv2_${model}_coco.yml

