#!/bin/bash

model=s

#CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=4 \
python train.py --test-only \
    -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml \
    -r datas/pretrained_models/deim_dfine_hgnetv2_${model}_coco_120e.pth

