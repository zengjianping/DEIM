#!/bin/bash

INPUT=datas/images/image00/small-vehicles1.jpeg
#INPUT=datas/videos/multi02.avi

python tools/inference/torch_inf_ndet.py \
    -c configs/deim_dfine/deim_hgnetv2_m_coco.yml \
    -r datas/models/deim_dfine_hgnetv2_m_coco_90e.pth \
    --input $INPUT \
    --device cuda:0

