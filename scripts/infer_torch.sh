#!/bin/bash

INPUT=datas/images/image00/small-vehicles1.jpeg
#INPUT=datas/videos/multi02.avi

python tools/inference/torch_inf.py \
    -c configs/deim_dfine/deim_hgnetv2_n_coco.yml \
    -r datas/models/deim_dfine_hgnetv2_n_coco_160e.pth \
    --input $INPUT \
    --device cuda:0

