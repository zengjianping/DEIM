#!/bin/bash

INPUT=datas/test_images/small-vehicles1.jpeg
#INPUT=datas/test_videos/multi02.avi

python tools/inference/trt_inf.py \
    --trt datas/pretrained_models/deim_dfine_hgnetv2_n_coco_160e.engine \
    --i $INPUT

