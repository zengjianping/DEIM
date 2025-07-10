#!/bin/bash

INPUT=datas/images/image00/small-vehicles1.jpeg
#INPUT=datas/videos/multi02.avi

python tools/inference/trt_inf.py \
    --trt datas/models/deim_dfine_hgnetv2_n_coco_160e.engine \
    --i $INPUT

