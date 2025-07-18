#!/bin/bash

INPUT=datas/images/image00/small-vehicles1.jpeg
#INPUT=datas/videos/multi02.avi

python tools/inference/onnx_inf_ndet.py \
    --onnx datas/models/deim_dfine_hgnetv2_m_coco_s1280x1280_b01_fp16.onnx \
    --input $INPUT

