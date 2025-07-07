#!/bin/bash

MODEL_NAME="datas/pretrained_models/deim_dfine_hgnetv2_n_coco_160e"

trtexec --fp16 \
    --onnx="$MODEL_NAME.onnx" \
    --saveEngine="$MODEL_NAME.engine"


