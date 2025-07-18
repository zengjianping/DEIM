#!/bin/bash

MODEL_NAME="datas/models/deim_dfine_hgnetv2_m_coco_90e"

trtexec --fp16 \
    --onnx="$MODEL_NAME.onnx" \
    --saveEngine="$MODEL_NAME.engine"


