#!/bin/bash

python tools/deployment/export_onnx.py --check \
    -c configs/deim_dfine/deim_hgnetv2_n_coco.yml \
    -r datas/pretrained_models/deim_dfine_hgnetv2_n_coco_160e.pth


