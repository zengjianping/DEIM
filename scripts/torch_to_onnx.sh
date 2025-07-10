#!/bin/bash

python tools/deployment/export_onnx_ndet.py --check \
    -c configs/deim_dfine/deim_hgnetv2_l_coco.yml \
    -r datas/models/deim_dfine_hgnetv2_l_coco_50e.pth


