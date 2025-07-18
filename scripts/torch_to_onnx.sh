#!/bin/bash

python tools/deployment/export_onnx_ndet.py \
    --check --simplify --max_dets 100 --thres_conf 0.3 \
    -c configs/deim_dfine/deim_hgnetv2_m_coco.yml \
    -r datas/models/deim_dfine_hgnetv2_m_coco_90e.pth


