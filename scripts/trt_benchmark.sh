#!/bin/bash

python tools/benchmark/trt_benchmark.py \
    --infer_dir /data/Datasets/PoseTrainData/MSCOCO/images/val2017/ \
    --engine_dir datas/models/
