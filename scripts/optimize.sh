#!/bin/bash

SRC_DIR=`dirname "$0"`/..
MODEL_DIR=$SRC_DIR/models/latest
TF=/usr/local/lib/python3.5/dist-packages/tensorflow

python3 -m tensorflow.python.tools.optimize_for_inference \
    --input=$MODEL_DIR/frozen_model.pb \
    --output=$MODEL_DIR/optimized_model.pb \
    --frozen_graph=True \
    --input_names=images \
    --output_names=heatmaps
