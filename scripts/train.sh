#! /bin/bash

cd `dirname "$0"`/..

mpirun -np 2 \
    -H localhost:2\
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 train.py
