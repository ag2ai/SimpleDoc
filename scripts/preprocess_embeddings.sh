#!/bin/bash

echo "Runngin Step 01: Generating ColQwen2.5 Embeddings"
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --num_processes 2 preprocess/generate_embeddings.py \
    --input_dir data/MMLongBench \
    --image_dpi 150 \
    --num_workers 1
