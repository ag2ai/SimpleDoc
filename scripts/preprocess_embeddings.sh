#!/bin/bash

echo "Running Step 01: Generating ColQwen2.5 Embeddings"

python preprocess/generate_embeddings.py \
    --input_dir data/MMLongBench \
    --image_dpi 150 \
    --num_workers 8
