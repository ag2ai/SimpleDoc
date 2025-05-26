#!/bin/bash

echo "Running Step 01: PDF Page Summarization"

python preprocess/generate_summaries.py \
    --input_file data/MMLongBench/samples.json \
    --data_base_path data/MMLongBench/documents \
    --output_dir outputs/qwen25_32b/summaries \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --api_key_file ./deepinfrakey \
    --base_url http://dgxh-3.hpc.engr.oregonstate.edu:8090/v1 \
    --prompt_file prompts/general_summary_prompt.txt \
    --image_dpi 150 \
    --max_tokens 2048 \
    --n_jobs 16 \
    --verbose \
    --skip_existing
