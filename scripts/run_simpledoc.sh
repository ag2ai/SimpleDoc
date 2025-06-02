#!/bin/bash

echo "Running SimpleDoc AG2 Chat Pipeline"

python pipeline/run_simpledoc_chat.py \
    --input_file data/MMLongBench/samples.json \
    --output_file outputs/simpledoc_chat/results.json \
    --summaries_dir outputs/qwen25_7b/summaries \
    --data_base_path data/MMLongBench/documents \
    --retrieval_model Qwen/Qwen3-30B-A3B \
    --qa_model Qwen/Qwen2.5-VL-7B-Instruct \
    --api_key_file ./deepinfrakey \
    --cache_seed 42 \
    --base_url_retrieval https://api.deepinfra.com/v1/openai \
    --base_url_qa http://dgxh-2.hpc.engr.oregonstate.edu:8090/v1 \
    --retrieval_prompt_file prompts/page_retrieval_prompt.txt \
    --qa_prompt_file prompts/doc_qa_prompt_v3.5.txt \
    --max_tokens_retrieval 32768 \
    --max_tokens_qa 2048 \
    --image_dpi 150 \
    --extract_text \
    --max_iter 3 \
    --max_pages 10 \
    --max_page_retrieval 30
