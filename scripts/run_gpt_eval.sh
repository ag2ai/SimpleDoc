#!/bin/bash

set -e  # Exit on error

# Loop over datasets
for DS_NAME in MMLongBench; do
    OUTPUT_DIR="outputs/simpledoc_eval/${DS_NAME}"
    GROUND_TRUTH="data/${DS_NAME}/samples.json"
    EVAL_MODEL="gpt-4.1"
    API_KEY_FILE="./openaikey"

    echo "Processing dataset: ${DS_NAME}"
    mkdir -p "$OUTPUT_DIR"

    RESULT_FILES=outputs/simpledoc_chat/"${DS_NAME}"/results.json

    for RESULT_FILE in $RESULT_FILES; do
        FILENAME=$(basename "$RESULT_FILE" .json)
        EVAL_OUTPUT="${OUTPUT_DIR}/eval_${FILENAME}.jsonl"

        echo "Evaluating $RESULT_FILE..."
        
        python evaluation/evaluate_responses.py \
            --results_file="$RESULT_FILE" \
            --ground_truth_file="$GROUND_TRUTH" \
            --output_file="$EVAL_OUTPUT" \
            --model="$EVAL_MODEL" \
            --api_key_file="$API_KEY_FILE" \
            --add_notanswerable

        echo "Done: Results saved to $EVAL_OUTPUT"
        echo "----------------------------------------"
    done
done

wait  # Wait for all background jobs to finish

# Summarize results
echo ""
echo "Summary of evaluation results:"
for DS_NAME in MMLongBench; do
    echo "Dataset: $DS_NAME"
    for EVAL_FILE in outputs/simpledoc_eval/"${DS_NAME}"/eval_*.jsonl; do
        FILENAME=$(basename "$EVAL_FILE" .jsonl)
        echo -n "  ${FILENAME}: "
        python -c "
import json
scores = []
with open('$EVAL_FILE', 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'score' in data and data['score'] != -1:
            scores.append(data['score'])
if scores:
    avg = sum(scores) / len(scores)
    print(f'Average score: {avg*100:.2f}% ({len(scores)} samples)')
else:
    print('No valid scores found')
"
    done
    echo ""
done

echo "Evaluation process completed!"
