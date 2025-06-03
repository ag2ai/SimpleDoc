#!/usr/bin/env python3
import os
import json
import argparse
import ast
from statistics import mean

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate retrieval metrics (retrieval rate, precision, recall, F1) from processed samples JSON."
    )
    parser.add_argument(
        "--input_file", type=str,
        default="outputs/mdocagent/retrievals/processed_samples_v2.json",
        help="Path to processed samples JSON file containing evidence_pages and relevant_pages."
    )
    parser.add_argument(
        "--output_file", type=str,
        default="outputs/mdocagent/retrievals/retrieval_metrics_v2.json",
        help="Path to output metrics JSON file."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    # Load processed samples
    with open(args.input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Define top-k thresholds and initialize metrics containers
    ks = [2, 6, 10]
    retrieval_hits = {k: [] for k in ks}
    precision_vals = {k: [] for k in ks}
    recall_vals = {k: [] for k in ks}
    f1_vals = {k: [] for k in ks}

    for sample in samples:
        # Parse ground truth pages
        gt_pages = sample.get("evidence_pages", [])
        if isinstance(gt_pages, str):
            try:
                gt_pages = ast.literal_eval(gt_pages)
            except Exception:
                gt_pages = []
        # Ensure list of ints
        if isinstance(gt_pages, list):
            gt_pages = [int(p) for p in gt_pages]
        else:
            gt_pages = []

        # Parse predicted pages
        pred_pages = sample.get("retrieved_pages_question", [])
        if isinstance(pred_pages, list):
            pred_pages = [int(p) for p in pred_pages]
        else:
            pred_pages = []

        # Compute retrieval metrics for each top-k cutoff
        for k in ks:
            pred_k = pred_pages[:k]
            # Hit: all ground truth pages retrieved within top-k
            hit_k = 1 if set(gt_pages).issubset(set(pred_k)) else 0
            retrieval_hits[k].append(hit_k)
            if gt_pages:
                tp_k = len(set(pred_k) & set(gt_pages))
                precision_k = tp_k / len(set(pred_k)) if pred_k else 0.0
                recall_k = tp_k / len(set(gt_pages))
                f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
            else:
                precision_k = 1.0 if not pred_k else 0.0
                recall_k = 1.0
                f1_k = 1.0 if precision_k == 1.0 else 0.0
            precision_vals[k].append(precision_k)
            recall_vals[k].append(recall_k)
            f1_vals[k].append(f1_k)

    # Aggregate metrics for each top-k cutoff
    metrics = {"num_samples": len(samples), "top_k": {}}
    for k in ks:
        metrics["top_k"][k] = {
            "retrieval_rate": mean(retrieval_hits[k]) * 100 if retrieval_hits[k] else 0.0,
            "precision": mean(precision_vals[k]) * 100 if precision_vals[k] else 0.0,
            "recall": mean(recall_vals[k]) * 100 if recall_vals[k] else 0.0,
            "f1": mean(f1_vals[k]) * 100 if f1_vals[k] else 0.0
        }

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_file) or '.'
    os.makedirs(out_dir, exist_ok=True)
    # Save metrics
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Print results
    print(f"Metrics saved to {args.output_file}")
    print(f"Number of samples: {metrics['num_samples']}")
    for k in ks:
        m = metrics["top_k"][k]
        print(f"Top-{k} Retrieval rate: {m['retrieval_rate']:.2f}%  Precision: {m['precision']:.2f}%  Recall: {m['recall']:.2f}%  F1: {m['f1']:.2f}%")

if __name__ == "__main__":
    main()