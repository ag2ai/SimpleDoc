#!/usr/bin/env python3
"""
Calculate average number of pages in the last iteration of each question
across all v4_note_top*_results.json files under outputs/simpleDoc/prediction/*.
"""

import glob
import json
import os
import sys
import pandas as pd

def main():
    pattern = os.path.join("outputs", "docr1", "prediction", "*", "v4_note_top*_results.json")
    files = glob.glob(pattern)
    if not files:
        print(f"No files found with pattern {pattern}", file=sys.stderr)
        sys.exit(1)

    # Reorganize stats to track both by dataset and by individual file
    dataset_stats = {}  # For dataset-level statistics
    file_stats = {}     # For individual file statistics

    for filepath in files:
        log_name = os.path.basename(filepath)
        dataset = os.path.basename(os.path.dirname(filepath))
        file_key = f"{dataset}/{log_name}"
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)
            continue

        # Initialize stats for this file
        if file_key not in file_stats:
            file_stats[file_key] = {"total_pages": 0, "total_pages_cum": 0, "total_pages_first": 0, "question_count": 0}
        
        # Initialize stats for this dataset if not already present
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {"total_pages": 0, "total_pages_cum": 0, "total_pages_first": 0, "question_count": 0}

        for item in data:
            iterations = item.get("iterations", [])
            if not iterations:
                continue
                
            cum_pages = sum(len(it.get("pages", [])) for it in iterations)
            
            # Get first iteration pages
            first_iter_objs = [it for it in iterations if it.get("iteration", 0) == 1]
            first_iter = first_iter_objs[0] if first_iter_objs else None
            pages_first = first_iter.get("pages", []) if first_iter else []
            
            # Get last iteration pages
            max_iter = max(it.get("iteration", 0) for it in iterations)
            last_iter_objs = [it for it in iterations if it.get("iteration", 0) == max_iter]
            
            if not last_iter_objs:
                continue
                
            last_iter = last_iter_objs[0]
            pages_last = last_iter.get("pages", [])
            
            # Update file-level stats
            file_stats[file_key]["total_pages"] += len(pages_last)
            file_stats[file_key]["total_pages_cum"] += cum_pages
            file_stats[file_key]["total_pages_first"] += len(pages_first)
            file_stats[file_key]["question_count"] += 1
            
            # Update dataset-level stats
            dataset_stats[dataset]["total_pages"] += len(pages_last)
            dataset_stats[dataset]["total_pages_cum"] += cum_pages
            dataset_stats[dataset]["total_pages_first"] += len(pages_first)
            dataset_stats[dataset]["question_count"] += 1

    if not file_stats:
        print("No question iterations found.", file=sys.stderr)
        sys.exit(1)

    # Build DataFrame for per-file statistics
    file_rows = []
    for file_key, vals in file_stats.items():
        q_count = vals["question_count"]
        if q_count == 0:
            continue
            
        avg_pages = vals["total_pages"] / q_count
        avg_cum_pages = vals["total_pages_cum"] / q_count
        avg_pages_first = vals["total_pages_first"] / q_count
        dataset, filename = file_key.split('/', 1)
        
        file_rows.append({
            "dataset": dataset,
            "file": filename,
            "question_count": q_count,
            "average_pages_first": round(avg_pages_first, 2),
            "average_pages_last": round(avg_pages, 2),
            "average_cum_pages": round(avg_cum_pages, 2),
        })
    
    # Build DataFrame for per-dataset statistics
    dataset_rows = []
    for dataset, vals in dataset_stats.items():
        q_count = vals["question_count"]
        if q_count == 0:
            continue
            
        avg_pages = vals["total_pages"] / q_count
        avg_cum_pages = vals["total_pages_cum"] / q_count
        avg_pages_first = vals["total_pages_first"] / q_count
        
        dataset_rows.append({
            "dataset": dataset,
            "question_count": q_count,
            "average_pages_first": round(avg_pages_first, 2),
            "average_pages_last": round(avg_pages, 2),
            "average_cum_pages": round(avg_cum_pages, 2),
        })

    # Print dataset-level summary first
    print("\n=== Dataset Summary ===")
    df_datasets = pd.DataFrame(dataset_rows).sort_values("dataset")
    print(df_datasets)
    
    # Print file-level details
    print("\n=== Individual Files ===")
    df_files = pd.DataFrame(file_rows).sort_values(["dataset", "file"])
    print(df_files)

if __name__ == "__main__":
    main() 