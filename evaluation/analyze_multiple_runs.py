import os
import json
import argparse
import numpy as np
import glob
from collections import defaultdict

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze evaluation results from multiple runs")
    
    # Input/output paths
    parser.add_argument("--eval_dir", type=str, required=True,
                       help="Directory containing evaluation result files")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save analysis results (default: {eval_dir}/analysis_results.json)")
    parser.add_argument("--file_pattern", type=str, default="eval_*_try*.jsonl",
                       help="Pattern to match evaluation files (default: eval_*_try*.jsonl)")
    
    return parser.parse_args()

def load_eval_results(eval_dir, file_pattern):
    """Load evaluation results from all matching files in the directory."""
    pattern = os.path.join(eval_dir, file_pattern)
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No evaluation files found matching pattern: {pattern}")
        return {}
    
    # Organize results by prompt type, try number, and sample ID
    results_by_prompt = defaultdict(lambda: defaultdict(list))
    results_by_try = defaultdict(lambda: defaultdict(dict))
    
    # Track try numbers for each prompt type
    try_numbers = defaultdict(set)
    
    for file_path in result_files:
        # Extract prompt type and try number from filename
        filename = os.path.basename(file_path)
        parts = filename.replace('.jsonl', '').split('_try')
        
        if len(parts) != 2:
            print(f"Warning: Could not parse prompt type and try number from {filename}, skipping")
            continue
            
        prompt_type = '_'.join(parts[0].split('_')[1:])  # Remove 'eval_' prefix
        try_num = int(parts[1])
        try_numbers[prompt_type].add(try_num)
        
        # Load results from this file
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Create a unique sample ID from doc_id and question
                        sample_id = f"{data.get('doc_id', '')}_{data.get('question', '')}"
                        
                        # Store success (1) or failure (0) for this sample in this try
                        score = data.get('score', -1)
                        if score != -1:  # Only include valid scores
                            success = 1 if score > 0 else 0
                            results_by_prompt[prompt_type][sample_id].append(success)
                            results_by_try[prompt_type][try_num][sample_id] = success
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse JSON in {file_path}, skipping line")
    
    return results_by_prompt, results_by_try, try_numbers

def calculate_metrics(results_by_prompt, results_by_try, try_numbers):
    """Calculate success rate metrics for each prompt type."""
    metrics = {}
    
    for prompt_type, samples in results_by_prompt.items():
        # Get success rates per try for this prompt type
        success_rates_per_try = []
        for try_num in sorted(try_numbers[prompt_type]):
            try_results = results_by_try[prompt_type][try_num]
            if try_results:  # Skip empty tries
                success_rate = np.mean(list(try_results.values()))
                success_rates_per_try.append(success_rate)
        
        # Calculate mean and standard error of success rates across tries
        if success_rates_per_try:
            mean_success = np.mean(success_rates_per_try)
            # Standard error of the mean (SEM)
            if len(success_rates_per_try) > 1:
                std_error = np.std(success_rates_per_try, ddof=1) / np.sqrt(len(success_rates_per_try))
            else:
                std_error = 0
        else:
            mean_success = 0
            std_error = 0
        
        # Collect all individual scores for overall statistics
        all_scores = []
        for sample_id, tries in samples.items():
            all_scores.extend(tries)
        
        # Calculate Pass@k metrics
        pass_at_metrics = {}
        k_values = [1, 3, 5, 10]  # Add or remove k values as needed
        
        # For each sample, calculate if it passed at different k values
        sample_pass_at_k = defaultdict(list)
        
        for sample_id, tries in samples.items():
            num_tries = len(tries)
            
            for k in k_values:
                if k <= num_tries:
                    # If any of the first k tries succeeded, count as passed@k
                    passed = 1 if sum(tries[:k]) > 0 else 0
                    sample_pass_at_k[k].append(passed)
        
        # Calculate average Pass@k across all samples
        for k in k_values:
            if sample_pass_at_k[k]:
                pass_rate = np.mean(sample_pass_at_k[k])
                pass_at_metrics[f"Pass@{k}"] = pass_rate
        
        metrics[prompt_type] = {
            "mean_success_rate": mean_success,
            "std_error": std_error,
            "pass_at_metrics": pass_at_metrics,
            "num_samples": len(samples),
            "num_total_evaluations": len(all_scores),
            "num_tries": len(success_rates_per_try),
            "success_rates_per_try": success_rates_per_try
        }
    
    return metrics

def main():
    """Main function to run the analysis."""
    args = parse_arguments()
    
    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = os.path.join(args.eval_dir, "analysis_results.json")
    
    print(f"Loading evaluation results from {args.eval_dir}...")
    results_by_prompt, results_by_try, try_numbers = load_eval_results(args.eval_dir, args.file_pattern)
    
    if not results_by_prompt:
        print("No valid evaluation results found. Exiting.")
        return
    
    print("Calculating metrics...")
    metrics = calculate_metrics(results_by_prompt, results_by_try, try_numbers)
    
    # Save results to file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary to console
    print("\nAnalysis Results:")
    print("================")
    for prompt_type, prompt_metrics in metrics.items():
        print(f"\nPrompt Type: {prompt_type}")
        print(f"Number of samples: {prompt_metrics['num_samples']}")
        print(f"Number of tries: {prompt_metrics['num_tries']}")
        print(f"Total evaluations: {prompt_metrics['num_total_evaluations']}")
        print(f"Mean success rate: {prompt_metrics['mean_success_rate'] * 100:.2f}% ± {prompt_metrics['std_error'] * 100:.2f}% (std error)")
        
        print("Success rates by try:")
        for i, rate in enumerate(prompt_metrics['success_rates_per_try']):
            print(f"  Try {i+1}: {rate * 100:.2f}%")
        
        print("Pass@k metrics:")
        for k, rate in prompt_metrics['pass_at_metrics'].items():
            print(f"  {k}: {rate * 100:.2f}%")

if __name__ == "__main__":
    main() 