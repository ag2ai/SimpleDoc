#!/usr/bin/env python3
"""
Evaluate pipeline results: compute retrieval rate and GPT-4 evaluation accuracy.
"""
import os
import json
import argparse
import ast
from utils.openai_helper import initialize_client
from eval.evaluate_responses import evaluate_response
from tqdm import tqdm
import joblib
from functools import partial
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and QA performance of the pipeline")
    parser.add_argument("--samples_file", type=str,
                        default="data/MMLongBench/samples.json",
                        help="Original samples with ground truth answers and pages")
    parser.add_argument("--results_file", type=str,
                        default="outputs/docr1/v1_results.json",
                        help="Pipeline results JSON file")
    parser.add_argument("--output_file", type=str,
                        default="outputs/docr1/v1_evaluation_metrics.json",
                        help="Output JSON file for evaluation metrics")
    parser.add_argument("--eval_model", type=str,
                        default="gpt-4.1",
                        help="Model name for GPT-4 evaluation")
    parser.add_argument("--api_key_file", type=str, default="./openaikey",
                        help="File containing the API key")
    parser.add_argument("--base_url", type=str,
                        default="https://api.openai.com/v1",
                        help="Base URL for the LLM API")
    parser.add_argument("--cache_seed", type=int, default=123,
                        help="Cache seed for LLM client")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens for evaluation LLM call")
    parser.add_argument("--n_jobs", type=int, default=16,
                        help="Number of parallel jobs to run (-1 for all cores)")
    return parser.parse_args()

def is_ground_truth_not_answerable(ground_truth):
    """Check if the ground truth answer indicates question is not answerable."""
    not_answerable_phrases = [
        "not answerable",
        "cannot be answered",
        "does not contain",
        "document doesn't provide",
        "no information",
        "information not provided",
        "not mentioned",
        "not discussed",
        "not in the document",
        "not available in",
        "not found in"
    ]
    
    if not ground_truth:
        return True
        
    ground_truth_lower = str(ground_truth).lower()
    return any(phrase in ground_truth_lower for phrase in not_answerable_phrases)

def process_sample(sample, result_map, client, eval_model, max_tokens, has_evidence_pages):
    """Process a single sample and return retrieval and accuracy metrics for each iteration."""
    key = (sample['doc_id'], sample['question'])
    res = result_map.get(key)
    
    if not res or not res.get('iterations'):
        return None
    
    # Ground truth pages
    if has_evidence_pages:
        try:
            gt_pages = sample.get('evidence_pages', [])
            if not isinstance(gt_pages, list):
                gt_pages = ast.literal_eval(gt_pages)
        except Exception:
            gt_pages = []
    
    # Create results for each iteration
    ground_truth = sample.get('answer', '')
    question = sample.get('question', '')
    
    # Check if ground truth indicates not answerable
    gt_not_answerable = is_ground_truth_not_answerable(ground_truth)
    
    iterations_results = []
    
    for i, iteration in enumerate(res.get('iterations', [])):
        iter_num = i + 1
        pred_pages = iteration.get('pages', [])
        
        # Calculate retrieval metrics only if evidence_pages field exists
        iteration_metrics = {}
        if has_evidence_pages:
            # Check retrieval hit
            retrieval_hit = 1 if set(pred_pages) >= set(gt_pages) else 0
            
            # Calculate precision, recall, and F1 for retrieval
            if gt_pages:
                true_positives = len(set(pred_pages) & set(gt_pages))
                precision = true_positives / len(set(pred_pages)) if pred_pages else 0.0
                recall = true_positives / len(set(gt_pages))
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                # If there are no ground truth pages, set metrics accordingly
                precision = 1.0 if not pred_pages else 0.0
                recall = 1.0  # All relevant pages (0) were retrieved
                f1 = 1.0 if precision == 1.0 else 0.0
            
            # Add retrieval metrics to iteration results
            iteration_metrics.update({
                'retrieval_hit': retrieval_hit,
                'retrieval_precision': precision,
                'retrieval_recall': recall,
                'retrieval_f1': f1,
            })
        
        # Get response_type and determine if not answerable
        response_type = iteration.get('response_type', '')
        is_not_answerable = (response_type != "answer")
        
        # Calculate not_answerable accuracy - correct when both prediction and ground truth agree
        not_answerable_correct = (is_not_answerable == gt_not_answerable)
        
        # Set answer to "not answerable" if response_type is not "answer"
        pred_answer = iteration.get('answer', '')
        if is_not_answerable:
            pred_answer = "The document does not contain the information needed to answer this question."
        
        # Evaluate answer correctness
        eval_res = evaluate_response(
            client,
            eval_model,
            pred_answer,
            str(ground_truth),
            question,
            max_tokens,
            temperature=0.0
        )
        
        # Add other metrics to iteration results
        iteration_metrics.update({
            'iteration': iter_num,
            'accuracy': 1 if eval_res.get('score', 0) == 1 else 0,
            'response_type': response_type,
            'is_not_answerable': is_not_answerable,
            'gt_not_answerable': gt_not_answerable,
            'not_answerable_correct': not_answerable_correct
        })
        
        # Store iteration results
        iterations_results.append(iteration_metrics)
        
    return {
        'doc_id': sample['doc_id'],
        'question': sample['question'],
        'iterations': iterations_results
    }

def main():
    args = parse_arguments()
    # Initialize GPT-4 evaluation client
    args.model = args.eval_model
    client = initialize_client(args)

    # Load samples and results
    with open(args.samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    with open(args.results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Determine if samples have evidence_pages (if first sample has it, all have it)
    has_evidence_pages = bool(samples) and 'evidence_pages' in samples[0]
    # Determine if samples have evidence_sources (if first sample has it, all have it)
    has_evidence_sources = bool(samples) and 'evidence_sources' in samples[0]

    # Map results by doc_id and original_question
    result_map = {
        (r['doc_id'], r['original_question']): r
        for r in results
    }

    total = len(samples)
    
    # Define the processing function with has_evidence_pages parameter
    process_func = partial(
        process_sample,
        result_map=result_map,
        client=client,
        eval_model=args.eval_model,
        max_tokens=args.max_tokens,
        has_evidence_pages=has_evidence_pages
    )
    
    # Process samples in parallel
    print(f"Processing {total} samples with {args.n_jobs} parallel jobs")
    all_results = joblib.Parallel(n_jobs=args.n_jobs, backend="threading")(
        joblib.delayed(process_func)(sample) for sample in tqdm(samples, desc="Evaluating samples")
    )
    
    # Filter out None results
    all_results = [result for result in all_results if result is not None]
    
    # Aggregate metrics by iteration
    max_iterations = max([
        len(result['iterations']) 
        for result in all_results
    ]) if all_results else 0
    
    metrics = {
        'total_samples': total,
        'processed_samples': len(all_results),
        'has_evidence_pages': has_evidence_pages,
        'has_evidence_sources': has_evidence_sources,
        'by_iteration': {},
        'final': {},
    }
    
    # Calculate metrics for each iteration
    for iteration in range(1, max_iterations + 1):
        retrieval_hits = []
        accuracy_hits = []
        not_answerable_correct_hits = []
        response_types = {}
        precision_vals = []
        recall_vals = []
        f1_vals = []
        
        for result in all_results:
            # Find the last available iteration up to the current one
            available_iters = [i for i in result['iterations'] if i['iteration'] <= iteration]
            if available_iters:
                # Use the latest iteration available (maximum iteration number less than or equal to current)
                iter_result = max(available_iters, key=lambda x: x['iteration'])
                
                # Only calculate retrieval metrics for answerable questions if we have evidence_pages
                if has_evidence_pages and not iter_result['gt_not_answerable']:
                    if 'retrieval_hit' in iter_result:
                        retrieval_hits.append(iter_result['retrieval_hit'])
                        precision_vals.append(iter_result.get('retrieval_precision', 0.0))
                        recall_vals.append(iter_result.get('retrieval_recall', 0.0))
                        f1_vals.append(iter_result.get('retrieval_f1', 0.0))
                
                # Count not_answerable accuracy
                not_answerable_correct_hits.append(iter_result['not_answerable_correct'])
                
                # Count accuracy for all cases
                accuracy_hits.append(iter_result['accuracy'])
                
                # Count response types
                response_type = iter_result.get('response_type', '')
                if response_type:
                    response_types[response_type] = response_types.get(response_type, 0) + 1
        
        # Store metrics for this iteration
        metrics_for_iter = {
            'samples': len(accuracy_hits),
            'accuracy': np.mean(accuracy_hits) * 100 if accuracy_hits else 0.0,
            'not_answerable_accuracy': np.mean(not_answerable_correct_hits) * 100 if not_answerable_correct_hits else 0.0,
            'response_types': response_types
        }
        
        # Add retrieval metrics only if we have evidence_pages and retrieval hits
        if has_evidence_pages and retrieval_hits:
            metrics_for_iter['retrieval_rate'] = np.mean(retrieval_hits) * 100
            metrics_for_iter['retrieval_precision'] = np.mean(precision_vals) * 100 if precision_vals else 0.0
            metrics_for_iter['retrieval_recall'] = np.mean(recall_vals) * 100 if recall_vals else 0.0
            metrics_for_iter['retrieval_f1'] = np.mean(f1_vals) * 100 if f1_vals else 0.0
        
        metrics['by_iteration'][iteration] = metrics_for_iter
    
    # Calculate final metrics (last iteration for each sample)
    final_retrieval_hits = []
    final_accuracy_hits = []
    final_not_answerable_correct_hits = []
    final_response_types = {}
    over_long_page_count = 0
    final_precision_vals = []
    final_recall_vals = []
    final_f1_vals = []
    
    for result in all_results:
        if result['iterations']:
            final_iter = result['iterations'][-1]
            
            # Only calculate retrieval metrics for answerable questions if we have evidence_pages
            if has_evidence_pages and not final_iter['gt_not_answerable']:
                if 'retrieval_hit' in final_iter:
                    final_retrieval_hits.append(final_iter['retrieval_hit'])
                    final_precision_vals.append(final_iter.get('retrieval_precision', 0.0))
                    final_recall_vals.append(final_iter.get('retrieval_recall', 0.0))
                    final_f1_vals.append(final_iter.get('retrieval_f1', 0.0))
            
            # Count not_answerable accuracy
            final_not_answerable_correct_hits.append(final_iter['not_answerable_correct'])
            
            # Count accuracy for all cases
            final_accuracy_hits.append(final_iter['accuracy'])
            
            # Count final response types
            response_type = final_iter.get('response_type', '')
            if response_type:
                final_response_types[response_type] = final_response_types.get(response_type, 0) + 1
        
        # Count over_long_page samples
        if res := result_map.get((result['doc_id'], result['question'])):
            if res.get('is_over_long_page', False):
                over_long_page_count += 1
    
    final_metrics = {
        'samples': len(final_accuracy_hits),
        'accuracy': (np.mean(final_accuracy_hits) * 100) if final_accuracy_hits else 0.0,
        'not_answerable_accuracy': (np.mean(final_not_answerable_correct_hits) * 100) if final_not_answerable_correct_hits else 0.0,
        'response_types': final_response_types,
        'over_long_page_rate': (over_long_page_count / len(all_results) * 100) if all_results else 0.0
    }
    
    # Add retrieval metrics only if we have evidence_pages
    if has_evidence_pages and final_retrieval_hits:
        final_metrics['retrieval_rate'] = np.mean(final_retrieval_hits) * 100
        final_metrics['retrieval_precision'] = np.mean(final_precision_vals) * 100
        final_metrics['retrieval_recall'] = np.mean(final_recall_vals) * 100
        final_metrics['retrieval_f1'] = np.mean(final_f1_vals) * 100
    
    metrics['final'] = final_metrics

    # Calculate subset metrics by evidence_sources and evidence_pages length
    if has_evidence_sources:
        sample_map = {(s['doc_id'], s['question']): s for s in samples}
        subset_by_source = {}
        for result in all_results:
            sample = sample_map.get((result['doc_id'], result['question']), {})
            try:
                sources = sample.get('evidence_sources', [])
                if not isinstance(sources, list):
                    sources = ast.literal_eval(sources)
            except Exception:
                sources = []
            for src in sources:
                subset_by_source.setdefault(src, []).append(result)
        metrics['by_evidence_source'] = {}
        for src, group in subset_by_source.items():
            acc_list = [res['iterations'][-1].get('accuracy', 0) for res in group]
            metrics['by_evidence_source'][src] = {
                'samples': len(acc_list),
                'accuracy': float(np.mean(acc_list) * 100) if acc_list else 0.0
            }
    if has_evidence_pages:
        sample_map = {(s['doc_id'], s['question']): s for s in samples}
        subset_by_length = {'no_pages': [], 'single_page': [], 'multiple_pages': []}
        for result in all_results:
            sample = sample_map.get((result['doc_id'], result['question']), {})
            try:
                pages = sample.get('evidence_pages', [])
                if not isinstance(pages, list):
                    pages = ast.literal_eval(pages)
            except Exception:
                pages = []
            l = len(pages)
            if l == 0:
                subset_by_length['no_pages'].append(result)
            elif l == 1:
                subset_by_length['single_page'].append(result)
            else:
                subset_by_length['multiple_pages'].append(result)
        metrics['by_evidence_pages_length'] = {}
        for category, group in subset_by_length.items():
            acc_list = [res['iterations'][-1].get('accuracy', 0) for res in group]
            metrics['by_evidence_pages_length'][category] = {
                'samples': len(acc_list),
                'accuracy': float(np.mean(acc_list) * 100) if acc_list else 0.0
            }

    # Save metrics
    out_dir = os.path.dirname(args.output_file) or '.'
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Evaluation metrics saved to {args.output_file}")
    
    # Print accuracy metrics for all samples
    print(f"Final metrics: Accuracy = {final_metrics['accuracy']:.2f}%")
    print(f"Not answerable accuracy = {final_metrics['not_answerable_accuracy']:.2f}%")
    print(f"Over long page rate = {final_metrics.get('over_long_page_rate', 0):.2f}%")
    
    # Print retrieval metrics only if we have evidence_pages
    if has_evidence_pages:
        if 'retrieval_rate' in final_metrics:
            print(f"Retrieval rate = {final_metrics.get('retrieval_rate', 0):.2f}%")
            print(f"Retrieval precision = {final_metrics.get('retrieval_precision', 0):.2f}%, Recall = {final_metrics.get('retrieval_recall', 0):.2f}%, F1 = {final_metrics.get('retrieval_f1', 0):.2f}%")

if __name__ == '__main__':
    main()