import os
import sys  # Added for sys module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from utils.openai_helper import initialize_client
import logging
import joblib
from functools import partial
import ast
import numpy as np
# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate document QA responses with GPT-4.1")
    
    # Input/output paths
    parser.add_argument("--results_file", type=str, default="outputs/gemma3/processed_results.jsonl",
                       help="Path to the processed results file to evaluate")
    parser.add_argument("--ground_truth_file", type=str, default="data/MMLongBench/samples.json",
                       help="Path to the ground truth data file")
    parser.add_argument("--output_file", type=str, default="outputs/gemma3/evaluation_results.jsonl",
                       help="Output file to save evaluation results")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-4.1",
                       help="Model name to use for evaluation (default: gpt-4.1)")
    parser.add_argument("--api_key_file", type=str, default="./openaikey",
                       help="File containing the OpenAI API key")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1",
                          help="Base URL for OpenAI API")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens for GPT responses")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for GPT responses (lower for more consistent evaluations)")
    parser.add_argument("--cache_seed", type=int, default=123,
                        help="Seed for OpenAI cache")
    parser.add_argument("--add_notanswerable", action="store_true",
                        help="Add 'Not answerable' to the ground truth")
    parser.add_argument("--n_jobs", type=int, default=8,
                        help="Number of parallel jobs to run (-1 for all cores)")
    
    return parser.parse_args()


def load_ground_truth(ground_truth_file):
    """Load the ground truth data."""
    # Load the ground truth data using pandas
    print("load_ground_truth")
    ground_truth_df = pd.read_json(ground_truth_file)

    # ground_truth_df = ground_truth_df[ground_truth_df.apply(lambda row: row['answer'] != 'Not answerable', axis=1)].reset_index(drop=True)
    
    # Create a mapping of doc_id+question to answer
    ground_truth_map = {}
    for _, row in ground_truth_df.iterrows():
        key = f"{row['doc_id']}_{row['question']}"
        ground_truth_map[key] = row.get('answer', '')
    
    return ground_truth_map

def load_results(results_file):
    """Load the model-generated results from a standard JSON file."""
    print("Loading results")
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_response(client, model, predicted_answer, ground_truth, question, max_tokens=1024, temperature=0.0):
    """
    Use GPT-4.1 to evaluate a predicted answer against the ground truth.
    Returns a score (0-1) in a dictionary and can be parsed by json.loads, e.g. {{"binary_correctness": 1}}
    """
    try:
        prompt = f"""Question: {question}
Predicted Answer: {predicted_answer}
Ground Truth Answer: {ground_truth}

Please evaluate if the predicted answer is correct compared to the ground truth.
Score the answer on:
Binary correctness (0-1): 1 if the answer is correct, 0 if it is incorrect

Return only a string with these scores in a dictionary and can be parsed by json.loads, e.g. {{"binary_correctness": 1}}"""
        print("Using evaluate response")

        response = client.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        
        try:
            # Parse the JSON response from the evaluation text
            evaluation_dict = json.loads(evaluation_text)
            score = evaluation_dict.get("binary_correctness", 0)
        except json.JSONDecodeError:
            score = -1,
        return {
            "score": score,
            "explanation": evaluation_text
        }
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return {"score": 0, "explanation": f"Evaluation error: {str(e)}"}

def extract_answer_from_original(original_response):
    """Extract answer from original response by splitting on 'Answer:' and taking the last part."""
    if not original_response:
        return ""
    
    if "Answer:" in original_response:
        split_word = "Answer:"
    elif "\\boxed{" in original_response:
        split_word = "\\boxed{"
    elif "<answer>" in original_response:
        split_word = "<answer>"
    elif "Answer" in original_response:
        split_word = "Answer"
    else:
        return original_response
    
    parts = original_response.split(split_word)
    if len(parts) > 1:
        return parts[-1].strip()

def process_item(item, client, model, ground_truth_map, max_tokens, temperature, add_notanswerable):
    """Process a single evaluation item."""
    
    try:
        key = f"{item['doc_id']}_{item['question']}"
        ground_truth = ground_truth_map.get(key, '')
        
        # Skip items with "Not answerable" ground truth
        if ground_truth == "Not answerable" and not add_notanswerable:
            return None
        
        qa_error = item.get('error', None)
        if qa_error is not None and type(qa_error) == str:
            return {
                "doc_id": item['doc_id'],
                "question": item['question'],
                "error": item['error'],
                "score": 0,
                "explanation": "Error during evaluation"
            }

        if not ground_truth:
            return {
                "doc_id": item['doc_id'],
                "question": item['question'],
                "error": "No ground truth found",
                "score": 0,
                "explanation": "No ground truth answer available for evaluation"
            }
        
        # Get the processed response or extract from original if not available
        processed_response = item.get('final_answer')
        if not processed_response and 'ori_response' in item:
            processed_response = extract_answer_from_original(item['ori_response'])
        
        # Evaluate the response
        eval_result = evaluate_response(
            client, 
            model, 
            processed_response or '', 
            ground_truth, 
            item['question'],
            max_tokens,
            temperature
        )
        
        return {
            "score": eval_result['score'],
            "doc_id": item['doc_id'],
            "question": item['question'],
            "predicted_answer": processed_response or '',
            "ground_truth": ground_truth,
            "explanation": eval_result['explanation'],
            "error": None
        }
    except Exception as e:
        logger.error(f"Error evaluating response for {item.get('doc_id')}: {str(e)}")
        return {
            "doc_id": item.get('doc_id'),
            "question": item.get('question'),
            "error": f"Evaluation error: {str(e)}",
            "score": 0,
            "explanation": f"Exception during evaluation: {str(e)}"
        }

def main():
    """Main function to run the evaluation process."""
    args = parse_arguments()
    
    # Initialize the OpenAI client
    client = initialize_client(args=args)
    if client is None:
        print("Exiting script as OpenAI client could not be initialized.")
        return
    print("In the main function")
    # Load the ground truth data
    ground_truth_map = load_ground_truth(args.ground_truth_file)
    # print(ground_truth_map)
    
    # Load the results
    results = load_results(args.results_file)
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Clear the output file at the beginning
    with open(args.output_file, 'w') as f:
        f.write('')
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_item,
        client=client,
        model=args.model,
        ground_truth_map=ground_truth_map,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        add_notanswerable=args.add_notanswerable
    )
    
    # Process items in parallel using joblib
    print(f"Processing {len(results)} items with {args.n_jobs} parallel jobs")
    evaluation_results = joblib.Parallel(n_jobs=args.n_jobs, backend="threading")(
        joblib.delayed(process_func)(item) for item in tqdm(results, desc="Evaluating responses")
    )
    
    # Filter out None results (skipped items)
    evaluation_results = [result for result in evaluation_results if result is not None]
    
    # Write results to file
    with open(args.output_file, 'a') as f:
        for evaluation in evaluation_results:
            f.write(json.dumps(evaluation) + '\n')
    
    # Calculate and print summary statistics
    scores = [item['score'] for item in evaluation_results if 'score' in item]
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"\nEvaluation complete. Average score: {average_score * 100 :.2f} %")
        print(f"Results saved to: {args.output_file}")
        # Calculate subset metrics by evidence_sources and evidence_pages length
        samples_df = pd.read_json(args.ground_truth_file)
        samples = samples_df.to_dict('records')
        sample_map = {(s['doc_id'], s['question']): s for s in samples}
        # Subset by evidence_sources
        subset_by_source = {}
        for result in evaluation_results:
            key = (result['doc_id'], result['question'])
            sample = sample_map.get(key, {})
            sources = sample.get('evidence_sources', [])
            if not isinstance(sources, list):
                try:
                    sources = ast.literal_eval(sources)
                except:
                    sources = []
            for src in sources:
                subset_by_source.setdefault(src, []).append(result)
        print("\nSubset metrics by evidence source:")
        for src, group in subset_by_source.items():
            scores_list = [item['score'] for item in group if 'score' in item]
            accuracy = float(np.mean(scores_list) * 100) if scores_list else 0.0
            print(f"{src}: samples={len(scores_list)}, accuracy={accuracy:.2f}%")
        # Subset by evidence_pages length
        subset_by_length = {'no_pages': [], 'single_page': [], 'multiple_pages': []}
        for result in evaluation_results:
            key = (result['doc_id'], result['question'])
            sample = sample_map.get(key, {})
            pages = sample.get('evidence_pages', [])
            if not isinstance(pages, list):
                try:
                    pages = ast.literal_eval(pages)
                except:
                    pages = []
            l = len(pages)
            if l == 0:
                subset_by_length['no_pages'].append(result)
            elif l == 1:
                subset_by_length['single_page'].append(result)
            else:
                subset_by_length['multiple_pages'].append(result)
        print("\nSubset metrics by evidence pages length:")
        for cat, group in subset_by_length.items():
            scores_list = [item['score'] for item in group if 'score' in item]
            accuracy = float(np.mean(scores_list) * 100) if scores_list else 0.0
            print(f"{cat}: samples={len(scores_list)}, accuracy={accuracy:.2f}%")
    else:
        print("\nEvaluation complete, but no valid scores were calculated.")

if __name__ == "__main__":
    main()