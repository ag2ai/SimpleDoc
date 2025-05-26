import os
import sys  # Added for sys module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import base64
import fitz  # PyMuPDF, imported as fitz
import pandas as pd
from tqdm import tqdm
import ast  # For safely evaluating string representations of lists if needed
from utils.openai_helper import initialize_client
import argparse  # Added for command-line argument parsing
import random  # For random subsampling
import json  # For JSON serialization
from joblib import Parallel, delayed  # For parallelization
from utils.pipeline_utils import generate_hash

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document page summarization with LLMs")
    
    # Data paths
    parser.add_argument("--data_base_path", type=str, default="data/MMLongBench/documents",
                        help="Base path to PDF documents")
    parser.add_argument("--input_file", type=str, default="data/MMLongBench/samples.json",
                        help="Input JSON file with questions and document references")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen25_32b/summaries",
                        help="Output directory to save result files")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct",
                        help="Model name to use")
    parser.add_argument("--api_key_file", type=str, default="./deepinfrakey",
                        help="File containing the API key")
    parser.add_argument("--base_url", type=str, 
                        default="http://cn-w-1.hpc.engr.oregonstate.edu:8000/v1",
                        help="Base URL for API")
    parser.add_argument("--cache_seed", type=int, default=123,
                        help="Seed for OpenAI cache")
    
    # Image and model parameters
    parser.add_argument("--image_dpi", type=int, default=150,
                        help="DPI for rendering PDF pages")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens for LLM response")
    parser.add_argument("--prompt_file", type=str, default="prompts/general_summary_prompt.txt",
                        help="Path to the prompt file")
    
    # Subsampling parameter
    parser.add_argument("--subsample", type=int, default=None,
                        help="Number of samples to randomly select from valid questions with target pages. If None, use all samples.")
    
    # Text extraction parameter
    parser.add_argument("--extract_text", action="store_true", default=True,
                        help="Extract text from PDF pages and include it in the prompt")
    
    # Parallelization parameter
    parser.add_argument("--n_jobs", type=int, default=32,
                        help="Number of parallel jobs for page processing. Default is -1 (use all cores).")
    
    # Verbosity parameter
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose output for parallel processing")
    
    # Skip existing files parameter
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Skip processing if output file already exists")
    
    return parser.parse_args()

# Import helper functions from target_page_qa.py
from modules.step03_target_page_qa import convert_pdf_pages_to_base64_images

def query_llm_for_summary(base64_images_list: list[str], model_name: str, client, max_tokens: int, prompt_file: str, page_texts: list[str] = None) -> str | None:
    """
    Sends a list of base64 encoded images to the LLM to generate a summary.

    Args:
        base64_images_list (list[str]): List of base64 encoded PNG image strings.
        model_name (str): The OpenAI model to use.
        client: The OpenAI client wrapper
        max_tokens (int): Maximum tokens for response
        prompt_file (str): Path to the prompt template file
        page_texts (list[str]): List of text extracted from PDF pages

    Returns:
        str | None: The LLM's response text, or None if an error occurs.
    """
    if not base64_images_list:
        print("No images provided to the LLM. Cannot proceed with the query.")
        return None
    
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    # Insert page text into prompt
    text_content = ""
    if page_texts is not None and any(t.strip() for t in page_texts):
        text_content = "\n\n".join(page_texts)
    prompt = prompt.format(PAGE_TEXT=text_content)

    messages_content = [{"type": "text", "text": prompt}]
    for img_b64 in base64_images_list:
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high"  # or "low" or "high"
            }
        })

    try:
        response = client.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": messages_content
                }
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.001,
            extra_body={"repetition_penalty": 1.05},
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

def extract_summary(response: str) -> str:
    """
    Extract the summary from the response (within <summary> tags).
    
    Args:
        response (str): The full response from the LLM.
        
    Returns:
        str: The extracted summary, or None if extraction fails.
    """
    try:
        # Extract content between <summary> tags
        start_idx = response.find('<summary>')
        end_idx = response.find('</summary>')
        
        if start_idx != -1 and end_idx != -1:
            summary = response[start_idx + len('<summary>'):end_idx].strip()
            return response[start_idx:end_idx + len('</summary>')]
        else:
            # If no tags found, return the whole response
            return response
    except Exception as e:
        raise e

def process_single_page(doc_id, page_num, pdf_file_path, args, client=None):
    """
    Process a single page of a document and generate summary.
    
    Args:
        doc_id (str): Document ID
        page_num (int): Page number
        pdf_file_path (str): Path to the PDF file
        args: Command line arguments
        client: OpenAI client wrapper (if None, will create a new client for thread safety)
        
    Returns:
        dict: Results for this page
    """
    try:
        # Create a new client for each worker to ensure thread safety
        if client is None:
            local_client = initialize_client(args)
        else:
            local_client = client
            
        if local_client is None:
            return {
                'page_num': page_num,
                'error': 'Failed to initialize OpenAI client'
            }
            
        # Load page text from pre-extracted JSON if requested
        if args.extract_text:
            text_file_path = pdf_file_path.replace('.pdf', '.txt').replace('/documents/', '/text_doc/')
            try:
                with open(text_file_path, 'r') as f:
                    pdf_data = json.load(f)
                page_text = str(pdf_data[page_num - 1]) if page_num - 1 < len(pdf_data) else ""
            except Exception as e:
                if args.verbose:
                    print(f"Failed to load text for {doc_id}, page {page_num}: {e}")
                page_text = ""
        else:
            page_text = ""

        # Convert the current page to base64 image without extracting text from PDF
        base64_images, _ = convert_pdf_pages_to_base64_images(pdf_file_path, [page_num], args.image_dpi, False)
        page_texts = [page_text]

        if not base64_images:
            if args.verbose:
                print(f"Failed to get image for {doc_id}, page {page_num}. Skipping LLM call.")
            return {
                'page_num': page_num,
                'error': 'Failed to convert page to image'
            }

        if args.verbose:
            print(f"Successfully converted page {page_num} to image for LLM.")
        
        # Generate summary for this page
        response = query_llm_for_summary(base64_images, args.model, local_client, args.max_tokens, args.prompt_file, page_texts)
        
        if response:
            summary = extract_summary(response)
            
            if args.verbose:
                print(f"Generated summary for {doc_id}, page {page_num}")
            
            return {
                'page_num': page_num,
                'summary': summary
            }
        else:
            if args.verbose:
                print(f"Failed to get a summary from LLM for {doc_id}, page {page_num}.")
            return {
                'page_num': page_num,
                'error': 'LLM query failed'
            }
    except Exception as e:
        if args.verbose:
            print(f"Error processing page {page_num} of document {doc_id}: {e}")
        return {
            'page_num': page_num,
            'error': str(e)
        }

def summarize_document_pages(dataset_df: pd.DataFrame, args, client):
    """
    Iterates through the dataset, processes each document page by page in parallel, and generates summaries.
    Creates a separate JSON file for each document.
    
    Args:
        dataset_df (pd.DataFrame): DataFrame containing the dataset
        args: Command line arguments
        client: OpenAI client wrapper
    """
    if not os.path.isdir(args.data_base_path):
        print(f"Error: The `data_base_path` ('{args.data_base_path}') does not exist or is not a directory.")
        print("Please ensure the path is correct and contains your PDF documents.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a set to track already processed documents
    processed_docs = set()
    
    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Processing Documents"):
        try:
            doc_id = row['doc_id']
            
            # Skip if document has already been processed
            if doc_id in processed_docs:
                print(f"Skipping duplicate document: {doc_id}")
                continue
            
            # Add to processed documents
            processed_docs.add(doc_id)
            
            # Check if output file already exists
            output_file = os.path.join(args.output_dir, f"{doc_id}.json")
            if args.skip_existing and os.path.exists(output_file):
                print(f"Skipping document {doc_id} - output file already exists: {output_file}")
                continue
            
            pdf_file_path = os.path.join(args.data_base_path, doc_id)
            print(f"\nProcessing Document ID: {doc_id}")

            if not os.path.exists(pdf_file_path):
                print(f"Error: PDF file not found at {pdf_file_path}")
                continue
                
            # Open the PDF to get the total page count
            doc = fitz.open(pdf_file_path)
            total_pages = doc.page_count
            doc.close()
            
            # Create a dictionary to store summaries for all pages of this document
            doc_results = {
                'doc_id': doc_id,
                'pages': {}
            }
            
            # Process each page of the document in parallel
            print(f"Processing {total_pages} pages in parallel with {args.n_jobs} jobs...")
            
            # Create a list of page numbers to process
            page_nums = list(range(1, total_pages + 1))
            
            # Don't pass the client to ensure each worker creates its own client instance for thread safety
            page_results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_single_page)(
                    doc_id, page_num, pdf_file_path, args, None
                ) for page_num in tqdm(page_nums, desc=f"Pages of {doc_id}", leave=False)
            )
            
            # Organize the results into the doc_results dictionary
            for page_result in page_results:
                page_num = page_result.pop('page_num')
                doc_results['pages'][str(page_num)] = page_result
            
            # Write the document results to a JSON file using document ID in the filename
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_results, f, ensure_ascii=False, indent=2)
                
            print(f"Saved results for document {doc_id} to {output_file}")

        except Exception as e:
            print(f"An unexpected error occurred while processing document '{row.get('doc_id', 'Unknown')}': {e}")
            # Write error to a separate error file using document ID
            doc_id = row.get('doc_id', 'unknown')
            error_file = os.path.join(args.output_dir, f"error_{doc_id}.json")
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'doc_id': doc_id,
                    'error': str(e)
                }, f, ensure_ascii=False, indent=2)

    print("\n--- Processing Complete ---")
    return None

def main():
    """Main function to run the document page summarization process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize the OpenAI client
    client = initialize_client(args)
    if client is None:
        print("Exiting script as OpenAI client could not be initialized.")
        return
    
    # Load the dataset
    try:
        dataset_df = pd.read_json(args.input_file)
    except Exception as e:
        print(f"Error loading dataset from {args.input_file}: {e}")
        return

    # Filter valid samples (those with valid documents)
    valid_samples = []
    processed_docs = set()
    
    for idx, row in dataset_df.iterrows():
        try:
            doc_id = row['doc_id']
            
            # Skip if we've already seen this document
            if doc_id in processed_docs:
                continue
            
            processed_docs.add(doc_id)
            
            # Check if PDF file exists
            pdf_file_path = os.path.join(args.data_base_path, doc_id)
            if os.path.exists(pdf_file_path):
                valid_samples.append(idx)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"Found {len(valid_samples)} valid unique documents out of {len(dataset_df)} total samples")
    
    # Apply subsampling if requested
    if args.subsample is not None and args.subsample < len(valid_samples):
        random.seed(42)  # For reproducibility
        selected_indices = random.sample(valid_samples, args.subsample)
        print(f"Randomly selected {args.subsample} documents for processing")
    else:
        selected_indices = valid_samples
        if args.subsample is not None:
            print(f"Requested {args.subsample} documents but only {len(valid_samples)} valid documents available")
    
    # Filter the dataset to include only the selected samples
    filtered_dataset_df = dataset_df.iloc[selected_indices].reset_index(drop=True)
    
    # Process the documents (will write individual JSON files during processing)
    summarize_document_pages(filtered_dataset_df, args, client)
    
    print(f"All document summaries have been saved to '{args.output_dir}'")

if __name__ == "__main__":
    main() 