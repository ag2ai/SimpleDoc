import os
import base64
import fitz  # PyMuPDF, imported as fitz
import pandas as pd
from tqdm import tqdm
import ast  # For safely evaluating string representations of lists if needed
from utils.openai_helper import initialize_client
import argparse  # Added for command-line argument parsing
import random  # For random subsampling
import json  # For reading the retrievals file
from joblib import Parallel, delayed  # For parallelization
import fcntl

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document reasoning with LLMs")
    
    # Data paths
    parser.add_argument("--data_base_path", type=str, default="data/MMLongBench/documents",
                        help="Base path to PDF documents")
    parser.add_argument("--input_file", type=str, default="data/MMLongBench/samples.json",
                        help="Input JSON file with questions and document references")
    parser.add_argument("--output_file", type=str, default="outputs/qwen25/processed_results.jsonl",
                        help="Output file to save results")
    parser.add_argument("--retrievals_file", type=str, default=None,
                        help="JSON file containing retrieved pages (from page_retrieval.py)")
    parser.add_argument("--use_golden_retrieval", action="store_true", default=False)
    parser.add_argument("--max_pages", type=int, default=20,
                        help="Maximum number of pages to pass to the LLM. Default is -1 (use all retrieved pages).")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct",
                        help="Model name to use")
    parser.add_argument("--api_key_file", type=str, default="./deepinfrakey",
                        help="File containing the API key")
    parser.add_argument("--base_url", type=str, 
                        default="http://cn-w-1.hpc.engr.oregonstate.edu:8000/v1",
                        help="Base URL for API")
    parser.add_argument("--cache_seed", type=int, default=None,
                        help="Seed for OpenAI cache")
    
    # Image and model parameters
    parser.add_argument("--image_dpi", type=int, default=150,
                        help="DPI for rendering PDF pages")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens for LLM response")
    parser.add_argument("--prompt_file", type=str, default="prompts/doc_qa_prompt.txt",
                        help="Path to the prompt file")
    
    # Subsampling parameter
    parser.add_argument("--subsample", type=int, default=None,
                        help="Number of samples to randomly select from valid questions with target pages. If None, use all samples.")
    
    # Text extraction parameter
    parser.add_argument("--extract_text", action="store_true", default=False,
                        help="Extract text from PDF pages and include it in the prompt")

    parser.add_argument("--text_only", action="store_true", default=False)
    parser.add_argument("--empty_doc_exp", action="store_true", default=False)
    parser.add_argument("--random_page_exp", action="store_true", default=False)
    
    # Parallelization parameter
    parser.add_argument("--n_jobs", type=int, default=32,
                        help="Number of parallel jobs for processing. Default is 1 (sequential).")
    
    # Filtering parameter for "Not answerable" items
    parser.add_argument("--add_notanswerable", action="store_true", default=False,
                        help="Include items with 'Ground truth: Not answerable' in processing. By default, these items are filtered out.")
    
    return parser.parse_args()

# --- Helper Functions ---

def safe_write_to_file(file_path, content):
    """
    Safely write content to a file using file locking
    """
    with open(file_path, 'a') as f:
        # Get an exclusive lock on the file
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # Write content
            f.write(content + '\n')
        finally:
            # Release the lock
            fcntl.flock(f, fcntl.LOCK_UN)


def convert_pdf_pages_to_base64_images(pdf_path: str, page_numbers: list[int], image_dpi: int, extract_text: bool = False) -> tuple[list[str], list[str]]:
    """
    Opens a PDF, extracts specified pages, renders them as PNG images,
    and returns them as a list of base64 encoded strings.
    Optionally extracts text from the pages as well.

    Args:
        pdf_path (str): The full path to the PDF file.
        page_numbers (list[int]): A list of 0-indexed page numbers to extract.
        image_dpi (int): DPI for rendering PDF pages
        extract_text (bool): Whether to extract text from the pages as well.

    Returns:
        tuple[list[str], list[str]]: A tuple of (base64_images, page_texts)
                   Returns empty lists if errors occur or PDF not found.
    """
    base64_images = []
    page_texts = []
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return base64_images, page_texts

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}")
        return base64_images, page_texts

    for page_num in sorted(list(set(page_numbers))):  # Process unique pages in order
        if 0 <= page_num - 1 < doc.page_count:
            try:
                page = doc.load_page(page_num - 1)  # Page numbers are 0-indexed in PyMuPDF
                
                # Render page to a pixmap (raster image)
                pix = page.get_pixmap(dpi=image_dpi)
                img_bytes = pix.tobytes("png")  # Get image bytes in PNG format
                base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                base64_images.append(base64_encoded_image)
                
                # Extract text if requested
                if extract_text:
                    text = page.get_text()
                    page_texts.append(text)
                else:
                    page_texts.append("")
                    
            except Exception as e:
                print(f"Error processing page {page_num} from '{pdf_path}': {e}")
        else:
            print(
                f"Warning: Page number {page_num} is out of range for '{pdf_path}' (Total pages: {doc.page_count}). Skipping this page.")

    doc.close()
    return base64_images, page_texts


def query_llm_with_images(question: str, base64_images_list: list[str], model_name: str, client, max_tokens: int, prompt_file: str, original_page_numbers: list[int], page_texts: list[str] = None, document_summary: str = None, args=None, client_img=None) -> str | None:
    """
    Sends a question and a list of base64 encoded images to the OpenAI API.
    Optionally includes extracted text from the PDF pages.

    Args:
        question (str): The question to ask the LLM.
        base64_images_list (list[str]): List of base64 encoded PNG image strings.
        model_name (str): The OpenAI model to use (e.g., "gpt-4o").
        client: The OpenAI client wrapper
        max_tokens (int): Maximum tokens for response
        prompt_file (str): Path to the prompt template file
        original_page_numbers (list[int]): List of original page numbers corresponding to the images
        page_texts (list[str]): List of text extracted from PDF pages
        document_summary (str): High-level summary of the document in relation to the question

    Returns:
        str | None: The LLM's response text, or None if an error occurs.
    """
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    # Format document summary if available
    doc_summary_text = document_summary.strip() if document_summary else ""
    
    # Format retrieved page numbers using original page numbers
    retrieved_page_numbers = ", ".join([str(page_num) for page_num in original_page_numbers])
    
    # Include extracted text in the prompt if available
    if page_texts and any(text.strip() for text in page_texts):
        # Use original page numbers for text sections
        text_content = "\n\n".join([f"--- Page {page_num} Text ---\n{text}" 
                                   for page_num, text in zip(original_page_numbers, page_texts) 
                                   if text.strip()])
        prompt = prompt.format(QUESTION=question, DOCUMENT_SUMMARY=doc_summary_text, RETRIEVED_PAGE_NUMBERS=retrieved_page_numbers, PAGE_TEXT=text_content)
    else:
        prompt = prompt.format(QUESTION=question, DOCUMENT_SUMMARY=doc_summary_text, RETRIEVED_PAGE_NUMBERS=retrieved_page_numbers, PAGE_TEXT="")
    
    messages_content = [{"type": "text", "text": prompt}]

    if not args.text_only:
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
            temperature=0.6 if args.text_only else 0.1,
            top_p=0.95 if args.text_only else 0.001,
            extra_body={"top_k": 20, "min_p": 0} if args.text_only else {"repetition_penalty": 1.05},
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

    if args.text_only and answer and "<visual_query>" in answer:
        start_idx = answer.index('<visual_query>') + len('<visual_query>')
        end_idx = answer.index('</visual_query>') if '</visual_query>' in answer else -1
        visual_query = answer[start_idx:end_idx].strip()

        # Create request to Qwen-VL with base64 images
        print(f"Because the answer contains <visual_query>, we will call Qwen-VL with query '{visual_query}' and images.")
        response = client_img.create(
            model="Qwen/Qwen2.5-VL-32B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": visual_query},
                        *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i64}", "detail": "high"}} for i64 in base64_images_list]
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.001,
            extra_body={"repetition_penalty": 1.05},
        )

        img_query_response = response.choices[0].message.content

        # Append the image query response to the original answer as user message and request again
        response = client.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": messages_content
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": img_query_response}
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            extra_body={"top_k": 20, "min_p": 0},
        )
        answer = response.choices[0].message.content

    return answer


def postprocess_answer(ori_response: str) -> tuple[str, str]:
    """
    Extract the answer from the original response. 
    
    Returns:
        tuple[str, str]: (response_text, response_type)
        response_type is one of: "answer", "not_answerable", "query_update", "fallback"
    """
    # Try to extract using <answer> tags
    try:
        start_idx = ori_response.index('<answer>') + len('<answer>')
        end_idx = ori_response.index('</answer>') if '</answer>' in ori_response else -1
        answer = ori_response[start_idx:end_idx].strip()
        if answer:
            return answer, "answer"
    except ValueError:
        pass  # If <answer> tags not found, try other tags
    
    # Try to extract using <not_answerable> tags
    try:
        start_idx = ori_response.index('<not_answerable>') + len('<not_answerable>')
        end_idx = ori_response.index('</not_answerable>') if '</not_answerable>' in ori_response else -1
        answer = ori_response[start_idx:end_idx].strip()
        if answer:
            return answer, "not_answerable"
    except ValueError:
        pass  # If <not_answerable> tags not found, try other tags

    # Try to extract using <visual_query> tags
    try:
        start_idx = ori_response.index('<visual_query>') + len('<visual_query>')
        end_idx = ori_response.index('</visual_query>') if '</visual_query>' in ori_response else -1
        answer = ori_response[start_idx:end_idx].strip()
        if answer:
            return answer, "visual_query"
    except ValueError:
        pass

    # Try to extract using <query_update> tags
    try:
        start_idx = ori_response.index('<query_update>') + len('<query_update>')
        end_idx = ori_response.index('</query_update>') if '</query_update>' in ori_response else -1
        answer = ori_response[start_idx:end_idx].strip()
        if "<notes>" in ori_response:
            start_idx = ori_response.index('<notes>') + len('<notes>')
            end_idx = ori_response.index('</notes>') if '</notes>' in ori_response else -1
            notes = ori_response[start_idx:end_idx].strip()
        else:
            notes = ""
        if answer:
            return (answer, notes), "query_update"
    except ValueError:
        pass  # If <query_update> tags not found, try fallback
    
    # Fallback: If no tags found, try splitting by "Answer:"
    if "answer:" in ori_response.lower():
        start_idx = ori_response.lower().index("answer:") + len("answer:")
        return ori_response[start_idx:], "answer"

    return ori_response, "original"  # Return None if no valid answer found

# --- Main Processing Logic ---
def process_single_document(row_data, args, retrieval_lookup=None, client=None):
    """
    Process a single document-question pair.
    
    Args:
        row_data (tuple): Tuple containing (idx, row) where idx is the row index and row is the DataFrame row
        args: Command line arguments
        retrieval_lookup (dict, optional): Dictionary mapping (doc_id, question) to relevant pages
        client: OpenAI client wrapper (if None, will create a new client for thread safety)
        
    Returns:
        dict: Result with the processed response and other metadata
    """
    idx, row = row_data
    
    try:
        # Create a new client for each worker to ensure thread safety
        if client is None:
            local_client = initialize_client(args)
        else:
            local_client = client
            
        doc_id = row['doc_id']
        question = row['question']
        
        # Initialize document summary
        document_summary = None
        
        # Use retrieved pages if available, otherwise use ground truth
        if retrieval_lookup:
            # Get retrieval info including pages and document summary if available
            if (doc_id, question) not in retrieval_lookup.keys():
                print(f"Row {idx}: No retrieval info found for {doc_id} and {question}, skip this row.")
                return None
            retrieval_info = retrieval_lookup.get((doc_id, question))

            evidence_pages = ast.literal_eval(row['evidence_pages']) if args.use_golden_retrieval else retrieval_info.get('pages', [])
            document_summary = retrieval_info.get('document_summary', None)
                
            # Limit number of pages if max_pages is set
            if args.max_pages > 0 and len(evidence_pages) > args.max_pages:
                evidence_pages = evidence_pages[:args.max_pages]
                print(f"Row {idx}: Limiting to first {args.max_pages} pages as per max_pages setting.")
                
            source_of_pages = "retrieved"
        else:
            evidence_pages_data = row['evidence_pages'] if 'evidence_pages' in row else []
            # Convert evidence_pages to a list of integers if it's a string
            if isinstance(evidence_pages_data, str):
                evidence_pages = ast.literal_eval(evidence_pages_data)
            else:
                evidence_pages = evidence_pages_data
            source_of_pages = "ground_truth"
            retrieval_info = {}

        pdf_file_path = os.path.join(args.data_base_path, doc_id)
        print(
            f"\nProcessing Row {idx}: Document ID: {doc_id}, Question: '{question}', {source_of_pages.capitalize()} Pages: {evidence_pages}")

        if args.extract_text:
            with open(pdf_file_path.replace('.pdf', '.txt').replace('/documents/', '/text_doc/'), 'rb') as f:
                pdf_data = json.load(f)
                page_texts = [str(pdf_data[int(i)-1]) for i in evidence_pages]
        else:
            page_texts = ""

        if args.random_page_exp:
            with open(pdf_file_path.replace('.pdf', '.txt').replace('/documents/', '/text_doc/'), 'rb') as f:
                pdf_data = json.load(f)
                page_len = len(pdf_data)
            # Randomly select 6 pages from the whole document
            evidence_pages = random.sample(range(1, page_len + 1), 6)

        if args.empty_doc_exp:
            base64_images = []
        else:
            base64_images, _ = convert_pdf_pages_to_base64_images(
                pdf_file_path, evidence_pages, args.image_dpi, extract_text=False
            )

        print(f"Row {idx}: Successfully converted {len(base64_images)} page(s) to images for LLM.")

        # 2. Query LLM with the question and images
        ori_response = query_llm_with_images(
            retrieval_info["updated_question"] if "updated_question" in retrieval_info.keys() else question,
            base64_images, 
            args.model, 
            local_client, 
            args.max_tokens, 
            args.prompt_file,
            evidence_pages,  # Original page numbers as required argument 
            page_texts, 
            document_summary,
            args=args,
        )
        
        # 3. Post-process the answer
        if ori_response:
            answer, response_type = postprocess_answer(ori_response)
        else:
            answer, response_type = None, None

        if answer:
            print(f"Row {idx}: LLM Response Type: {response_type}, Content: {answer}.")  # Print a snippet
        else:
            print(f"Row {idx}: Failed to get an answer from LLM for {doc_id}.")

        result = {
            'processed_response': answer,
            'response_type': response_type,
            'doc_id': doc_id,
            'question': question,
            'evidence_pages': evidence_pages,
            'ori_response': ori_response,
            'page_source': source_of_pages,
            'error': None if ori_response else 'LLM query failed',
            'idx': idx
        }
        
        # Add information about page limiting if applicable
        if retrieval_lookup and args.max_pages > 0:
            result['pages_limited'] = len(retrieval_lookup.get((doc_id, question), {}).get('pages', [])) > args.max_pages
            if result['pages_limited']:
                result['original_page_count'] = len(retrieval_lookup.get((doc_id, question), {}).get('pages', []))
        
        safe_write_to_file(
            args.output_file, 
            pd.Series(result).to_json(force_ascii=False)
        )
        return result
    
    except Exception as e:
        print(f"An unexpected error occurred while processing row {idx}: {e}")
        result = {
            'doc_id': row.get('doc_id', 'Unknown'),
            'question': row.get('question', 'Unknown'),
            'error': str(e),
            'idx': idx
        }
        
        # Write result to output file immediately
        safe_write_to_file(
            args.output_file, 
            pd.Series(result).to_json(force_ascii=False)
        )
        return result

def process_documents(dataset_df: pd.DataFrame, args, client, retrieval_data=None):
    """
    Iterates through the dataset, processes each document, and queries the LLM.
    Writes results to JSONL file during processing.
    Uses parallel processing for improved performance.
    
    Args:
        dataset_df (pd.DataFrame): DataFrame containing the dataset
        args: Command line arguments
        client: OpenAI client wrapper
        retrieval_data (list, optional): List of retrievals from page_retrieval.py
    """
    if not os.path.isdir(args.data_base_path):
        print(f"Error: The `data_base_path` ('{args.data_base_path}') does not exist or is not a directory.")
        print("Please ensure the path is correct and contains your PDF documents.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Clear the output file at the beginning
    with open(args.output_file, 'w') as f:
        f.write('')  # Clear file content
    
    # Create a lookup for retrieval data if provided
    retrieval_lookup = {}
    if retrieval_data:
        for item in retrieval_data:
            if "original_question" in item:
                # Use the original question for matching
                key = (item['doc_id'], item['original_question'])
                # Store both relevant pages and document summary
                retrieval_lookup[key] = {
                    'pages': item.get('relevant_pages', []),
                    'document_summary': item.get('document_summary', ''),
                    'updated_question': item['question']
                }
            else:
                key = (item['doc_id'], item['question'])
                # Store both relevant pages and document summary
                retrieval_lookup[key] = {
                    'pages': item.get('relevant_pages', []),
                    'document_summary': item.get('document_summary', '')
                }
    
    # Prepare data for parallel processing
    row_data = [(idx, row) for idx, row in dataset_df.iterrows()]
    
    if args.n_jobs > 1:
        print(f"Processing {len(row_data)} documents in parallel with {args.n_jobs} jobs...")
        
        # Process the documents in parallel
        # Don't pass the client to ensure each worker creates its own client instance for thread safety
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_single_document)(
                data, args, retrieval_lookup, None
            ) for data in tqdm(row_data, desc="Processing Documents")
        )
    else:
        print(f"Processing {len(row_data)} documents sequentially...")
        results = []
        for data in tqdm(row_data, desc="Processing Documents"):
            result = process_single_document(data, args, retrieval_lookup, client)
            results.append(result)
    
    # Convert results to a DataFrame for return value
    results_df = pd.DataFrame(results)
    print("\n--- Processing Complete ---")
    if len(results_df) > 0:
        print(results_df.head())
    
    return results_df

def main():
    """Main function to run the document reasoning process."""
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
    
    # Filter out "Not answerable" items unless --add_notanswerable is set
    if not args.add_notanswerable:
        # Count before filtering
        original_count = len(dataset_df)
        
        # Apply the filter
        dataset_df = dataset_df[dataset_df.apply(lambda row: row['answer'] != 'Not answerable', axis=1)].reset_index(drop=True)
        
        # Print statistics
        filtered_count = len(dataset_df)
        print(f"Filtered out {original_count - filtered_count} 'Not answerable' items. Remaining: {filtered_count} items.")
    
    # Load retrievals data if provided
    retrieval_data = None
    if args.retrievals_file:
        try:
            with open(args.retrievals_file, 'r', encoding='utf-8') as f:
                retrieval_data = json.load(f)
            print(f"Loaded {len(retrieval_data)} retrieval results from {args.retrievals_file}")
        except Exception as e:
            print(f"Error loading retrieval data from {args.retrievals_file}: {e}")
            return

    valid_samples = list(range(len(dataset_df)))
    
    # Apply subsampling if requested
    if args.subsample is not None and args.subsample < len(valid_samples):
        random.seed(42)  # For reproducibility
        selected_indices = random.sample(valid_samples, args.subsample)
        print(f"Randomly selected {args.subsample} samples for processing")
    else:
        selected_indices = valid_samples
        if args.subsample is not None:
            print(f"Requested {args.subsample} samples but only {len(valid_samples)} valid samples available")
    
    # Filter the dataset to include only the selected samples
    filtered_dataset_df = dataset_df.iloc[selected_indices].reset_index(drop=True)
    
    # Process the documents (will write to output file during processing)
    processed_results_df = process_documents(filtered_dataset_df, args, client, retrieval_data)
    
    # No need to save results here as they are already saved during processing
    if processed_results_df is not None:
        print(f"All results have been saved to '{args.output_file}'")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()