#!/usr/bin/env python3
"""
Integrated pipeline: for each sample, perform page retrieval and question answering,
with iterative query updates based on <query_update> tags.
"""
import os
import json
import argparse
import random
import logging
from utils.openai_helper import initialize_client
from step02_page_retrieval import query_llm_for_page_retrieval, extract_relevant_pages, extract_document_summary
from step03_target_page_qa import convert_pdf_pages_to_base64_images, query_llm_with_images, postprocess_answer
from joblib import Parallel, delayed
from tqdm import tqdm
from os.path import join

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Retrieval+QA pipeline with iterative query updates")
    parser.add_argument("--input_file", type=str,
                        default="data/MMLongBench/samples.json",
                        help="Input JSON file with samples (list of {doc_id, question, ...})")
    parser.add_argument("--summaries_dir", type=str,
                        default="outputs/qwen25_32b/summaries",
                        help="Directory containing page summary JSON files from step01")
    parser.add_argument("--data_base_path", type=str,
                        default="data/MMLongBench/documents",
                        help="Base directory for PDF documents")
    parser.add_argument("--output_file", type=str,
                        default="outputs/docr1/v1_results.json",
                        help="Output JSON file for pipeline results")
    parser.add_argument("--retrieval_model", type=str,
                        default="Qwen/Qwen3-30B-A3B",
                        help="Model name for page retrieval")
    parser.add_argument("--qa_model", type=str,
                        default="Qwen/Qwen2.5-VL-32B-Instruct",
                        help="Model name for document QA")
    parser.add_argument("--api_key_file", type=str, default="./deepinfrakey",
                        help="File containing the API key")
    parser.add_argument("--base_url_retrieval", type=str,
                        default="http://dgxh-4.hpc.engr.oregonstate.edu:8000/v1",
                        help="Base URL for the LLM API")
    parser.add_argument("--base_url_qa", type=str,
                        default="http://cn-w-1.hpc.engr.oregonstate.edu:8000/v1",
                        help="Base URL for the LLM API")
    parser.add_argument("--cache_seed", type=int, default=123,
                        help="Cache seed for LLM client")
    parser.add_argument("--retrieval_prompt_file", type=str,
                        default="prompts/page_retrieval_prompt.txt",
                        help="Prompt template for page retrieval")
    parser.add_argument("--qa_prompt_file", type=str,
                        default="prompts/doc_qa_prompt_v3.5.txt",
                        help="Prompt template for document QA")
    parser.add_argument("--max_tokens_retrieval", type=int, default=32768,
                        help="Max tokens for retrieval LLM call")
    parser.add_argument("--max_tokens_qa", type=int, default=2048,
                        help="Max tokens for QA LLM call")
    parser.add_argument("--image_dpi", type=int, default=150,
                        help="DPI for rendering PDF pages to images")
    parser.add_argument("--extract_text", action="store_true", default=True,
                        help="Whether to extract page text for QA prompts")
    parser.add_argument("--text_only", action="store_true", default=False)
    parser.add_argument("--max_iter", type=int, default=3,
                        help="Maximum retrieval+QA iterations per sample")
    parser.add_argument("--max_pages", type=int, default=10,
                        help="Maximum number of pages to pass to the LLM. Default is -1 (use all retrieved pages).")
    parser.add_argument("--max_page_retrieval", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs for sample processing")
    parser.add_argument("--use_page_augmentation", action="store_true",
                        help="Whether to use page augmentation", default=False)
    parser.add_argument("--use_embedding_based_retrieval", action="store_true",
                        help="Whether to use embedding-based retrieval", default=False)
    return parser.parse_args()


def subsample_half_preserve_order(original_list):
    """
    Randomly subsample a list to half of its original size while preserving the original order.

    Args:
        original_list: The list to subsample

    Returns:
        A new list containing a random half of the elements from the original list in their original order
    """

    # Set the seed for reproducibility
    random.seed(123)

    # Calculate half the size of the list (using integer division)
    half_size = len(original_list) // 2

    # Get all indices of the original list
    all_indices = list(range(len(original_list)))

    # Randomly select half of the indices
    selected_indices = sorted(random.sample(all_indices, half_size))

    # Create a new list with elements at the selected indices (preserving order)
    subsampled_list = [original_list[i] for i in selected_indices]

    return subsampled_list


def page_augmentation(pages, total_num_pages, num_augment_page=1):
    """
    Augment the pages by adding the next `num_augment_page` pages to each page.
    The order of the original pages is preserved in the output, with augmented
    pages inserted after their respective original page. Duplicates are removed
    based on first appearance.
    The pages is a list of page numbers.
    The total number of pages is `total_num_pages`.

    Args:
        pages: List of page numbers.
        total_num_pages: Total number of pages in the document.
        num_augment_page: Number of pages to augment each page with.
    Returns:
        A list of page numbers with augmentations, preserving original page order
        and removing duplicates based on first appearance.
    """
    if not pages:
        return []

    interim_pages_with_duplicates = []
    for page in pages:
        # Add the original page
        if 1 <= page <= total_num_pages: # Ensure original page is valid
            interim_pages_with_duplicates.append(page)
        
        # Add augmented pages
        for i in range(1, num_augment_page + 1):
            next_page = page + i
            if 1 <= next_page <= total_num_pages:
                interim_pages_with_duplicates.append(next_page)
            else:
                # Stop augmenting for this page if we go out of bounds
                break
    
    # Remove duplicates while preserving order of first appearance
    final_ordered_pages = []
    seen_pages = set()
    for p in interim_pages_with_duplicates:
        if p not in seen_pages:
            final_ordered_pages.append(p)
            seen_pages.add(p)
            
    return final_ordered_pages


def process_sample(sample, args, device=None, processor_emb=None, model_emb=None):
    """Process a single sample with iterative retrieval and QA."""

    # Initialize LLM clients for retrieval and QA
    client_r = initialize_client(args, base_url=args.base_url_retrieval, model_name=args.retrieval_model)
    client_q = initialize_client(args, base_url=args.base_url_qa, model_name=args.qa_model)
    client_img = initialize_client(args, base_url="http://cn-w-1.hpc.engr.oregonstate.edu:8000/v1", model_name="Qwen/Qwen2.5-VL-32B-Instruct")

    doc_id = sample.get('doc_id')
    orig_q = sample.get('question')
    # Load page summaries
    summary_path = os.path.join(args.summaries_dir, f"{doc_id}.json")
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    original_pages_summary = summary_data.get('pages', {})
    pages_summary = original_pages_summary.copy()

    question = orig_q
    iteration_records = []
    notes = ""
    if args.use_embedding_based_retrieval:
        page_embeddings = torch.load(join(args.data_base_path.replace("documents", "embeddings"),
                                          f"{doc_id.replace(".pdf", ".pt")}"), map_location=device)
        page_keys = list(range(1, len(page_embeddings) + 1))

    for it in range(1, args.max_iter + 1):
        if args.use_embedding_based_retrieval:
            # Restore original pages_summary and filter by embedding similarity
            pages_summary = original_pages_summary.copy()

            # Compute query embedding
            batch_queries = processor_emb.process_queries([question]).to(device)
            with torch.no_grad():
                query_embeddings = model_emb(**batch_queries)
            # Compute similarity scores
            scores = processor_emb.score_multi_vector(query_embeddings, page_embeddings)
            scores = scores.squeeze(0).tolist()
            # Select top-k pages
            scores_pages = list(zip(page_keys, scores))
            scores_pages_sorted = sorted(scores_pages, key=lambda x: x[1], reverse=True)
            top_pages = [p for p,_ in scores_pages_sorted[:args.max_page_retrieval]]
            # Filter pages_summary to top scoring pages
            pages_summary = {str(p): pages_summary[str(p)] for p in top_pages}

        is_over_long_page = False
        while True:
            message = query_llm_for_page_retrieval(
                question,
                pages_summary,
                args.retrieval_model,
                client_r,
                args.max_tokens_retrieval,
                args.retrieval_prompt_file
            )
            if message is not None:
                if message.model_extra.get("reasoning_content", None) is not None and message.content is None:
                    message.content = message.model_extra["reasoning_content"]
                break

            is_over_long_page = True
            pages_summary = dict(subsample_half_preserve_order(list(pages_summary.items())))
            print(f"Subsampling pages for {doc_id}: {len(pages_summary)} pages because of API error")

        assert message.content is not None, f"Retrieval failed for {doc_id} with question: {question}. Reasoning: {message.model_extra["reasoning_content"]}"

        pages = extract_relevant_pages(message.content)
        # filter pages to be in the range of 1 to total_num_pages
        total_num_pages = len(original_pages_summary)
        pages = [p for p in pages if 1 <= p <= total_num_pages]

        doc_sum = extract_document_summary(message.content)

        if args.use_page_augmentation:
            pages = page_augmentation(pages, total_num_pages=len(pages_summary), num_augment_page=1)

        # Limit number of pages if max_pages is set
        if 0 < args.max_pages < len(pages):
            pages = pages[:args.max_pages]

        # QA step
        pdf_path = os.path.join(args.data_base_path, doc_id)

        images, _ = convert_pdf_pages_to_base64_images(
            pdf_path, pages, args.image_dpi, extract_text=False
        )
        if args.extract_text:
            with open(pdf_path.replace('.pdf', '.txt').replace('/documents/', '/text_doc/'), 'rb') as f:
                pdf_data = json.load(f)
                texts = [str(pdf_data[int(i)-1]) for i in pages]
        else:
            texts = ""

        qa_resp = query_llm_with_images(
            question,
            images,
            args.qa_model,
            client_q,
            args.max_tokens_qa,
            args.qa_prompt_file,
            pages,
            texts,
            f"{doc_sum}\n\n{notes}",
            args=args,
            client_img=client_img,
        )
        answer, resp_type = postprocess_answer(qa_resp or '')
        if resp_type == 'query_update':
            answer, notes = answer[0], answer[1]

        iteration_records.append({
            'iteration': it,
            'query': question,
            'retrieval_response': message.content,
            'pages': pages,
            'document_summary': doc_sum,
            'qa_response': qa_resp,
            'answer': answer,
            'notes': notes,
            'response_type': resp_type,
            'is_over_long_page': is_over_long_page
        })
        if resp_type == 'query_update':
            question = answer
            continue
        break

    # if resp_type == 'query_update':
    #     iteration_records[-1]["answer"] = "The document does not contain the information needed to answer this question."
    #     iteration_records[-1]["response_type"] = "not_answerable"

    final = iteration_records[-1] if iteration_records else {}
    return {
        'doc_id': doc_id,
        'original_question': orig_q,
        'iterations': iteration_records,
        'final_answer': final.get('answer'),
        'final_response_type': final.get('response_type'),
        'is_over_long_page': final.get('is_over_long_page')
    }


def main():
    args = parse_arguments()

    # Load samples
    with open(args.input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if args.use_embedding_based_retrieval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor_emb = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
        model_emb = ColQwen2_5.from_pretrained("vidore/colqwen2.5-v0.2", torch_dtype=torch.bfloat16).to(device).eval()

    # Process samples in parallel if n_jobs > 1 else sequentially
    if args.n_jobs == 1:
        results = [process_sample(sample, args, device, processor_emb, model_emb) for sample in tqdm(samples)] # For debugging
    else:
        results = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(process_sample)(sample, args, device, processor_emb, model_emb) for sample in tqdm(samples)
        )

    # Save results
    out_dir = os.path.dirname(args.output_file) or '.'
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Pipeline completed, results saved to {args.output_file}")

if __name__ == '__main__':
    main()