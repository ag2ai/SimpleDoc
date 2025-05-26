import os
import json
import argparse
from tqdm import tqdm
from types import SimpleNamespace

from pipeline.groupchat_controller import create_groupchat


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run SimpleDoc with AG² GroupChat")
    parser.add_argument("--input_file", type=str, default="data/MMLongBench/samples.json")
    parser.add_argument("--output_file", type=str, default="outputs/simpledoc_chat/results.json")
    parser.add_argument("--summaries_dir", type=str, default="outputs/qwen25_32b/summaries")
    parser.add_argument("--data_base_path", type=str, default="data/MMLongBench/documents")
    parser.add_argument("--retrieval_model", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--qa_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--api_key_file", type=str, default="./deepinfrakey")
    parser.add_argument("--base_url_retrieval", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--base_url_qa", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--retrieval_prompt_file", type=str, default="prompts/page_retrieval_prompt.txt")
    parser.add_argument("--qa_prompt_file", type=str, default="prompts/doc_qa_prompt_v3.5.txt")
    parser.add_argument("--max_tokens_retrieval", type=int, default=32768)
    parser.add_argument("--max_tokens_qa", type=int, default=2048)
    parser.add_argument("--image_dpi", type=int, default=150)
    parser.add_argument("--extract_text", action="store_true", default=True)
    parser.add_argument("--text_only", action="store_true", default=False)
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--max_pages", type=int, default=10)
    parser.add_argument("--max_page_retrieval", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_arguments()
    args = SimpleNamespace(**vars(args))  # allows dot notation access

    # Load dataset
    with open(args.input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Create output dir
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Run each sample through the GroupChat pipeline
    all_results = []
    for sample in tqdm(samples, desc="Running SimpleDoc Chat"):
        manager = create_groupchat(args)
        final_message = manager.initiate_chat(message={"content": sample})
        final_result = final_message.get("content", {})
        all_results.append(final_result)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"All results saved to {args.output_file}")


if __name__ == "__main__":
    main()
