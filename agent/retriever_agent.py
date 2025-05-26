from autogen import Agent
import os
import json
from modules.step02_page_retrieval import (
    query_llm_for_page_retrieval,
    extract_relevant_pages,
    extract_document_summary
)
from utils.openai_helper import initialize_client


class RetrieverAgent(Agent):
    def __init__(self, args, **kwargs):
        super().__init__(name="RetrieverAgent", **kwargs)
        self.args = args
        self.client = initialize_client(args, base_url=args.base_url_retrieval, model_name=args.retrieval_model)

    def generate_response(self, messages, sender, config):
        message = messages[-1]["content"]
        doc_id = message["doc_id"]
        question = message["question"]

        summary_path = os.path.join(self.args.summaries_dir, f"{doc_id}.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        page_summaries = summary_data.get("pages", {})

        llm_response = query_llm_for_page_retrieval(
            question,
            page_summaries,
            self.args.retrieval_model,
            self.client,
            self.args.max_tokens_retrieval,
            self.args.retrieval_prompt_file
        )

        if llm_response is None or not hasattr(llm_response, "content"):
            return {
                "doc_id": doc_id,
                "question": question,
                "relevant_pages": [],
                "document_summary": "",
                "error": "LLM failed"
            }

        relevant_pages = extract_relevant_pages(llm_response.content)
        document_summary = extract_document_summary(llm_response.content)

        return {
            "doc_id": doc_id,
            "question": question,
            "relevant_pages": relevant_pages,
            "document_summary": document_summary
        }
