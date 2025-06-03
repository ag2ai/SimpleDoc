from autogen import Agent, ConversableAgent
import os
import json
import torch
from modules.step02_page_retrieval import (
    query_llm_for_page_retrieval,
    extract_relevant_pages,
    extract_document_summary
)
from utils.openai_helper import initialize_client
from os.path import join
import ast

# Only import if embedding retrieval is enabled
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class RetrieverAgent(ConversableAgent):
    def __init__(self, args, **kwargs):
        super().__init__(name="RetrieverAgent", **kwargs)
        self.args = args
        self.client = initialize_client(args, base_url=args.base_url_retrieval, model_name=args.retrieval_model)

        # Set up embedding-based retrieval
        if getattr(args, "use_embedding_based_retrieval", False):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.processor_emb = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
            self.model_emb = ColQwen2_5.from_pretrained(
                "vidore/colqwen2.5-v0.2", torch_dtype=torch.bfloat16
            ).to(self.device).eval()

        self.register_reply(trigger=[Agent, None], reply_func=custom_generate_reply, position=0)

def custom_generate_reply(agent, messages, sender, config):
    message = json.loads(messages[-1]["content"])
    print(type(message))
    print(message)
    doc_id = message["doc_id"]
    question = message["question"]

    summary_path = os.path.join(agent.args.summaries_dir, f"{doc_id}.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    page_summaries = summary_data.get("pages", {})
    original_summaries = page_summaries.copy()

    if getattr(agent.args, "use_embedding_based_retrieval", False):
        page_embeddings = torch.load(join(
            agent.args.data_base_path.replace("documents", "embeddings"),
            f"{doc_id.replace('.pdf', '.pt')}"
        ), map_location=agent.device)
        page_keys = list(range(1, len(page_embeddings) + 1))
        batch_queries = agent.processor_emb.process_queries([question]).to(agent.device)
        with torch.no_grad():
            query_embeddings = agent.model_emb(**batch_queries)
        scores = agent.processor_emb.score_multi_vector(query_embeddings, page_embeddings)
        scores = scores.squeeze(0).tolist()
        top_pages = [p for p, _ in sorted(zip(page_keys, scores), key=lambda x: x[1], reverse=True)[:agent.args.max_page_retrieval]]
        page_summaries = {str(p): original_summaries[str(p)] for p in top_pages}

    llm_response = query_llm_for_page_retrieval(
        question,
        page_summaries,
        agent.args.retrieval_model,
        agent.client,
        agent.args.max_tokens_retrieval,
        agent.args.retrieval_prompt_file
    )
    print(llm_response)

    if llm_response is None or not hasattr(llm_response, "content"):
        return True, {
            "doc_id": doc_id,
            "question": question,
            "relevant_pages": [],
            "document_summary": "",
            "error": "LLM failed"
        }

    relevant_pages = extract_relevant_pages(llm_response.content)
    document_summary = extract_document_summary(llm_response.content)

    result = {
        "doc_id": doc_id,
        "question": question,
        "relevant_pages": relevant_pages,
        "document_summary": document_summary
    }

    return True, json.dumps(result)