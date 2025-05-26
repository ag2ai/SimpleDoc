from autogen import Agent
import os
import json
from modules.step03_target_page_qa import (
    convert_pdf_pages_to_base64_images,
    query_llm_with_images,
    postprocess_answer
)
from utils.openai_helper import initialize_client


class ReasoningAgent(Agent):
    def __init__(self, args, **kwargs):
        super().__init__(name="ReasoningAgent", **kwargs)
        self.args = args
        self.client = initialize_client(args, base_url=args.base_url_qa, model_name=args.qa_model)
        self.client_img = self.client  # fallback if needed

    def generate_response(self, messages, sender, config):
        msg = messages[-1]["content"]
        doc_id = msg["doc_id"]
        question = msg["question"]
        retrieved_pages = msg["relevant_pages"]
        document_summary = msg.get("document_summary", "")

        pdf_path = os.path.join(self.args.data_base_path, doc_id)

        if self.args.extract_text:
            try:
                with open(pdf_path.replace('.pdf', '.txt').replace('/documents/', '/text_doc/'), 'rb') as f:
                    pdf_data = json.load(f)
                page_texts = [str(pdf_data[int(i) - 1]) for i in retrieved_pages]
            except:
                page_texts = ["" for _ in retrieved_pages]
        else:
            page_texts = ["" for _ in retrieved_pages]

        images, _ = convert_pdf_pages_to_base64_images(
            pdf_path, retrieved_pages, self.args.image_dpi, extract_text=False
        )

        llm_response = query_llm_with_images(
            question,
            images,
            self.args.qa_model,
            self.client,
            self.args.max_tokens_qa,
            self.args.qa_prompt_file,
            retrieved_pages,
            page_texts,
            document_summary,
            args=self.args,
            client_img=self.client_img
        )

        if llm_response:
            answer, response_type = postprocess_answer(llm_response)
        else:
            answer, response_type = None, "no_response"

        return {
            "doc_id": doc_id,
            "question": question,
            "final_answer": answer,
            "response_type": response_type
        }
