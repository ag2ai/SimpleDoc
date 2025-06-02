from autogen import Agent, ConversableAgent
import os
import json
from modules.step03_target_page_qa import (
    convert_pdf_pages_to_base64_images,
    query_llm_with_images,
    postprocess_answer
)
from utils.openai_helper import initialize_client


class ReasoningAgent(ConversableAgent):
    def __init__(self, args, **kwargs):
        super().__init__(name="ReasoningAgent", **kwargs)
        self.args = args
        # print(f"[ReasoningAgent] Using QA model: {args.qa_model}")
        self.client = initialize_client(args, base_url=args.base_url_qa, model_name=args.qa_model)
        self.client_img = self.client  # fallback if needed
        self.register_reply(trigger=[Agent, None], reply_func=custom_generate_reply, position=0)

    # def generate_response(self, messages, sender, config):
    #     print(f"[ReasoningAgent] Triggered for question: {question}")
    #     msg = json.loads(messages[-1]["content"])
    #     doc_id = msg["doc_id"]
    #     question = msg["question"]
    #     retrieved_pages = msg["relevant_pages"]
    #     document_summary = msg.get("document_summary", "")

    #     pdf_path = os.path.join(self.args.data_base_path, doc_id)
        
    #     if self.args.extract_text:
    #         try:
    #             with open(pdf_path.replace('.pdf', '.txt').replace('/documents/', '/text_doc/'), 'rb') as f:
    #                 pdf_data = json.load(f)
    #             page_texts = [str(pdf_data[int(i) - 1]) for i in retrieved_pages]
    #         except:
    #             page_texts = ["" for _ in retrieved_pages]
    #     else:
    #         page_texts = ["" for _ in retrieved_pages]

    #     images, _ = convert_pdf_pages_to_base64_images(
    #         pdf_path, retrieved_pages, self.args.image_dpi, extract_text=False
    #     )
    #     print(f"[ReasoningAgent] Retrieved {len(images)} images for doc_id: {doc_id}")

    #     print(f"[ReasoningAgent] Sending query to model: {self.args.qa_model}")
    #     llm_response = query_llm_with_images(
    #         question,
    #         images,
    #         self.args.qa_model,
    #         self.client,
    #         self.args.max_tokens_qa,
    #         self.args.qa_prompt_file,
    #         retrieved_pages,
    #         page_texts,
    #         document_summary,
    #         args=self.args,
    #         client_img=self.client_img
    #     )

    #     if llm_response:
    #         answer, response_type = postprocess_answer(llm_response)
    #     else:
    #         answer, response_type = None, "no_response"

    #     return {
    #         "doc_id": doc_id,
    #         "question": question,
    #         "final_answer": answer,
    #         "response_type": response_type
    #     }


def custom_generate_reply(agent, messages, sender, config):
    msg = json.loads(messages[-1]["content"])
    question = msg["question"]
    doc_id = msg["doc_id"]
    retrieved_pages = msg["relevant_pages"]
    document_summary = msg.get("document_summary", "")

    print(f"[ReasoningAgent] Triggered for question: {question}")

    pdf_path = os.path.join(agent.args.data_base_path, doc_id)

    if agent.args.extract_text:
        try:
            with open(pdf_path.replace('.pdf', '.txt').replace('/documents/', '/text_doc/'), 'rb') as f:
                pdf_data = json.load(f)
            page_texts = [str(pdf_data[int(i) - 1]) for i in retrieved_pages]
        except:
            page_texts = ["" for _ in retrieved_pages]
    else:
        page_texts = ["" for _ in retrieved_pages]

    images, _ = convert_pdf_pages_to_base64_images(
        pdf_path, retrieved_pages, agent.args.image_dpi, extract_text=False
    )
    print(f"[ReasoningAgent] Retrieved {len(images)} images for doc_id: {doc_id}")

    print(f"[ReasoningAgent] Sending query to model: {agent.args.qa_model}")
    llm_response = query_llm_with_images(
        question,
        images,
        agent.args.qa_model,
        agent.client,
        agent.args.max_tokens_qa,
        agent.args.qa_prompt_file,
        retrieved_pages,
        page_texts,
        document_summary,
        args=agent.args,
        client_img=agent.client_img
    )

    if llm_response:
        answer, response_type = postprocess_answer(llm_response)
    else:
        answer, response_type = None, "no_response"

    result = {
        "doc_id": doc_id,
        "question": question,
        "final_answer": answer,
        "response_type": response_type
    }

    return True, json.dumps(result)