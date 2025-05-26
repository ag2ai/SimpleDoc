from agents.retriever_agent import RetrieverAgent
from agents.reasoning_agent import ReasoningAgent


class SimpleDocAgent:
    def __init__(self, args):
        self.args = args
        self.retriever = RetrieverAgent(args)
        self.reasoner = ReasoningAgent(args)

    def run(self, sample):
        # Step 1: Page Retrieval
        retrieval_result = self.retriever.generate_response(
            messages=[{"content": sample}], sender=None, config={}
        )

        # Step 2: QA Reasoning
        reasoning_input = {**sample, **retrieval_result}
        reasoning_result = self.reasoner.generate_response(
            messages=[{"content": reasoning_input}], sender=None, config={}
        )

        # Final output
        return {
            **retrieval_result,
            **reasoning_result
        }
