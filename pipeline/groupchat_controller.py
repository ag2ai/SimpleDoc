import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autogen import GroupChat, GroupChatManager
from agent.retriever_agent import RetrieverAgent
from agent.reasoning_agent import ReasoningAgent


def create_groupchat(args):
    """
    Initializes a GroupChat with RetrieverAgent and ReasoningAgent.
    """
    retriever = RetrieverAgent(args)
    reasoner = ReasoningAgent(args)

    groupchat = GroupChat(
        agents=[retriever, reasoner],
        messages=[],
        max_round=args.max_iter
    )

    def custom_selector(last, groupchat):
        if last is None:
            return retriever 
        return reasoner if last == retriever else retriever

    groupchat.select_speaker = custom_selector

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=False  # no default config since agents already use args
    )
    return manager
