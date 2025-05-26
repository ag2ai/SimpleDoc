from autogen import GroupChat, GroupChatManager
from agents.retriever_agent import RetrieverAgent
from agents.reasoning_agent import ReasoningAgent


def create_groupchat(args):
    """
    Initializes a GroupChat with RetrieverAgent and ReasoningAgent.
    """
    retriever = RetrieverAgent(args)
    reasoner = ReasoningAgent(args)

    groupchat = GroupChat(
        agents=[retriever, reasoner],
        messages=[],
        max_round=args.max_iter,
        allow_repeat_user=False
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=False  # no default config since agents already use args
    )
    return manager
