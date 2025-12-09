from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Optional, TypedDict  # ✅ added Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---- Load Ubuntu QA FAISS index ----
print("Loading FAISS index 'ubuntu_qa_index' ...")

ubuntu_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ubuntu_vs = FAISS.load_local(
    "ubuntu_qa_index",
    ubuntu_embeddings,
    allow_dangerous_deserialization=True,  # needed for FAISS load
)

print("FAISS index loaded.")


# DEFINE STATE 
class AgentState(TypedDict):
    messages: List[BaseMessage]
    intent: Optional[str]  # for now, will become Enum later


# Creates the LLM (uses your OPENAI_API_KEY from the environment)
llm = ChatOpenAI(model="gpt-4o-mini")  # you can change the model later


# Defines the function that LangGraph will call
def call_llm(state: AgentState) -> AgentState:
    """Takes the conversation so far, retrieves knowledge, and returns a new AI message."""
    messages: List[BaseMessage] = state["messages"]

    # Find the most recent user message (if any)
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    augmented_messages: List[BaseMessage] = list(messages)

    if user_query:
        # Use FAISS index to retrieve relevant Q&A
        docs = ubuntu_vs.similarity_search(user_query, k=5)

        if docs:
            kb_text = "\n\n".join(d.page_content for d in docs)
            kb_system = SystemMessage(
                content=(
                    "You are a telecom / Ubuntu support assistant. "
                    "You have access to the following Q&A knowledge base snippets. "
                    "Use them when they are relevant, but you may also rely on your own reasoning.\n\n"
                    f"Retrieved knowledge:\n{kb_text}"
                )
            )
            # Prepend retrieved knowledge as a system message
            augmented_messages = [kb_system] + messages

    # Call the model with the augmented messages (or original if no query)
    response = llm.invoke(augmented_messages)

    # Create a new state based on the old one, but with updated messages
    new_state: AgentState = dict(state)  # copy the existing state (including intent)
    new_state["messages"] = messages + [response]
    return new_state


# Builds the LangGraph
builder = StateGraph(AgentState)
builder.add_node("model", call_llm)
builder.set_entry_point("model")
builder.add_edge("model", END)

# Uses an in-memory checkpointer so the agent remembers the conversation
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def main():
    print("Thesis LangGraph agent with FAISS retrieval. Type 'quit' to exit.\n")

    # One thread_id = one conversation
    config = {"configurable": {"thread_id": "thesis-demo"}}

    while True:
        user_text = input("You: ")
        if user_text.strip().lower() in {"quit", "exit"}:
            print("Bye!")
            break

        result = graph.invoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "intent": None,  #  provides initial intent
            },
            config=config,
        )

        # Gets the last AI message from the conversation
        last_message = result["messages"][-1]
        print("Agent:", last_message.content)
        print() 


if __name__ == "__main__":
    main()
