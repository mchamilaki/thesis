import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver


# 1. Create the LLM (uses your OPENAI_API_KEY from the environment)
llm = ChatOpenAI(model="gpt-4o-mini")  # you can change the model later


# 2. Define the function that LangGraph will call
def call_llm(state: MessagesState) -> MessagesState:
    """Takes the conversation so far and returns a new AI message."""
    messages: List[BaseMessage] = state["messages"]
    response = llm.invoke(messages)
    # Add the AI response to the conversation
    return {"messages": messages + [response]}


# 3. Build the LangGraph
builder = StateGraph(MessagesState)
builder.add_node("model", call_llm)
builder.set_entry_point("model")
builder.add_edge("model", END)

# Use an in-memory checkpointer so the agent remembers the conversation
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def main():
    print("Thesis LangGraph agent. Type 'quit' to exit.\n")

    # One thread_id = one conversation
    config = {"configurable": {"thread_id": "thesis-demo"}}

    while True:
        user_text = input("You: ")
        if user_text.strip().lower() in {"quit", "exit"}:
            print("Bye!")
            break

        # Add the new user messagepython main.py

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

        # Get the last AI message from the conversation
        last_message = result["messages"][-1]
        print("Agent:", last_message.content)
        print()  # blank line for readability


if __name__ == "__main__":
    main()
