from dotenv import load_dotenv
load_dotenv()

import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.intents import Intent

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings







print("Initializing semantic memory (FAISS index)…")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "data", "ubuntu_qa_index")

ubuntu_embeddings = HuggingFaceEmbeddings(
 
 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ubuntu_vs = FAISS.load_local(
    INDEX_DIR,
    ubuntu_embeddings,
    allow_dangerous_deserialization=True,
)

print("Semantic memory ready.")


# Creates the LLM (uses your OPENAI_API_KEY from the environment)
llm = ChatOpenAI(model="gpt-4o-mini")  # TODO: experiment with other models


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
                    "You are a telecom provider support assistant. "
                    "You have access to the following Q&A knowledge base snippets. "
                    "Use them when they are relevant, but you may also rely on your own reasoning.\n\n"
                    f"Retrieved knowledge:\n{kb_text}"
                )
            )
            # Prepend retrieved knowledge as a system message
            augmented_messages = [kb_system] + messages

    # Call the model
    response = llm.invoke(augmented_messages)

    # ✅ IMPORTANT: return ONLY the new message; reducer appends it
    return {"messages": [response], "current_flow": "technical_support"}


# Creating a router node based on simple keyword matching

def router_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = msg.content.lower()
            break

    # ✅ Fix: define current_flow from state before using it
    current_flow = state.get("current_flow")

    # ✅ If we're already in a flow, continue it
    if current_flow:
        print(f"[Router] Continuing current flow: {current_flow}")
        return {
            "intent": state.get("intent", Intent.UNKNOWN),
            "route_to": current_flow,
            "current_flow": current_flow,
        }

    #  Otherwise, route using keyword matching - rules to be replaced with a proper intent classifier later)
    if any(w in user_text for w in ["bill", "payment", "invoice", "charge", "charged", "due"]):
        intent = Intent.BILLING
        route_to = "billing"
    elif any(w in user_text for w in ["error", "not working", "issue", "disconnect", "router", "internet", "wifi", "slow"]):
        intent = Intent.TECHNICAL_SUPPORT
        route_to = "technical_support"
    elif any(w in user_text for w in ["hi", "hello", "thanks", "how are you"]):
        intent = Intent.CHITCHAT
        route_to = "chitchat"
    elif any(w in user_text for w in ["human", "agent", "representative", "complaint", "escalate"]):
        intent = Intent.ESCALATION
        route_to = "escalation"
    else:
        intent = Intent.UNKNOWN
        route_to = "fallback"

    return {
        "intent": intent,
        "intent_confidence": 0.6,   # placeholder for now
        "route_to": route_to,
    }

    print(f"[Router] New intent: {intent} → route_to: {route_to}")

    return {
        "intent": intent,
        "intent_confidence": 0.6,
        "route_to": route_to,
        "current_flow": route_to,  # ✅ Start new flow
    }



def billing_node(state: AgentState) -> AgentState:
    msg = AIMessage(
        content=(
            "I can’t access your account billing directly, but I can still help.\n\n"
            "What do you want to check?\n"
            "1) Latest invoice total\n"
            "2) Amount due / balance\n"
            "3) Itemized charges\n\n"
            "Reply with 1/2/3 and tell me which provider portal/app you use, "
            "and I’ll guide you step-by-step."
        )
    )
    return {
        "messages": [msg],
        "needs_handoff": False,
        "current_flow": "billing",  # ✅ Keep conversation in billing flow
    }


def info_lookup_node(state: AgentState) -> AgentState:
    result = call_llm(state)
    result["current_flow"] = "info_lookup"
    return result



def escalation_node(state: AgentState) -> AgentState:
    msg = AIMessage(
        content=(
            "I can help escalate this to customer support.\n\n"
            "Please share:\n"
            "• a 1–2 sentence description of the issue\n"
            "• when it started\n"
            "• service type (mobile/fixed) and your city\n"
            "• anything you already tried\n\n"
            "Then I’ll draft a concise ticket message you can paste to support."
        )
    )
    return {
        "messages": [msg],
        "needs_handoff": True,
        "current_flow": "escalation",
    }



def chitchat_node(state: AgentState) -> AgentState:
    msg = AIMessage(content="Hey! 😊 How can I help today—billing, technical support, or general info?")
    return {
        "messages": [msg],
        "current_flow": "chitchat",
    }



def fallback_node(state: AgentState) -> AgentState:
    msg = AIMessage(
        content=(
            "I can help with:\n"
            "• technical support\n"
            "• billing questions\n"
            "• general information\n\n"
            "Which one do you need?"
        )
    )
    return {
        "messages": [msg],
        "current_flow": "fallback",
    }



# Builds the LangGraph
builder = StateGraph(AgentState)

# Nodes
builder.add_node("router", router_node)
builder.add_node("technical_support", call_llm)
builder.add_node("billing", billing_node)
builder.add_node("info_lookup", info_lookup_node)
builder.add_node("escalation", escalation_node)
builder.add_node("chitchat", chitchat_node)
builder.add_node("fallback", fallback_node)



# Entry point
builder.set_entry_point("router")

# Route from router -> chosen node
builder.add_conditional_edges(
    "router",
    lambda s: s["route_to"],
    {
        "technical_support": "technical_support",
        "billing": "billing",
        "info_lookup": "info_lookup",
        "escalation": "escalation",
        "chitchat": "chitchat",
        "fallback": "fallback",
    },
)


# Specialist -> END
builder.add_edge("technical_support", END)
builder.add_edge("billing", END)
builder.add_edge("info_lookup", END)
builder.add_edge("escalation", END)
builder.add_edge("chitchat", END)
builder.add_edge("fallback", END)


# In-memory checkpointer so the agent remembers the conversation
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)



def main():
    print("Thesis LangGraph agent with FAISS retrieval. Type 'quit' to exit.\n")

    config = {"configurable": {"thread_id": "thesis-demo"}}

    # ✅ Maintain full conversation history
    messages: List[BaseMessage] = []

    while True:
        user_text = input("You: ")
        if user_text.strip().lower() in {"quit", "exit"}:
            print("Thank you for using our service! Feel free to reach out anytime. Goodbye!")
            break

        # ✅ Add new user message to history
        messages.append(HumanMessage(content=user_text))

        # ✅ Invoke graph with full message history
        result = graph.invoke(
            {"messages": messages},
            config=config,
        )

        # ✅ Get and append AI message(s)
        new_agent_messages = result.get("messages", [])
        for msg in new_agent_messages:
            if isinstance(msg, AIMessage):
                print("Agent:", msg.content)
                messages.append(msg)  # append to history


if __name__ == "__main__":
    main()
