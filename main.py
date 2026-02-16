from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.state import AgentState
from src.intents import Intent

from langchain_community.vectorstores import FAISS #vector search library, fast similarity search 
from langchain_huggingface import HuggingFaceEmbeddings

from src.tools.billing_tools import fetch_invoice

from datetime import datetime, timezone

def trace(state: Dict[str, Any], node: str, **extra) -> None:
    state.setdefault("debug", []).append(
        {"ts": datetime.now(timezone.utc).isoformat(), "node": node, **extra}
    )





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

billing_llm = llm.bind_tools([fetch_invoice])

tools = [fetch_invoice]
tool_node = ToolNode(tools)


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

    # IMPORTANT: return ONLY the new message; reducer appends it
    trace(state, "technical_support_llm", used_kb=bool(user_query), tool_calls=bool(getattr(response, "tool_calls", None)))
    return {"messages": [response]}




# Creating a router node based on simple keyword matching

def router_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = msg.content.lower()
            break

    # OVERRIDE: user asks for human at any time → escalate immediately
    if any(w in user_text for w in ["human", "agent", "representative", "complaint", "escalate"]):
        trace(state, "router", mode="override_escalation")
        return {
            "intent": Intent.ESCALATION,
            "intent_confidence": 1.0,
            "route_to": "escalation",
            "current_flow": "escalation",
        }        



    # Define current_flow from state before using it
    current_flow = state.get("current_flow")

    # Continue ONLY multi-turn flows
    current_flow = state.get("current_flow")
    if current_flow in {"billing"}:
        trace(state, "router", mode="continue_flow", current_flow=current_flow)
        return {
            "intent": state.get("intent", Intent.UNKNOWN),
            "intent_confidence": state.get("intent_confidence", 0.6),
            "route_to": current_flow,
            "current_flow": current_flow,
        }


    #  Otherwise, route using keyword matching - rules to be replaced with a proper intent classifier later)
    if any(w in user_text for w in ["bill", "payment", "invoice", "charge", "charged", "due"]):
        intent = Intent.BILLING
        route_to = "billing"
    elif any(w in user_text for w in ["error","not working","issue","disconnect","router","internet","wifi","slow","connection","down","offline","outage","no internet"]):
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
    new_flow = route_to if route_to in {"billing", "escalation"} else None

    trace(state, "router", mode="new_intent", intent=str(intent), route_to=route_to, current_flow=new_flow)
    print(f"[Router] New intent: {intent} → route_to: {route_to}")

    return {
        "intent": intent,
        "intent_confidence": 0.6,
        "route_to": route_to,
        "current_flow": new_flow,
    }






def billing_node(state: AgentState) -> AgentState:
    messages = state["messages"]

    # Get latest user message
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = msg.content.strip().lower()
            break

    # Handle menu choices (1 / 2 / 3) for billing flow
    if user_text == "1":
        reply = (
            "Latest invoice total selected.\n\n"
            "Open your provider app or website and go to:\n"
            "Billing → Invoices → Latest invoice → Total amount.\n\n"
            "If you want, tell me which provider you use and I can give exact steps."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "current_flow": None,  # billing flow completed
        }

    if user_text == "2":
        reply = (
            "Amount due / balance selected.\n\n"
            "Usually found under:\n"
            "Billing → Balance / Amount due → Due date.\n\n"
            "Is this a mobile or a fixed service?"
        )
        return {
            "messages": [AIMessage(content=reply)],
            "current_flow": None,
        }

    if user_text == "3":
        reply = (
            "Itemized charges selected.\n\n"
            "Go to:\n"
            "Billing → Invoice details → Breakdown / Usage / Charges.\n\n"
            "If something looks wrong, I can help you prepare a support message."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "current_flow": None,
        }

    # Otherwise, show the menu (first turn)
    msg = AIMessage(
        content=(
            "I can’t access your account billing directly, but I can still help.\n\n"
            "What do you want to check?\n"
            "1) Latest invoice total\n"
            "2) Amount due / balance\n"
            "3) Itemized charges\n\n"
            "Reply with 1 / 2 / 3."
        )
    )
    return {
        "messages": [msg],
        "current_flow": "billing",  # keep flow open
    }


def billing_llm_node(state: AgentState) -> AgentState:
    messages: List[BaseMessage] = state["messages"]
    response = billing_llm.invoke(messages)

    trace(
        state,
        "billing_llm",
        has_tool_calls=bool(getattr(response, "tool_calls", None)),
    )
    return {"messages": [response], "current_flow": "billing"}


def billing_should_use_tools(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "done"


def info_lookup_node(state: AgentState) -> AgentState:
    result = call_llm(state)
    result["current_flow"] = None
    return result

def escalation_node(state: AgentState) -> AgentState:
    msg = AIMessage(
        content=(
            "Thanks — please wait while I connect you to one of our human agents. "
            "You’ll be transferred shortly."
        )
    )
    trace(state, "handoff", needs_handoff=True)
    return {
        "messages": [msg],
        "needs_handoff": True,
        "current_flow": None,
    }



def chitchat_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "You are a friendly telecom customer service assistant. "
            "Engage naturally in small talk while remaining professional."
        )
    )

    response = llm.invoke([system_prompt] + state["messages"])

    trace(state, "chitchat_llm")

    return {
        "messages": [response],
        "current_flow": None,
    }




def fallback_node(state: AgentState) -> AgentState:
    system_guardrail = SystemMessage(
        content=(
            "You are a telecom customer service assistant. "
            "The user's request did not clearly match billing, technical support, "
            "or escalation. Respond conversationally, clarify their need, "
            "and gently guide them toward supported topics if necessary."
        )
    )

    augmented_messages = [system_guardrail] + state["messages"]

    response = llm.invoke(augmented_messages)

    trace(state, "fallback_llm")

    return {
        "messages": [response],
        "current_flow": None,
    }





# Builds the LangGraph
builder = StateGraph(AgentState)

# Nodes
builder.add_node("router", router_node)
builder.add_node("technical_support", call_llm)
builder.add_node("billing", billing_llm_node)
builder.add_node("info_lookup", info_lookup_node)
builder.add_node("escalation", escalation_node)
builder.add_node("chitchat", chitchat_node)
builder.add_node("fallback", fallback_node)
builder.add_node("tools", tool_node)




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

#Agentic tools flow for billing
builder.add_conditional_edges(
    "billing",
    billing_should_use_tools,
    {"tools": "tools", "done": END},
)
builder.add_edge("tools", "billing")



# All other specialist nodes end normally
builder.add_edge("technical_support", END)
builder.add_edge("info_lookup", END)
builder.add_edge("chitchat", END)
builder.add_edge("fallback", END)

# Escalation only when router routes to it
builder.add_edge("escalation", END)

memory=MemorySaver()  # Saves all states to an in-memory list, can be replaced with a database saver for production
graph = builder.compile(checkpointer=memory)



def main():
    print("Thesis LangGraph agent with FAISS retrieval. Type 'quit' to exit.\n")
    config = {"configurable": {"thread_id": "thesis-demo"}}

    while True:
        user_text = input("Type your message here: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            print("Thank you for using our service! Feel free to reach out anytime. Goodbye!")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

        last_ai = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None,
        )

        if last_ai:
            print("Agent:", last_ai.content)
            print()



   

if __name__ == "__main__":
    main()
