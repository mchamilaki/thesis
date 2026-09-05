print("RUNNING FILE:", __file__)

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.state import AgentState
from src.intents import Intent



from src.tools.billing_tools import fetch_invoice
from src.tools.retrieval_tools import search_knowledge_base


def trace(state: Dict[str, Any], node: str, **extra) -> None:
    state.setdefault("debug", []).append(
        {"ts": datetime.now(timezone.utc).isoformat(), "node": node, **extra}
    )





# Creates the LLM (uses your OPENAI_API_KEY from the environment)
llm = ChatOpenAI(model="gpt-4o-mini",temperature=1.8)  
# The default temperature is 0.0 for deterministic behavior; it will be overridden in the sweep.
#For evaluation, we will sweep over temperatures 0.0, 0.3, and 0.7 to see how the router's performance changes with different levels of randomness in its responses.

# Billing agent: LLM with the invoice tool bound
billing_llm = llm.bind_tools([fetch_invoice])

# Technical support agent: LLM with the knowledge-base search tool bound
tech_llm = llm.bind_tools([search_knowledge_base])

TECH_SYSTEM = SystemMessage(content="""
You are a telecom technical support assistant.
For any technical question or problem, ALWAYS search the knowledge base before
answering, using search_knowledge_base with a concise technical query you
formulate yourself. You may search more than once if the user describes multiple
issues or the first results are not relevant.
Only skip searching when the user is merely clarifying or acknowledging
(e.g. "thanks", "how long will that take?").
Base your answer on the retrieved results when they are relevant; if they are
not, say so honestly and give your best general guidance.
""")



class IntentClassification(BaseModel):
    """Structured output schema for the router."""
    intent: Literal[
        "billing", "technical_support", "chitchat",
        "escalation", "info_lookup", "unknown"
    ] = Field(description="The user's intent category")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the classification, 0.0 to 1.0"
    )


ROUTER_SYSTEM = """You are an intent classifier for a telecom customer service chatbot.
Classify the user's message into exactly one category:

- billing: invoices, payments, charges, amounts due
- technical_support: internet/connection problems, errors, outages, device issues
- chitchat: greetings, thanks, small talk
- escalation: explicit requests for a human agent, complaints
- info_lookup: questions about plans, prices, packages, offers, roaming
- unknown: anything that doesn't clearly fit the above

Consider the recent conversation context when the latest message is ambiguous
(e.g. "yes", "the second one", an account number)."""

# Separate LLM instance for routing: T=0 for deterministic classification
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(
    IntentClassification
)


# System Prompts 

BILLING_SYSTEM = SystemMessage(content="""
You are a telecom billing assistant.
If the user hasn't provided their account ID, ask for it before doing anything.
Once you have it, call fetch_invoice to retrieve their billing details.
After receiving the result, explain the status clearly and helpfully.
If the account is not found, apologize and suggest they double-check their ID.
Amounts are in euros (EUR); format them with the € symbol, e.g. €45.50.
""")


#NODES

# Creating agentic nodes for each flow and its routing logic
def tech_support_node(state: AgentState) -> AgentState:
    """Agentic technical support: decides when/what to search, then answers."""
    messages: List[BaseMessage] = state["messages"]
    response = tech_llm.invoke([TECH_SYSTEM] + messages)

    trace(
        state,
        "tech_support_llm",
        has_tool_calls=bool(getattr(response, "tool_calls", None)),
    )
    return {"messages": [response], "current_flow": None}


def tech_should_use_tools(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "done"



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
            "current_flow": "None",
        }        


    
    
    # LLM-based intent classification (replaces keyword matching)
    # Include recent context so follow-ups like "yes" or "123" classify correctly
    recent = messages[-6:]
    convo_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in recent
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    )

    try:
        result = router_llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Conversation:\n{convo_text}\n\nClassify the latest user message."),
        ])
        intent_str = result.intent
        confidence = result.confidence
    except Exception as e:
        # Graceful degradation: if classification fails, fall back instead of crashing
        trace(state, "router", mode="llm_error", error=str(e))
        intent_str, confidence = "unknown", 0.0

    intent_map = {
        "billing": (Intent.BILLING, "billing"),
        "technical_support": (Intent.TECHNICAL_SUPPORT, "technical_support"),
        "chitchat": (Intent.CHITCHAT, "chitchat"),
        "escalation": (Intent.ESCALATION, "escalation"),
        "info_lookup": (Intent.INFO_LOOKUP, "info_lookup"),
        "unknown": (Intent.UNKNOWN, "fallback"),
    }
    intent, route_to = intent_map[intent_str]

    # Low-confidence classifications go to fallback for clarification
    if confidence < 0.5 and route_to != "escalation":
        route_to = "fallback"

    new_flow = route_to if route_to in {"billing", "escalation"} else None

    trace(state, "router", mode="llm_classification", intent=intent_str, confidence=confidence)

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "route_to": route_to,
        "current_flow": new_flow,
    }    

    
def billing_llm_node(state: AgentState) -> AgentState:
    """Agentic billing node: asks for account ID if needed, calls fetch_invoice, explains result."""
    messages: List[BaseMessage] = state["messages"]

    # Prepend the billing system prompt on every invocation
    augmented_messages = [BILLING_SYSTEM] + messages

    response = billing_llm.invoke(augmented_messages)

    trace(
        state,
        "billing_llm",
        has_tool_calls=bool(getattr(response, "tool_calls", None)),
    )
    return {"messages": [response], "current_flow": "None"}




def billing_should_use_tools(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "done"


def info_lookup_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "You are a telecom customer service assistant answering questions about "
            "plans, pricing, packages, and services. Answer helpfully and concisely."
        )
    )
    response = llm.invoke([system_prompt] + state["messages"])
    trace(state, "info_lookup_llm")
    return {"messages": [response], "current_flow": None}

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

# Tool nodes — one per agentic loop
billing_tool_node = ToolNode([fetch_invoice])            
tech_tool_node = ToolNode([search_knowledge_base])        

# Nodes
builder.add_node("router", router_node)
builder.add_node("technical_support", tech_support_node)
builder.add_node("billing", billing_llm_node)
builder.add_node("info_lookup", info_lookup_node)
builder.add_node("escalation", escalation_node)
builder.add_node("chitchat", chitchat_node)
builder.add_node("fallback", fallback_node)
builder.add_node("billing_tools", billing_tool_node)
builder.add_node("tech_tools", tech_tool_node)




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

# Billing agentic loop
builder.add_conditional_edges(
    "billing",
    billing_should_use_tools,
    {"tools": "billing_tools", "done": END},
)
builder.add_edge("billing_tools", "billing")

# Technical support agentic loop
builder.add_conditional_edges(
    "technical_support",
    tech_should_use_tools,
    {"tools": "tech_tools", "done": END},
)
builder.add_edge("tech_tools", "technical_support")



# All other specialist nodes end normally

builder.add_edge("info_lookup", END)
builder.add_edge("chitchat", END)
builder.add_edge("fallback", END)

# Escalation only when router routes to it
builder.add_edge("escalation", END)

memory=MemorySaver()  # Saves all states to an in-memory list, can be replaced with a database saver for production
graph = builder.compile(checkpointer=memory)

def get_graph():
    """Export the compiled graph for use in app.py."""
    return graph


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
    from langchain_core.messages import HumanMessage, AIMessage

    config = {"configurable": {"thread_id": "debug"}}

    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() in {"exit", "quit"}:
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        last_ai = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None,
        )

        print("Bot:", last_ai.content if last_ai else "No response.")