# src/state.py
from __future__ import annotations

from typing import TypedDict, List, Optional, Literal, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.intents import Intent


class RetrievedDoc(TypedDict):
    """Small, serializable representation of a retrieved document."""
    content: str
    source: Optional[str]
    score: Optional[float]


Route = Literal[
    "technical_support",
    "billing",
    "info_lookup",
    "escalation",
    "chitchat",
    "fallback",
]


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    total=False means fields are optional until written.
    """

    # Conversation (✅ reducer appends messages instead of overwriting)
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str

    # Routing
    intent: Intent
    intent_confidence: float
    route_to: Route
    current_flow: Optional[Route]  #lets the router_node remember what kind of conversation the user is currently in

    # Retrieval (RAG)
    retrieved_docs: List[RetrievedDoc]
    retrieval_query: str

    # Reasoning / outcomes
    draft_answer: str
    final_answer: str
    follow_up_questions: List[str]

    # Safety / robustness
    error: Optional[str]
    needs_handoff: bool
    debug: Any