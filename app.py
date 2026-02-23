import streamlit as st
import os
from typing import Any, Dict, List
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- LangChain & LangGraph Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Import your custom modules ---
# Ensure these exist in your src/ folder or adjust imports accordingly
from src.state import AgentState
from src.intents import Intent
from src.tools.billing_tools import fetch_invoice

# --- Page Config (Tab Title & Icon) ---
st.set_page_config(page_title="Telecom Agent Demo", page_icon="🤖")

# --- 1. Setup Graph & Resources (Cached) ---
# We use @st.cache_resource so the model/vector store loads only ONCE, not on every click.

@st.cache_resource
def setup_graph():
    """Initializes the FAISS index, LLM, and LangGraph builder."""
    
    # A. Setup Vector Store
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_DIR = os.path.join(BASE_DIR, "data", "ubuntu_qa_index")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Safety check if index exists
    if os.path.exists(INDEX_DIR):
        vector_store = FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        # Fallback if no index found (prevents crash during demo)
        vector_store = None 

    # B. Setup LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    billing_llm = llm.bind_tools([fetch_invoice])
    
    # --- Define Nodes (Scoped inside to capture llm/vs) ---
    
    def call_llm(state: AgentState):
        messages = state["messages"]
        user_query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        
        # RAG Logic
        augmented_messages = list(messages)
        if user_query and vector_store:
            docs = vector_store.similarity_search(user_query, k=3)
            if docs:
                kb_text = "\n\n".join(d.page_content for d in docs)
                kb_system = SystemMessage(content=f"Relevant Knowledge:\n{kb_text}")
                augmented_messages = [kb_system] + messages
        
        response = llm.invoke(augmented_messages)
        return {"messages": [response]}

    def router_node(state: AgentState):
        messages = state["messages"]
        last_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        user_text = last_msg.content.lower() if last_msg else ""

        # Simple keyword routing (Same as your original logic)
        route_to = "fallback"
        intent = Intent.UNKNOWN

        if any(w in user_text for w in ["bill", "payment", "invoice"]):
            route_to = "billing"
            intent = Intent.BILLING
        elif any(w in user_text for w in ["slow", "internet", "router"]):
            route_to = "technical_support"
            intent = Intent.TECHNICAL_SUPPORT
        elif any(w in user_text for w in ["human", "agent"]):
            route_to = "escalation"
            intent = Intent.ESCALATION
        elif any(w in user_text for w in ["hi", "hello"]):
            route_to = "chitchat"
            intent = Intent.CHITCHAT

        return {"route_to": route_to, "intent": intent}

    def billing_llm_node(state: AgentState):
        return {"messages": [billing_llm.invoke(state["messages"])], "current_flow": "billing"}

    def billing_should_use_tools(state: AgentState):
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else "done"

    # Simple Nodes
    def escalation_node(state):
        return {"messages": [AIMessage(content="Transferring you to a human agent...")]}
    
    def chitchat_node(state):
        return {"messages": [llm.invoke([SystemMessage(content="Be friendly.")] + state["messages"])]}

    def fallback_node(state):
        return {"messages": [llm.invoke([SystemMessage(content="Politely ask for clarification.")] + state["messages"])]}

    # --- Build Graph ---
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("technical_support", call_llm)
    builder.add_node("billing", billing_llm_node)
    builder.add_node("escalation", escalation_node)
    builder.add_node("chitchat", chitchat_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("tools", ToolNode([fetch_invoice]))

    builder.set_entry_point("router")
    
    builder.add_conditional_edges("router", lambda s: s["route_to"], {
        "technical_support": "technical_support",
        "billing": "billing",
        "escalation": "escalation",
        "chitchat": "chitchat",
        "fallback": "fallback",
    })
    
    builder.add_conditional_edges("billing", billing_should_use_tools, {"tools": "tools", "done": END})
    builder.add_edge("tools", "billing")
    builder.add_edge("technical_support", END)
    builder.add_edge("escalation", END)
    builder.add_edge("chitchat", END)
    builder.add_edge("fallback", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# Initialize Graph
graph = setup_graph()

# --- 2. Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history for UI
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

# --- 3. Sidebar (For Thesis Demo) ---
with st.sidebar:
    st.header("Thesis Debug Panel")
    st.write(f"**Session ID:** `{st.session_state.thread_id}`")
    
    if st.button("Clear Memory"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.info("The sidebar helps you explain the 'State' to supervisors without cluttering the main chat.")

# --- 4. Main Chat Interface ---
st.title("Telecom Support Agent")
st.caption("Master's Thesis Demo | LangGraph + RAG")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
if user_input := st.chat_input("How can I help you?"):
    # A. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # B. Run Graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # We use a spinner to show activity
    with st.spinner("Agent is thinking..."):
        # Run the graph
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        # Extract the last AI message
        last_ai_msg = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), 
            None
        )
        
        if last_ai_msg:
            response_content = last_ai_msg.content
            
            # C. Display Agent Message
            with st.chat_message("assistant"):
                st.markdown(response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            
            # Optional: Show 'Intent' in sidebar for the demo
            if "intent" in result:
                st.sidebar.success(f"Detected Intent: {result['intent']}")