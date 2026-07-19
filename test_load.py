print("3. importing torch explicitly...")
import torch

print("4. importing ChatOpenAI...")
from langchain_openai import ChatOpenAI

print("5. importing langgraph...")
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

print("6. importing src modules...")
from src.state import AgentState
from src.intents import Intent
from src.tools.billing_tools import fetch_invoice

print("1. importing HuggingFaceEmbeddings...")
from langchain_huggingface import HuggingFaceEmbeddings

print("7. creating embeddings model AFTER all imports...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("   OK, vector length:", len(emb.embed_query("hello")))

print("8. ALL GOOD")