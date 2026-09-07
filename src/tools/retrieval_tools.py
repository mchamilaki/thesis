#retrieval_tools.py

import os

from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Semantic Memory (FAISS) ---
print("Initializing semantic memory (FAISS index)…")

# project root = two levels up from src/tools/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


@tool
def search_knowledge_base(query: str) -> str:
    """Search the technical support knowledge base for solutions to connectivity,
    router, and service issues. Use a concise, technical search query describing
    the problem (e.g. 'intermittent wifi disconnects router')."""
    docs = ubuntu_vs.similarity_search(query, k=5)
    if not docs:
        return "No relevant results found in the knowledge base."
    return "\n\n---\n\n".join(d.page_content for d in docs)