from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.tools.retrieval_tools import ubuntu_vs

print("chunks indexed:", ubuntu_vs.index.ntotal)
print("embedding dim:", ubuntu_vs.index.d)
print("index type:", type(ubuntu_vs.index).__name__)
print("distance strategy:", ubuntu_vs.distance_strategy)

docs = list(ubuntu_vs.docstore._dict.values())
lengths = [len(d.page_content) for d in docs]
print("mean chars:", sum(lengths) / len(lengths))
print("min/max chars:", min(lengths), max(lengths))
print("\n--- first chunk ---")
print(docs[0].page_content[:500])
print("metadata:", docs[0].metadata)