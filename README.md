This repository contains the code and documentation for my MSc thesis project, which focuses on the design and implementation of an agentic AI customer support chatbot for a telecommunications company.

The system uses Large Language Models (LLMs) and LangGraph to create a modular, multi-step, and tool-aware agent capable of reasoning, retrieving information, and performing task-oriented actions.


Setup & Installation
1. Clone the repository
git clone https://github.com/your-username/agentic-telecom-chatbot.git
cd agentic-telecom-chatbot

2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Environment variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here

▶️ Running the Agent
python src/main.py


The agent initializes:

FAISS vector store

LLM

LangGraph execution flow

You can then interact with the chatbot via the command line.
