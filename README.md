This repository contains the code and documentation for my MSc thesis project, which focuses on the design and implementation of an agentic AI customer support chatbot for a telecommunications company.

The system uses Large Language Models (LLMs) and LangGraph to create a modular, multi-step, and tool-aware agent capable of reasoning, retrieving information, and performing task-oriented actions.

⚙️ Setup & Installation

Clone the repository:

git clone https://github.com/your-username/agentic-telecom-chatbot.git
cd agentic-telecom-chatbot


Create and activate a virtual environment:
python -m venv venv


Mac/Linux:
source venv/bin/activate

Windows:
venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt

🔐 Environment Variables

Create a .env file and put your own API key here:
OPENAI_API_KEY=your_api_key_here

▶️ Running the System
python main.py


The agent will initialize:

FAISS semantic memory
LLM
LangGraph execution graph
Checkpoint-based memory

You can then interact with the chatbot via the command line.

