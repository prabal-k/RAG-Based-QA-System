RAG-Based QA System with Groq

A Retrieval-Augmented Generation (RAG) system for answering HR policy questions, combining document retrieval with LLM-powered generation. Supports dynamic tool calling for knowledge base vs. web search routing.
Features

    Custom HR Policy QA: Answers questions from a curated HR knowledge base

    Tool-Calling LLM: Uses Groq's ultra-fast LLMs (Llama 3/Mixtral) to select between:

        Vector store retrieval (HR policies)

        Web search (general questions)

    LangGraph Workflow: Manages tool selection and response generation

Prerequisites

    Python 3.9+

    Groq API key (free at console.groq.com)

    DuckDuckGo Search API (free)

Setup

    Clone the repository
    bash

git clone https://github.com/yourusername/RAG-Based-QA-System.git
cd RAG-Based-QA-System

Create and activate virtual environment
bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

Install dependencies
bash

pip install -r requirements.txt

Set up environment variables
Create a .env file:
bash

    echo "GROQ_API_KEY=your_groq_api_key" > .env

Running the System
1. Initialize the Knowledge Base

Place your HR policy documents in data/hr_policies/ and run:
bash

python build_vectorstore.py

2. Start the QA System

Run the main application:
bash

python main.py

3. Test with Sample Queries

The system will prompt for questions. Try:

    "How many sick days do we get?"

    "What's our remote work policy?"

    "Who won the last World Cup?"

Configuration

Edit config.py to customize:
python

MODEL_NAME = "mixtral-8x7b-32768"  # Alternatives: llama3-70b-8192
TOOL_TIMEOUT = 30  # seconds
KNOWLEDGE_BASE_DIR = "data/hr_policies/"