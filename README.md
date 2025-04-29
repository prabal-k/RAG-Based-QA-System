# RAG-Based HR Policy Chatbot

A **Retrieval-Augmented Generation (RAG)** system that answers HR-related questions using a custom knowledge base and falls back to web search when questions fall outside of the dataset. Built with **Python**, **Streamlit**, **LangChain**, **Groq API**, and **Chroma vector store**.

---

## Objective

To build a question-answering system that:
- Retrieves relevant HR policy information from a custom knowledge base.
- Uses an LLM to generate human-like responses.
- Falls back to DuckDuckGo web search when the user asks questions unrelated to HR content.

---

## Prerequisites

- Python 3.9 or higher
- [Create Groq API Key]( https://console.groq.com/keys)

---

## Features

- ✅ **Interactive Streamlit UI** for user-friendly Q&A experience.
- ✅ **Retrieval System**: Embeds HR FAQ chunks using Hugging Face sentence transformers and stores them in Chroma.
- ✅ **RAG Pipeline**: Retrieves top-3 most relevant chunks from the vector store and feeds them to the LLM to generate a context-aware answer.
- ✅ **Agent Logic with LangGraph**:
  - If the query relates to HR, it searches the knowledge base.
  - If not, it routes to DuckDuckGo search.
- ✅ **Web Search Fallback** using DuckDuckGo for out-of-domain queries.
- ✅ **Dynamic Prompting**:
  - If context is missing: "I couldn't find information about this."
  - If LLM says “I don’t know”: Suggests user to rephrase the question.
- ✅ **Lightweight, Local Dataset** (`company_QA.csv`) with HR questions and answers.
- ✅ **Easily Extensible**: Can be modified to work for other domains like movies, recipes, etc.

---
## System Architecture

![Image](https://github.com/user-attachments/assets/a19be4f1-db35-410a-9b32-fc58106c9d4e)

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/prabal-k/RAG-Based-QA-System

cd hr-rag-chatbot
```

### 2. Create and Activate a Python Virtual Environment

### On Windows:
```
python -m venv venv

venv\Scripts\activate
```
### On Linux/macOS:
```
python3 -m venv venv

source venv/bin/activate
```
### 3. Install Required Dependencies
``
pip install -r requirements.txt
``
### 4. Configure Environment Variables

Create a .env file in the root folder with the following content:

GROQ_API_KEY = "your_groq_api_key_here"

### 5. Run the Application
``
streamlit run app.py
``





