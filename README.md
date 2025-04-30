# RAG-Based HR Policy Chatbot

A **Retrieval-Augmented Generation (RAG)** system that answers HR-related questions using a custom knowledge base and falls back to web search when questions fall outside of the dataset. Built with **Python**, **Streamlit**, **LangChain**,**LangGraph**, **Groq API**, and **Chroma vector store**.

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

- **Interactive Streamlit UI** for user-friendly Q&A experience.
- **Conversational Window Buffer Memory** – Maintains state for the last 2-3 interactions, allowing follow-up questions without full context loss.
- **Retrieval System**: Embeds HR FAQ chunks using Hugging Face sentence transformers and stores them in Chroma.
- **RAG Pipeline**: Retrieves top-3 most relevant chunks from the vector store and feeds them to the LLM to generate a context-aware answer.
- **Agent Logic with LangGraph**:
  - If the query relates to HR, it searches the knowledge base.
  - If not, it routes to DuckDuckGo search.
- **Web Search Fallback** using DuckDuckGo for out-of-domain queries.
- **Dynamic Prompting**:
  - If context is missing: "I couldn't find information about this."
  - If LLM says “I don’t know”: Suggests user to rephrase the question.
- **Lightweight, Local Dataset** (`company_QA.csv`) with HR questions and answers.
- **Easily Extensible**: Can be modified to work for other domains like movies, recipes, etc.

---
## System Architecture

![Image](https://github.com/user-attachments/assets/a19be4f1-db35-410a-9b32-fc58106c9d4e)

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/prabal-k/RAG-Based-QA-System
```

### 2. Open with VsCode ,Create and Activate a Python Virtual Environment

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

---

## Challenges & Design Choices in HR Policy Chatbot Implementation

### 1. Introduction

Outlines the key challenges and design decisions made during the development of an HR Policy Chatbot using LangChain, Streamlit, and Groq. 

### 2. Key Challenges & Solutions

### 2.1 Memory Management in Streamlit

#### Problem:

  1. Streamlit reruns the entire script on user interaction, causing loss of conversation history.

  2. A new thread_id was being generated on each interaction, breaking memory continuity.

#### Solution:

  1. Stored thread_id in st.session_state to maintain consistency.

  2. Used LangGraph's MemorySaver with a stateful StateGraph to preserve conversation context.

### 2.2 Tool Calling with LLMs

#### Problem:

  1. Initially tested Hugging Face LLMs, but they struggled with structured tool calling (e.g., choosing between RAG and DuckDuckGo Search).

  2. Some Models (e.g., Mistral-7b model) ignored tool specifications or hallucinated function calls.

### Solution:

  1. Shifted to Groq’s API (Qwen-Qwq-32b) for better tool-calling reliability.

  2. Implemented a conditional edge-based StateGraph to route queries to:

        RAG (Chroma DB) for HR policy questions.

        DuckDuckGo Search for general knowledge.

### 2.3 RAG Implementation Challenges

#### Problem:

  1. MMR (Maximal Marginal Relevance) retrieval sometimes returned irrelevant documents, leading to incorrect answers.

### Solution:

  1. Paramater setup: search_kwargs (k=2, lambda_mult=0.4) to balance relevance and diversity.

  2. Used lazy loading (CSVLoader.lazy_load()) to handle large files efficiently.

---

### 3. Key Design Choices

### 3.1 Hybrid Retrieval System

  1. Two-tool architecture:

        VectorStore Retriever (Chroma DB) for structured HR policy queries.

        DuckDuckGo Search for real-time external knowledge.

  2. Conditional routing via tools_condition in StateGraph.

### 3.2 Performance Trade-offs

  1. Groq’s speed vs. cost: Faster than Hugging Face models but API-dependent.


## Snapshots

### This demonstrates the Application along with the friendly user interface.

![Image](https://github.com/user-attachments/assets/49e93e2b-e4bd-4912-ab25-fd9f02ec5e16)

![Image](https://github.com/user-attachments/assets/01f8b54e-64b3-46d8-a3f9-5cb96b3c51ce)

---

![Image](https://github.com/user-attachments/assets/17efe124-9f93-459c-bd9e-e49e55a67570)

---
![Image](https://github.com/user-attachments/assets/4956af8f-ba75-4044-bdb2-d5f9dfd64382)

---

![Image](https://github.com/user-attachments/assets/ae6befc3-0303-4b22-9166-748f066bc653)



