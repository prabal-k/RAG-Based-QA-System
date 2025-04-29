# app.py
import streamlit as st
from your_rag_system import get_ai_response  # Import your RAG system function

# Page config
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="💬",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your HR Policy Assistant. Ask me about company policies or general questions."}
    ]

# Sidebar with settings
with st.sidebar:
    st.title("Settings")
    model_choice = st.selectbox(
        "Choose AI Model",
        ["Mixtral-8x7b", "Llama3-70b"],
        index=0
    )
    st.divider()
    st.markdown("💡 Powered by Groq LPUs")
    st.markdown("[Get API Key](https://console.groq.com/)")

# Main chat interface
st.title("💬 HR Policy Chatbot")
st.caption("Ask about company policies or general knowledge")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Your question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response (replace with your RAG system call)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_ai_response(prompt, model=model_choice)
        st.markdown(response)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})