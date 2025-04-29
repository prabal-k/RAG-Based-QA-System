import os
import streamlit as st   #For the User Interface
st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬", layout="centered")
from langchain_huggingface import HuggingFaceEmbeddings # Load the  embedding model from huggingface
from langchain_chroma import Chroma #Vectorstore to store the embedded vectors
from langchain_community.document_loaders.csv_loader import CSVLoader #To load the csv file (data containing companys faq)
from langchain_community.tools import DuckDuckGoSearchRun #Search user queries Online
from langchain.chains import create_retrieval_chain # "Combines a retriever (to fetch docs) with the 'create_stuff_document_chain' to automate end-to-end retrieval + answering."
from langchain_groq import ChatGroq  #Load the open source Groq Models
from dotenv import load_dotenv  #Load environemnt variables from .env
load_dotenv()

# --- Initialize Components such as llm ,embedding model and DuckDUCKSearch ---
@st.cache_resource
def init_components():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #Load the hf Embedding model
    llm = ChatGroq(temperature=0.4, model_name="Qwen-Qwq-32b",max_tokens=400) #Initialize the llm
    search = DuckDuckGoSearchRun()  #Duckducksearch
    return embedding_model, llm, search

# --- Create or Load Vectorstore ---
@st.cache_resource
def get_vectorstore(_embedding_model):
    persist_dir = "chroma_index" #Location to store embedding 
    file_path = "company_QA.csv" #Path of the data source

    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=_embedding_model)
    else:
        loader = CSVLoader(file_path=file_path)  #Load the CSV file
        docs = []
        for doc in loader.lazy_load(): #Perfom lazy_load() to load the file content
            docs.append(doc)  
        vectorstore = Chroma.from_documents(documents=docs, embedding=_embedding_model, persist_directory=persist_dir) #Create a vector embeddings
        vectorstore.persist() #Save the vector store
        return vectorstore
    
# --- Initialize the components ---
embedding_model, model, search = init_components()
vectorstore = get_vectorstore(embedding_model)

# --- Streamlit UI ---
def main():

    st.title("💼 HR Policy Chatbot")
    st.caption("Ask about company policies or general knowledge")


if __name__ == "__main__":
    main()
