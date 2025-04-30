import os
import streamlit as st   #For the User Interface
st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬", layout="centered")
from langchain_huggingface import HuggingFaceEmbeddings # Load the  embedding model from huggingface
from langchain_chroma import Chroma #Vectorstore to store the embedded vectors
from langchain_community.document_loaders.csv_loader import CSVLoader #To load the csv file (data containing companys faq)
from langchain_community.tools import DuckDuckGoSearchRun #Search user queries Online
from langchain.prompts import PromptTemplate #Create a template
from langchain.chains.combine_documents import create_stuff_documents_chain #form a final prompt with 'context' and ;query'
from langchain.chains import create_retrieval_chain # "Combines a retriever (to fetch docs) with the 'create_stuff_document_chain' to automate end-to-end retrieval + answering."
from langchain_groq import ChatGroq  #Load the open source Groq Models
from langgraph.graph import StateGraph, START, END #Define the State for langgraph
from langgraph.prebuilt import ToolNode,tools_condition #specialized node designed to execute tools within our workflow.
from langchain_core.messages import AnyMessage #Human message or Ai Message
from langgraph.graph.message import add_messages  ## Reducers in Langgraph ,i.e append the messages instead of replace
from typing_extensions import Annotated,TypedDict #Annotated for labelling and TypeDict to maintain graph state 
from langchain_core.tools import tool
from langchain_core.messages import trim_messages # Trim the message and keep past 2 conversation
from langgraph.checkpoint.memory import MemorySaver #Implement langgraph memory
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv  #Load environemnt variables from .env
load_dotenv()
# Create a Unique Id for each user conversation
import uuid

def main():
    # --- Initialize Components such as llm ,embedding model and DuckDUCKSearch ---
    @st.cache_resource
    def init_components():
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #Load the hf Embedding model
        llm = ChatGroq(temperature=0.4, model_name='Qwen-Qwq-32b',max_tokens=650) #Initialize the llm
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
            # vectorstore.persist() #Save the vector store
            return vectorstore
        
    # --- RAG Chain Response ---
    def get_rag_response(query, vectorstore, llm):
        # Creating a MMR retriever ['k': select top 2 similar documents and 'lambda_mult': for diverse documents ]
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.1})
        template = PromptTemplate(
        template= """You are an HR assistant with access to our internal company policies. Answer the user question based only on the given context . If the answer is not present in the context 
        or you dont know the answer then reply with 'I donot have access to the information you are asking for.'
        'context:'
        {context}

        'Question:' 
        {input}
        """,
        input_variable = ['context','input']
        )
        # To create a chain that "Formats retrieved documents + question into a prompt and passes it to the LLM for answering."
        combine_docs_chain = create_stuff_documents_chain(llm, template)
        # To create a final chain to reterive, format prompt and generate answer 
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        result = rag_chain.invoke({"input": query})
        return result['answer']
        
    # --- Initialize the components i.e llm,embedding model ,vectorstore ---
    embedding_model, model, search = init_components() 
    vectorstore = get_vectorstore(embedding_model)

    # Tool A. VectorStore Retriever tool (Convert the rag_chain into a tool)
    @tool
    def retrieve_vectorstore_tool(query: str) -> str:
        """Direct when the user question about a company policy/HR FAQ's."""
        return get_rag_response(query, vectorstore, model)

    #DuckDuckSeach Tool
    @tool
    def duckducksearch_tool(query: str) -> str:
        """Perform DuckDuckGo search for general user queries other then the company's policy."""
        return search.invoke(query)


    # --- Tools and bind the tools with llm ---
    tools = [retrieve_vectorstore_tool, duckducksearch_tool]
    llm_with_tools = model.bind_tools(tools=tools)

    # Initialize the StateGraph
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages] #List of messages appended

    #Function that decides which tool to use for serving the userquery
    def tool_calling_llm(state:State)->State:
        print(state['messages'])
        selected_msg = trim_messages(
            state["messages"],
            token_counter=len,  #len will count the number of messages rather than tokens
            max_tokens=10,  # allow up to 10 messages.(i.e 2-3 past conversation between human and Ai)
            strategy="last",        
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
        return {"messages":[llm_with_tools.invoke(selected_msg)]}

    # Initialize the StateGraph
    builder = StateGraph(state_schema=State)

    #Adding Nodes
    builder.add_node('tool_calling_llm',tool_calling_llm) #returns the tools that is to be used
    builder.add_node('tools',ToolNode(tools=tools)) #Uses the tool specified to fetch result

    #Adding Edges
    builder.add_edge(START,'tool_calling_llm')
    builder.add_conditional_edges(
        'tool_calling_llm',
        # If the latest message from AI is a tool call -> tools_condition routes to tools
        # If the latest message from AI is a not a tool call -> tools_condition routes to LLM, then generate final response and END
        tools_condition
    )
    builder.add_edge('tools','tool_calling_llm')
    memory = MemorySaver()

    #Compile the graph
    graph = builder.compile(
        checkpointer=memory
    )


    # Initialize thread_id(unique id for each conversation) in session_state if not exists
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your HR Policy Assistant. Ask me about company policies or general questions."}
        ]

    st.title("HR Policy Chatbot")
    st.caption("Ask about company policies or general knowledge")

    # display enitire chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question"):
        # adding user message/query to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt) 
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Creating a conversation between human and AI
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # Invoke the graph with full message history(i.e human and ai message)
                try:
                    response = graph.invoke(
                        {"messages": langchain_messages}, #Pass the entire chat history 
                        config=config
                    )
                    final_response = response['messages'][-1].content #last msg from AI which is the final response for user's query
                except Exception as e:
                    final_response = f"There was a problem this time, please try again. {e}"  #Incase LLM fails to answer, even after using the tools 
            
                st.markdown(final_response)
        
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()