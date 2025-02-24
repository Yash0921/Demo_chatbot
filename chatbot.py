import os
import openai
import faiss
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set OpenAI API Key (hardcoded as per your request)
API_KEY = "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"
openai.api_key = API_KEY

st.title("File-Based Chatbot (RAG Model)")

def load_pdf(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    return documents

# Use a raw string for Windows file path to handle backslashes correctly
pdf_path = r"E:\Demo_chatbot\inception_movie_info.pdf"
if os.path.exists(pdf_path):
    st.write("Using preloaded movie information PDF: Inception")
    documents = load_pdf(pdf_path)
else:
    uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    if uploaded_file is not None:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        documents = load_pdf("uploaded_file.pdf")
    else:
        documents = []

if documents:
    # Split the document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Generate embeddings with the API key provided directly
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    
    # Setup the Chat Model with the API key explicitly provided
    chat_model = ChatOpenAI(model_name="gpt-4", openai_api_key=API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever, chain_type="stuff")

    st.subheader("Ask a Question Based on the File")
    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer") and user_query:
        response = qa_chain.run(user_query)
        st.write("### Answer:")
        st.write(response)
else:
    st.write("No documents loaded. Please upload a PDF file.")
