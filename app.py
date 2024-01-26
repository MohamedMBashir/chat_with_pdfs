import streamlit as st
import os
import tempfile
from langchain_logic import Rag_chain
import openai
from dotenv import load_dotenv, find_dotenv

# from dotenv import load_dotenv, find_dotenv
try:
    _ = load_dotenv(find_dotenv()) # read local .env file
except:
    pass

# Load and set OpenAI API Key
if 'OPENAI_API_KEY' not in os.environ:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        openai.api_key = api_key
else:
    openai.api_key = os.environ['OPENAI_API_KEY']


# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None

# PDF Loader
st.title("Chat with Your PDF")
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])


if pdf_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        temp_file_path = tmp_file.name

    st.success("PDF Loaded successfully!")
    # Process PDF and create QA Chain
    st.session_state['qa_chain'] = Rag_chain(temp_file_path, n_documents=5)

    # Remove the temporary file after processing (optional)
    os.remove(temp_file_path)

# Chat Interface
if st.session_state['qa_chain']:
    st.header("Chat with your PDF")
    user_input = st.text_input("Your question:", key="user_query")

    if user_input:
        # Get response from the QA Chain
        response = st.session_state['qa_chain']({'question': user_input})['answer']
        st.text_area("Response:", value=response, height=100, disabled=True)
