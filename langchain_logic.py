import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Constants
llm_name = "gpt-3.5-turbo-1106"

# Create a Retrieval Chain
def Rag_chain(file, n_documents, chain_type="stuff"):

    # Load pdf
    loader = PyPDFLoader(file)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=128,
    separators=["\n\n", "\n", ". ", " ", ""])

    chunks = text_splitter.split_documents(documents)

    # Define embedding function
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key if openai_api_key else None)

    # Create vector database from data
    vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./db/")
    
    # Retriver
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": n_documents})

    # llm
    llm=ChatOpenAI(model_name=llm_name, temperature=0)

    # Memory 
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


    #ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type=chain_type,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        return_generated_question=False,
    )

    return qa_chain





