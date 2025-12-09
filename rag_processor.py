# rag_processor.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain import RetrievalQA
# <-- THIS IS THE CORRECT, stable import.

# 1. Load the environment variables (API Key)
load_dotenv()
# Note: Ensure OPENAI_API_KEY is set in your .env file!

def build_rag_chain(document_path: str):
    """
    Sets up the RAG pipeline: Load -> Split -> Embed -> Store -> Create Chain
    """
    # 2. Load the document
    print("Loading and processing document...")
    loader = PyPDFLoader(r"C:\Users\DELL\Downloads\iqigaiOR.pdf")
    documents = loader.load()

    # 3. Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 4. Create embeddings and store in a vector database (ChromaDB)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)

    # 5. Create the RetrievalQA Chain
    # This chain handles: (1) retrieving relevant chunks, (2) injecting them into the prompt, (3) calling the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # RAG Chain setup
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' means stuffing all retrieved documents into the prompt
        retriever=vectorstore.as_retriever()
    )
    print("RAG chain successfully built.")
    return qa_chain

def get_answer(qa_chain, question: str):
    """
    Queries the RAG chain and returns the result.
    """
    print(f"\nAsking: {question}")
    # This is where the magic happens: retrieve context + generate answer
    result = qa_chain({"query": question})
    return result['result']

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found. Please set it in your .env file.")
    else:
        # Replace 'sample_doc.pdf' with the name of your file
        rag_chain = build_rag_chain('sample_doc.pdf')
        
        # Example Query
        query = "What is the main topic of this document?"
        answer = get_answer(rag_chain, query)
        
        print("\n--- Answer ---")
        print(answer)
        print("--------------")
        
        # Now try a more specific question!