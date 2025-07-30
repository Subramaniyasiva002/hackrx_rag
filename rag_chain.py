import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Changed from Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from tempfile import NamedTemporaryFile
import gc
import hashlib

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = "./vector_cache"  # For persistent vector stores

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Hugging Face API call with memory optimization
def generate_response(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct:novita",
        "messages": [
            {"role": "system", "content": "You are a helpful health insurance assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300
    }

    try:
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # Add timeout to prevent hanging
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    finally:
        gc.collect()  # Clean up after API call

# Optimized prompt template
template = """[INST]
You are a professional Health Insurance Assistant.
Provide a short and policy-specific answer in one sentence using only verified content from this policy.
Do not include any explanations or formatting.

Policy Text:
{context}

User Question:
{query}
[/INST]"""
prompt = ChatPromptTemplate.from_template(template)

def load_remote_pdf(url: str) -> str:
    # Generate unique filename based on URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    temp_path = f"/tmp/{url_hash}.pdf"
    
    # Skip download if file exists
    if os.path.exists(temp_path):
        return temp_path

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "application/pdf" not in content_type:
        raise ValueError("URL did not return a PDF file.")

    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return temp_path

# Optimized RAG chain with caching
def build_rag_chain(pdf_path: str):
    # Generate unique cache key for this PDF
    file_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.faiss")
    
    # Use cached vector store if available
    if os.path.exists(cache_path):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Load and split PDF with smaller chunks
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # More memory-efficient splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Reduced from 500
            chunk_overlap=50,  # Reduced from 100
            length_function=len
        )
        chunks = splitter.split_documents(docs)
        
        # Use FAISS instead of Chroma for memory efficiency
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(cache_path)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduced from 5
    
    # Memory-efficient RAG pipeline
    return (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | (lambda chat_prompt: generate_response(chat_prompt.to_string()))
        | StrOutputParser()
    )
