# rag_chain.py

import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from tempfile import NamedTemporaryFile

# Load environment variables (HF_TOKEN)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face LLaMA 3 API call
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

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",  # Use router or correct endpoint
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Prompt template for RAG
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
    # Optional: basic sanity check (skip .endswith('.pdf'))
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "application/pdf" not in content_type:
        raise ValueError("URL did not return a PDF file.")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        return tmp.name# Return local path to temp PDF

# RAG chain build function
def build_rag_chain(pdf_path: str):
    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings & Vector Store
    embeddings = SentenceTransformerEmbeddings(model_name="intfloat/e5-small-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # RAG pipeline
    return (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | (lambda chat_prompt: generate_response(chat_prompt.to_string()))  # FIXED
        | StrOutputParser()
    )

