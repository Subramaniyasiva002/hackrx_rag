import os
import re
import requests
import hashlib
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# === Hugging Face LLaMA 3 Call ===
def generate_response(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": "openai/gpt-oss-120b:novita",
        "messages": [
            {"role": "system", "content": "You are a helpful health insurance assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800
    }

    response = requests.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === Prompt Template ===

template = """You are an expert Health Insurance Policy Assistant.

Your tasks:
1. First, determine whether the user is asking for:
   - a factual explanation (intent-based), or
   - a coverage decision (decision-based).

2. Then:
- If the query is intent-based, answer in **1-2 clear sentences** based on the provided policy excerpt [Detailed reason/Detailed Explanation]. 
- If the query is decision-based, Identify both coverage clause or benefit section , Exclusion or waiting period clause (if any applies) respond in this format:
  [Yes/No] – [Procedure] is [covered/not covered] under [Coverage Clause/Section] and subject to [Exclusion/Waiting Period Clause/Section] because [Detailed reason/Detailed Explanation].


User question: {query}
Policy excerpt: {context}

Your answer:
"""


prompt = ChatPromptTemplate.from_template(template)

# === PDF Utilities ===
def load_remote_pdf(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "application/pdf" not in content_type:
        raise ValueError("URL did not return a PDF file.")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        return tmp.name

# === Text Cleaner ===
def clean_and_normalize_text(text: str) -> str:
    text = text.replace("Section C.1.", "\n[WAITING_PERIOD]\nSection C.1.")
    text = text.replace("Section C.2.", "\n[STANDARD_EXCLUSIONS]\nSection C.2.")
    text = text.replace("Section C.3.", "\n[SPECIFIC_EXCLUSIONS]\nSection C.3.")
    text = text.replace("Specified disease/procedure waiting period (Excl02)", "\n[EXCL02_SPECIFIC_DISEASE]\nSpecified disease/procedure waiting period (Excl02)")
    text = text.replace("Pre-existing Diseases (Excl01)", "\n[EXCL01_PRE_EXISTING]\nPre-existing Diseases (Excl01)")
    text = text.replace("Room Rent Limit", "\n[ROOM_RENT_LIMIT]\nRoom Rent Limit")
    text = text.replace("Ayush Benefit", "\n[AYUSH_BENEFIT]\nAyush Benefit")
    text = text.replace("Ectopic pregnancy", "\n[EXCEPTION_ECTOPIC]\nEctopic pregnancy")

    text = re.sub(r'\nPage \d+\s*\|.*?\n', '\n', text)
    text = re.sub(r'HDFC ERGO.*?license\.', '', text, flags=re.DOTALL)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\n(?=\w)', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


# === QUERY PREPROCESSOR ===
def query_preprocessor(query: str) -> str:
    import re

    query = query.strip()

    # Patterns
    age_pattern = re.search(r'\b(\d{1,3})\s*(?:yo|year[- ]?old)?\s*[mMfF]?\b', query)
    gender_pattern = re.search(r'\b(?:male|female|[mMfF])\b', query)
    procedure_pattern = re.search(r"(c[- ]?section|caesarean|surgery|dialysis|stroke|cataract|heart attack|delivery|obesity|knee replacement|ayush)", query, re.IGNORECASE)
    location_pattern = re.search(r"in\s+([a-zA-Z\s]+)", query)
    duration_pattern = re.search(r"(\d+)\s*[-]?\s*month", query, re.IGNORECASE)

    # Compose output
    parts = []

    if age_pattern:
        parts.append(f"Age: {age_pattern.group(1)}")
    if gender_pattern:
        gender = gender_pattern.group(0).upper()
        gender = 'Male' if gender.startswith('M') else 'Female'
        parts.append(f"Gender: {gender}")
    if procedure_pattern:
        parts.append(f"Procedure: {procedure_pattern.group(0).strip().title()}")
    if location_pattern:
        parts.append(f"Location: {location_pattern.group(1).strip().title()}")
    if duration_pattern:
        parts.append(f"Policy Duration: {duration_pattern.group(1)} months")

    parts.append(f"Original Query: {query}")
    return ". ".join(parts)

# === FINAL RAG CHAIN WITH QUERY ENHANCEMENT ===
def build_rag_chain(pdf_path: str, rebuild_index=False):
    embeddings = SentenceTransformerEmbeddings(model_name="intfloat/e5-small-v2")
    final_chunks = []  

    if not rebuild_index and os.path.exists("/persistent/faiss_index"):
        print("🔹 Loading existing FAISS index...")
        vectorstore = FAISS.load_local("/persistent/faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Also reload chunks for BM25
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            text = clean_and_normalize_text(doc.page_content)
            doc.page_content = text
            final_chunks.append(doc)

    else:
        print("🔹 Building FAISS index from scratch...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        for doc in docs:
            text = clean_and_normalize_text(doc.page_content)
            doc.page_content = text
            if "[WAITING_PERIOD]" in text:
                doc.metadata["section"] = "waiting"
            elif "[STANDARD_EXCLUSIONS]" in text:
                doc.metadata["section"] = "standard_exclusion"
            elif "[SPECIFIC_EXCLUSIONS]" in text:
                doc.metadata["section"] = "specific_exclusion"
            elif "Schedule of Benefits" in text:
                doc.metadata["section"] = "schedule"
            else:
                doc.metadata["section"] = "general"

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        for doc in docs:
            splits = splitter.split_text(doc.page_content)
            for chunk_text in splits:
                final_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))

        vectorstore = FAISS.from_documents(final_chunks, embeddings)
        vectorstore.save_local("/persistent/faiss_index")

    # Create retrievers
    bm25_retriever = BM25Retriever.from_documents(final_chunks)
    bm25_retriever.k = 4

    retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})
        ],
        weights=[0.4, 0.6]
    )

    chain = (
        {
            "context": retriever,
            "query": lambda q: f"Original Query: {q}\n\nPreprocessed Query: {query_preprocessor(q)}"
        }
        | prompt
        | (lambda chat_prompt: generate_response(chat_prompt.to_string()))
        | StrOutputParser()
    )

    return chain
