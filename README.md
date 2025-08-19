## Health Insurance Policy QA System – HackRx 2025

We (Mythili, Sriram, and I) developed a Retrieval-Augmented Generation (RAG) system to interpret and answer complex health insurance policy questions with accuracy and explainability. This was our submission for HackRx 2025 – Round 2, deployed on Hugging Face Spaces (FastAPI).

Policy ingestion & tagging – Cleaned PDF docs, added logical markers (e.g., [WAITING_PERIOD], [EXCL01])
Hybrid retrieval – FAISS (semantic + MMR) + BM25 (lexical)
Query preprocessing – Extracted age, gender, duration, procedure, etc.
Intent classification – Distinguish factual vs. decision-based queries
Clause-backed responses – Answers tied to coverage sections & exclusions
LLMs tested – GPT OSS 120B
Langchain Framework used
Huggingface spaces deployment

## Results

✅ Explainable answers with clear clause references
✅ Multi-clause reasoning supported
✅Retrieval accuracy is good (tokenization/index persistence challenges on free tier Hugging Face)

## Takeaway

This project gave us hands-on experience combining semantic retrieval + structured reasoning in the health insurance domain. With better tokenization and persistent vector DB, accuracy can significantly improve.
