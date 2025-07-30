from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from rag_chain import build_rag_chain, load_remote_pdf
import uvicorn
import os
from dotenv import load_dotenv
import gc

# Load env variables
load_dotenv()
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY")

app = FastAPI()

# Security scheme
bearer_scheme = HTTPBearer()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Token verification
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_query(data: QueryRequest):
    try:
        pdf_path = load_remote_pdf(data.documents)
        rag_chain = build_rag_chain(pdf_path)
        
        # Process questions in batches if many
        answers = []
        for question in data.questions:
            answers.append(rag_chain.invoke(question))
            gc.collect()  # Clean up after each question
        
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()

if __name__ == "__main__":
    # Configure for lower memory usage
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker to reduce memory
        limit_concurrency=5,  # Limit simultaneous requests
        timeout_keep_alive=30  # Close idle connections
    )
