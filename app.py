from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from rag_chain import build_rag_chain, load_remote_pdf
import uvicorn
import os
from dotenv import load_dotenv

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

# Dependency that checks for the token
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_query(data: QueryRequest):
    try:
        pdf_path = load_remote_pdf(data.documents)
        rag_chain = build_rag_chain(pdf_path)
        answers = [rag_chain.invoke(q) for q in data.questions]
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
