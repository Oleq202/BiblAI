import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src" / "agent"))

from graph import verify_statement
from schemas import StatementVerdict

app = FastAPI(
    title="BiblAI",
    description="Weryfikacja stwierdzeń w oparciu o Biblię Tysiąclecia (RAG + LangGraph)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerifyRequest(BaseModel):
    statement: str

class HealthResponse(BaseModel):
    status: Literal["ok"]

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.post("/verify", response_model=StatementVerdict)
def verify(request: VerifyRequest):
    statement = request.statement.strip()
    if not statement:
        raise HTTPException(status_code=400, detail="Statement cannot be empty.")
    if len(statement) > 500:
        raise HTTPException(status_code=400, detail="Statement too long (max 500 characters).")

    try:
        verdict = verify_statement(statement)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Verification failed: {e}")

    return verdict