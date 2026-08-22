import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src" / "agent"))
sys.path.insert(0, str(BASE_DIR / "src" / "api"))

from chapter_service import get_chapter
from graph import verify_statement
from schemas import StatementVerdict

app = FastAPI(
    title="BiblAI",
    description="Weryfikacja stwierdzeń w oparciu o Biblię Tysiąclecia (RAG + LangGraph)",
    version="0.1.0",
)

FRONTEND_DIR = BASE_DIR.parent / "frontend"
if (FRONTEND_DIR / "public").exists():
    app.mount("/public", StaticFiles(directory=str(FRONTEND_DIR / "public")), name="public")


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


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/index.html", methods=["GET", "HEAD"])
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
def favicon():
    fav = FRONTEND_DIR / "public" / "favicon.ico"
    if fav.exists():
        return FileResponse(fav)
    return FileResponse(FRONTEND_DIR / "index.html")


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
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Verification failed: {e}")

    return verdict


@app.get("/chapter")
def get_bible_chapter(
    book: str = Query(..., description="Book abbreviation e.g. 'Wj', 'Rdz', '1 Sm'"),
    chapter: int = Query(..., ge=1, description="Chapter number e.g. 3"),
):
    book_clean = book.strip()
    if not book_clean:
        raise HTTPException(status_code=400, detail="Book abbreviation cannot be empty.")

    result = get_chapter(book_clean, chapter)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Chapter {chapter} of book '{book_clean}' was not found.")

    return result
