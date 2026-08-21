FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"

COPY . .

RUN python -c "import sys; sys.path.insert(0, 'backend/src/agent'); sys.path.insert(0, 'backend/src/api'); from bm_25 import get_bm25_index; get_bm25_index(); from chapter_service import get_chapter; get_chapter('Wj', 3)"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]