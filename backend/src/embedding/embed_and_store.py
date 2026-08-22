import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
QDRANT_DIR = BASE_DIR / "data" / "qdrant"
COLLECTION_NAME = "biblia_tysiaclecia"
MODEL_NAME = "models/gemini-embedding-001"
VECTOR_DIM = 768
BATCH_SIZE = 100


def _get_client():
    if "GEMINI_API_KEY" not in os.environ:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_and_store():
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    total_chunks = len(chunks)
    print(f"Loaded {total_chunks} chunks")

    client_genai = _get_client()

    target_dir = BASE_DIR / "data" / "qdrant_gemini"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to embedded Qdrant at {target_dir}...")
    client_qdrant = QdrantClient(path=str(target_dir))

    if not client_qdrant.collection_exists(COLLECTION_NAME):
        client_qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    current_count = client_qdrant.count(COLLECTION_NAME).count
    start_index = current_count
    print(f"Starting from index {start_index}/{total_chunks}...")

    print(f"Embedding and inserting in batches of {BATCH_SIZE} using {MODEL_NAME}...")
    point_id = start_index
    for i in range(start_index, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]

        embeddings = None
        for attempt in range(10):
            try:
                res = client_genai.models.embed_content(
                    model=MODEL_NAME,
                    contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=VECTOR_DIM),
                )
                embeddings = [e.values for e in res.embeddings]
                break
            except Exception as e:
                err_str = str(e)
                wait_time = 25 if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str else (3 * (attempt + 1))
                print(f"  [Attempt {attempt + 1}] Rate limit or error at index {i}: waiting {wait_time}s...")
                time.sleep(wait_time)

        if embeddings is None:
            raise RuntimeError(f"Failed to embed batch starting at index {i}")

        points = []
        for chunk, embedding in zip(batch, embeddings):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "book_abbr": chunk["book_abbr"],
                        "chapter": chunk["chapter"],
                        "verse_start": chunk["verse_start"],
                        "verse_end": chunk["verse_end"],
                        "verse_refs": chunk["verse_refs"],
                        "text": chunk["text"],
                    },
                )
            )
            point_id += 1

        client_qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Processed {min(i + BATCH_SIZE, total_chunks)}/{total_chunks}")
        time.sleep(1.0)

    count = client_qdrant.count(COLLECTION_NAME).count
    print(f"Done. Collection '{COLLECTION_NAME}' has {count} points.")
    client_qdrant.close()

    import shutil

    final_dir = BASE_DIR / "data" / "qdrant"
    try:
        shutil.rmtree(final_dir, ignore_errors=True)
    except Exception:
        pass
    if target_dir.exists():
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        try:
            target_dir.rename(final_dir)
            print("Successfully updated backend/data/qdrant!")
        except Exception:
            print("Note: Finished indexing in backend/data/qdrant_gemini")


if __name__ == "__main__":
    embed_and_store()
