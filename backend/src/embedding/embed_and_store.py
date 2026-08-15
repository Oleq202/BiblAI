import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_PATH = BASE_DIR / "data" / "processed" / 'chunks.jsonl'
QDRANT_DIR = BASE_DIR / "data" / "qdrant"
COLLECTION_NAME = "biblia_tysiaclecia"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

BATCH_SIZE = 64

def load_chunks(path):
    chunks = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def embed_and_store():
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    vector_size = model.get_embedding_dimension()

    print("Connecting to embedded Qdrant...")
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_DIR))

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),)

    print("Embedding and inserting in batches...")
    point_id = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        points = []
        for chunk, embedding in zip(batch, embeddings):
            points.append(PointStruct(
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
            ))
            point_id += 1

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")
    
    count = client.count(COLLECTION_NAME).count
    print(f"Done. Collection '{COLLECTION_NAME}' has {count} points.")
    client.close()


if __name__ == "__main__":
    embed_and_store()