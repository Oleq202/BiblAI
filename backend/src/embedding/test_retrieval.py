from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
QDRANT_DIR = BASE_DIR / "data" / "qdrant"
COLLECTION_NAME = "biblia_tysiaclecia"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path=str(QDRANT_DIR))

query = "stworzenie świata"
embedding = model.encode([query]).tolist()[0]

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=5,
).points

for point in results:
    payload = point.payload
    refs = "; ".join(payload["verse_refs"])
    print(f"[{refs}]  (score={point.score:.3f})")
    print(f"  {payload['text'][:150]}...")
    print()

client.close()
