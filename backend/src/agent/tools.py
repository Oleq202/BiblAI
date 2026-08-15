import os
import json
import time
import atexit
from pathlib import Path

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

try:
    from .schemas import StatementVerdict, Verdict
except (ImportError, ValueError):
    from schemas import StatementVerdict, Verdict

BASE_DIR = Path(__file__).resolve().parents[2]
QDRANT_DIR = BASE_DIR / "data" / "qdrant"
COLLECTION_NAME = "biblia_tysiaclecia"
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
GEMINI_MODEL = "models/gemini-3.5-flash-lite"

if "GEMINI_API_KEY" not in os.environ:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")

_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
_qdrant = QdrantClient(path=str(QDRANT_DIR))
atexit.register(_qdrant.close)


def retrieve(query, top_k=5):
    embedding = _embed_model.encode([query]).tolist()[0]
    results = _qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=top_k,
    ).points
    return [point.payload for point in results]


VERDICT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["directly_supported", "directly_contradicted", "not_directly_stated"],
        },
        "confidence": {"type": "number"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "quote": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["supports", "contradicts", "related_but_not_direct"],
                    },
                },
                "required": ["ref", "quote", "relation"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "citations", "reasoning"],
}

SYSTEM_PROMPT = """Jesteś asystentem sprawdzającym, czy dane stwierdzenie jest \
bezpośrednio poparte, bezpośrednio sprzeczne, czy wprost niewspomniane w \
podanych fragmentach Pisma Świętego (Biblia Tysiąclecia).

Zasady:
- "directly_supported": tekst wprost stwierdza to samo co użytkownik.
- "directly_contradicted": tekst wprost stwierdza coś przeciwnego.
- "not_directly_stated": żaden werset nie odnosi się wprost do tego \
stwierdzenia - dotyczy to zwłaszcza stwierdzeń interpretacyjnych, \
teologicznych lub wymagających wnioskowania. W tym wypadku podaj \
najbliżej powiązane fragmenty, ale NIE wymuszaj werdyktu "supported" \
ani "contradicted" tylko dlatego, że temat jest powiązany.
- Cytuj TYLKO fragmenty rzeczywiście obecne w podanym kontekście. \
Nigdy nie wymyślaj wersetów ani cytatów.
- confidence powinno być niższe, gdy dowody są pośrednie lub sprzeczne \
wewnętrznie."""


def classify_support(statement, retrieved_chunks, max_retries=3):
    context = "\n\n".join(
        f"[{chunk['verse_refs']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )

    user_message = f"""Stwierdzenie do sprawdzenia:
"{statement}"

Fragmenty Pisma Świętego znalezione przez wyszukiwanie (Biblia Tysiąclecia):

{context}

Oceń, czy powyższe stwierdzenie jest bezpośrednio poparte, sprzeczne, czy \
niewspomniane wprost w tych fragmentach."""

    last_error = None
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=VERDICT_RESPONSE_SCHEMA,
                    max_output_tokens=2048,
                ),
            )
            result = json.loads(response.text)
            return StatementVerdict(
                statement=statement,
                verdict=Verdict(result["verdict"]),
                confidence=result["confidence"],
                citations=result["citations"],
                reasoning=result["reasoning"],
            )
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  attempt {attempt + 1} failed ({e.__class__.__name__}: {e}), retrying in {wait}s...")
            time.sleep(wait)

    raise last_error