import os
import json
import time
import hashlib
import atexit
import urllib.request
from pathlib import Path

from google import genai
from google.genai import types
from qdrant_client import QdrantClient

try:
    from .schemas import StatementVerdict, Verdict
except (ImportError, ValueError):
    from schemas import StatementVerdict, Verdict

BASE_DIR = Path(__file__).resolve().parents[2]
QDRANT_DIR = BASE_DIR / "data" / "qdrant"
COLLECTION_NAME = "biblia_tysiaclecia"
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
GEMINI_MODEL = "models/gemini-3.5-flash-lite"
USE_CROSS_ENCODER = os.environ.get("USE_CROSS_ENCODER", "false").lower() == "true"
RERANK_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

CACHE_DIR = BASE_DIR / "data" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = None
_embed_model = None
_qdrant = None
_reranker = None


def _get_client():
    global _client
    if _client is None:
        if "GEMINI_API_KEY" not in os.environ:
            env_file = BASE_DIR / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client


def _encode_query_hf(query: str, token: str) -> list[float]:
    model_id = f"sentence-transformers/{EMBED_MODEL_NAME}" if not EMBED_MODEL_NAME.startswith("sentence-transformers/") else EMBED_MODEL_NAME
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    payload = json.dumps({
        "inputs": query,
        "options": {"wait_for_model": True, "use_cache": True}
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                return result[0]
            return result
        raise ValueError(f"Unexpected response from HF inference: {result}")



def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_embedding(query: str) -> list[float]:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if hf_token:
        try:
            return _encode_query_hf(query, hf_token)
        except Exception as e:
            print(f"HF API inference error: {e}, falling back to local embedder.")
    
    embed_model = _get_embed_model()
    import torch
    with torch.inference_mode():
        return embed_model.encode([query], show_progress_bar=False).tolist()[0]


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(path=str(QDRANT_DIR))
        atexit.register(_qdrant.close)
    return _qdrant


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def retrieve(query, top_k=5, return_scores=False):
    embedding = _get_embedding(query)
    qdrant = _get_qdrant()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=top_k,
    ).points
    payloads = [point.payload for point in results]
    if return_scores:
        scores = [point.score for point in results]
        return payloads, scores
    return payloads



def rerank(query, chunks, scores=None):
    if USE_CROSS_ENCODER:
        reranker = _get_reranker()
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return reranked
    
    if scores is not None and len(scores) == len(chunks):
        return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, 1.0) for chunk in chunks]



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
wewnętrznie.
- Jeśli odpowiedź wymaga fragmentu, którego NIE MA w podanym kontekście, \
NIE wspominaj, gdzie taki fragment by się znajdował ani co by powiedział - \
po prostu stwierdź, że dostarczony kontekst nie zawiera odpowiedzi."""


def _cache_key(statement, retrieved_chunks):
    chunk_ids = sorted(c["chunk_id"] for c in retrieved_chunks)
    raw = statement + "||" + "|".join(chunk_ids) + "||" + GEMINI_MODEL
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def classify_support(statement, retrieved_chunks, max_retries=5, use_cache=True):
    cache_path = CACHE_DIR / f"{_cache_key(statement, retrieved_chunks)}.json"

    if use_cache and cache_path.exists():
        return StatementVerdict.model_validate_json(cache_path.read_text(encoding="utf-8"))

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
            client = _get_client()
            response = client.models.generate_content(
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
            verdict = StatementVerdict(
                statement=statement,
                verdict=Verdict(result["verdict"]),
                confidence=result["confidence"],
                citations=result["citations"],
                reasoning=result["reasoning"],
            )
            if use_cache:
                cache_path.write_text(verdict.model_dump_json(), encoding="utf-8")
            return verdict
        except Exception as e:
            last_error = e
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                wait = 2 * (attempt + 1)
            else:
                wait = 2 ** attempt
            time.sleep(wait)

    raise last_error