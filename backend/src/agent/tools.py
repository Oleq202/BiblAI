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
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[WARN] GEMINI_API_KEY is not set in environment!", flush=True)
        _client = genai.Client(api_key=api_key)
    return _client


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
    embed_model = _get_embed_model()
    import torch
    with torch.inference_mode():
        return embed_model.encode([query], show_progress_bar=False).tolist()[0]


def _get_embeddings(queries: list[str]) -> list[list[float]]:
    embed_model = _get_embed_model()
    import torch
    with torch.inference_mode():
        return embed_model.encode(queries, show_progress_bar=False).tolist()


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


def multi_retrieve(queries: list[str], top_k_per_query: int = 15, return_scores: bool = False):
    if not queries:
        return ([], []) if return_scores else []
    
    embeddings = _get_embeddings(queries)
    qdrant = _get_qdrant()
    
    chunk_map = {}
    for q_text, emb in zip(queries, embeddings):
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=emb,
            limit=top_k_per_query,
        ).points
        for point in results:
            cid = point.payload["chunk_id"]
            if cid not in chunk_map or point.score > chunk_map[cid][1]:
                chunk_map[cid] = (point.payload, point.score)
                
    sorted_items = sorted(chunk_map.values(), key=lambda x: x[1], reverse=True)
    payloads = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    if return_scores:
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


EXPANSION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista 2-4 wariantów fraz lub hipotetycznych wersetów w języku Biblii Tysiąclecia.",
        }
    },
    "required": ["queries"],
}

EXPANSION_SYSTEM_PROMPT = """Jesteś ekspertem tekstu, stylistyki i języka Biblii Tysiąclecia.
Twoim zadaniem jest przekształcenie współczesnego lub teologicznego stwierdzenia użytkownika \
na 3-5 pełnych, hipotetycznych wersetów lub zdań biblijnych w języku Biblii Tysiąclecia (technika HyDE - Hypothetical Document Embeddings).

Zasady:
1. Pismo Święte nie używa współczesnych pojęć abstrakcyjnych, lecz konkretnego, starożytnego języka opisowego.
   (np. zamiast 'homoseksualizm' -> 'Nie będziesz obcował z mężczyzną tak jak się obcuje z kobietą, to jest obrzydliwość', 'Mężczyźni współżyjący ze sobą ani rozpustnicy nie odziedziczą królestwa Bożego', 'Mężczyźni porzuciwszy współżycie z kobietą zapałali żądzą ku sobie, mężczyźni z mężczyznami uprawiając bezwstyd';
   zamiast 'eutanazja' -> 'Nie będziesz zabijał niewinnego ani cierpiącego', 'Bóg daje życie i Bóg je odbiera';
   zamiast 'aborcja' -> 'Zanim ukształtowałem cię w łonie matki, znałem cię', 'Kto uderzy kobietę brzemienną, tak że poroni').
2. Wygeneruj 3-5 pełnych zdań / wersetów w języku polskim w stylu Biblii Tysiąclecia (zbieżnych z prawem Starego Testamentu, mowami proroków oraz listami Nowego Testamentu).
3. Zwróć wynik w zadanym schemacie JSON."""


def _expansion_cache_key(statement: str) -> str:
    raw = "expansion||" + statement.strip().lower() + "||" + GEMINI_MODEL
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def expand_query(statement: str, max_retries=3, use_cache=True) -> list[str]:
    cache_path = CACHE_DIR / f"{_expansion_cache_key(statement)}.json"
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"[expand_query] Loaded from cache ({len(cached)} queries)", flush=True)
            return cached
        except Exception as e:
            print(f"[expand_query] Cache read error: {e}", flush=True)

    queries = [statement]
    for attempt in range(max_retries):
        try:
            client = _get_client()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"Stwierdzenie użytkownika: \"{statement}\"",
                config=types.GenerateContentConfig(
                    system_instruction=EXPANSION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=EXPANSION_RESPONSE_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            result = json.loads(response.text)
            gen_queries = result.get("queries", [])
            if gen_queries:
                seen = set()
                combined = []
                for q in [statement] + gen_queries:
                    clean_q = q.strip().strip('"\'')
                    if clean_q and clean_q.lower() not in seen:
                        seen.add(clean_q.lower())
                        combined.append(clean_q)
                queries = combined
            print(f"[expand_query] Generated {len(queries)} queries for: '{statement}'", flush=True)
            if use_cache:
                try:
                    cache_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"[expand_query] Cache write error: {e}", flush=True)
            return queries
        except Exception as e:
            print(f"[expand_query] Attempt {attempt+1} failed: {e}", flush=True)
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                wait = 2 * (attempt + 1)
            else:
                wait = 2 ** attempt
            time.sleep(wait)

    print(f"[expand_query] Fallback to original query: {statement}", flush=True)
    return queries



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

SYSTEM_PROMPT = """Jesteś asystentem weryfikującym zgodność stwierdzeń ze wskazanymi fragmentami Pisma Świętego (Biblia Tysiąclecia).

Zasady klasyfikacji:
- "directly_supported": Tekst wprost lub poprzez bezpośrednie synonimy / opisowe sformułowania biblijne stwierdza to samo, co użytkownik. Pismo Święte używa języka starożytnego i opisowego (np. zakaz/potępienie 'obcowania z mężczyzną jak z kobietą' czy 'mężczyzn współżyjących ze sobą' jako obrzydliwości bezpośrednio popiera stwierdzenie 'homoseksualizm jest grzechem'). Jeśli treść biblijna wyraża dokładnie tę samą normę moralną, zakaz lub fakt, uznaj to za bezpośrednie poparcie ("directly_supported").
- "directly_contradicted": Tekst wprost lub opisowo stwierdza coś przeciwnego do twierdzenia użytkownika.
- "not_directly_stated": Żaden dostarczony fragment nie odnosi się do tego zagadnienia, temat jest całkowicie nieobecny lub wymaga daleko idących, niejednoznacznych spekulacji.
- Cytuj TYLKO fragmenty rzeczywiście obecne w podanym kontekście. Nigdy nie wymyślaj wersetów ani cytatów.
- Jeśli odpowiedź wymaga fragmentu, którego NIE MA w podanym kontekście, po prostu stwierdź, że dostarczony kontekst nie zawiera odpowiedzi."""


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
                    temperature=0.0,
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