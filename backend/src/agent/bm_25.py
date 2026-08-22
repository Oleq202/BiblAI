import gzip
import importlib.resources as pkg_resources
import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
BM25_CACHE_PATH = BASE_DIR / "data" / "bm25_index.pkl"

_bm25_index = None
_bm25_chunks = None
_stemmer = None

POLISH_STOPWORDS = {
    "a",
    "aby",
    "ach",
    "acz",
    "aczkolwiek",
    "aj",
    "albo",
    "ale",
    "ani",
    "aż",
    "bardziej",
    "bardzo",
    "bo",
    "bowiem",
    "by",
    "byli",
    "być",
    "był",
    "była",
    "było",
    "były",
    "będzie",
    "będą",
    "cali",
    "cała",
    "cały",
    "ci",
    "cię",
    "ciebie",
    "co",
    "cokolwiek",
    "coś",
    "czasami",
    "czasem",
    "czemu",
    "czy",
    "czyli",
    "dla",
    "dlaczego",
    "dlatego",
    "do",
    "dobrze",
    "dokąd",
    "dość",
    "dużo",
    "dwa",
    "dwaj",
    "dwie",
    "dwoje",
    "dziś",
    "dzisiaj",
    "gdy",
    "gdyby",
    "gdyż",
    "gdzie",
    "gdziekolwiek",
    "gdzieś",
    "i",
    "ich",
    "ile",
    "im",
    "inna",
    "inne",
    "inny",
    "innych",
    "iż",
    "ja",
    "ją",
    "jak",
    "jakaś",
    "jakby",
    "jaki",
    "jakich",
    "jakie",
    "jakiś",
    "jako",
    "jakoś",
    "je",
    "jeden",
    "jedna",
    "jedno",
    "jednak",
    "jednakże",
    "jego",
    "jej",
    "jemu",
    "jest",
    "jestem",
    "jesteście",
    "jesteśmy",
    "jeśli",
    "jeżeli",
    "już",
    "każdy",
    "kiedy",
    "kilka",
    "kim",
    "kto",
    "ktokolwiek",
    "ktoś",
    "która",
    "które",
    "którego",
    "której",
    "który",
    "których",
    "którym",
    "którzy",
    "ku",
    "lub",
    "ma",
    "mają",
    "mam",
    "mamy",
    "mało",
    "mną",
    "mnie",
    "mój",
    "moim",
    "moja",
    "moje",
    "może",
    "możliwe",
    "można",
    "musi",
    "mu",
    "my",
    "na",
    "nad",
    "nam",
    "nami",
    "nas",
    "nasi",
    "nasz",
    "nasza",
    "nasze",
    "naszego",
    "naszych",
    "nawet",
    "nic",
    "nich",
    "nie",
    "niego",
    "niej",
    "niemu",
    "nigdy",
    "nim",
    "nimi",
    "niż",
    "no",
    "o",
    "obok",
    "od",
    "około",
    "on",
    "ona",
    "one",
    "oni",
    "ono",
    "oraz",
    "oto",
    "owszem",
    "pan",
    "pana",
    "pani",
    "po",
    "pod",
    "podczas",
    "pomimo",
    "ponad",
    "ponieważ",
    "powinien",
    "powinna",
    "prawie",
    "przecież",
    "przed",
    "przede",
    "przez",
    "przy",
    "roku",
    "również",
    "sam",
    "sama",
    "są",
    "się",
    "skąd",
    "sobie",
    "sobą",
    "sposób",
    "swoje",
    "ta",
    "tak",
    "taka",
    "taki",
    "takie",
    "także",
    "tam",
    "te",
    "tego",
    "tej",
    "ten",
    "teraz",
    "też",
    "to",
    "tobie",
    "tobą",
    "toteż",
    "totobą",
    "trzeba",
    "tu",
    "tutaj",
    "twoim",
    "twoja",
    "twoje",
    "twój",
    "twym",
    "ty",
    "tych",
    "tylko",
    "tym",
    "u",
    "w",
    "wam",
    "wami",
    "was",
    "wasz",
    "wasza",
    "wasze",
    "we",
    "według",
    "wraz",
    "właśnie",
    "wtedy",
    "wy",
    "z",
    "za",
    "zawsze",
    "zaś",
    "ze",
    "znowu",
    "znów",
    "został",
}


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from pystempel import Stemmer
            from pystempel.data import polimorf as polimorf_pkg
            from pystempel.streams import DataInputStream

            resource = pkg_resources.files(polimorf_pkg).joinpath("stemmer_polimorf.tbl.gz")
            with resource.open("rb") as raw_f:
                with gzip.open(raw_f, "rb") as gz_f:
                    _stemmer = Stemmer.from_stream(DataInputStream(gz_f, None))
        except Exception:
            try:
                from pystempel import Stemmer

                _stemmer = Stemmer.polimorf()
            except Exception:
                _stemmer = None
    return _stemmer


def tokenize_text(text):
    words = re.findall(r"\b\w+\b", text.lower())
    stemmer = _get_stemmer()
    tokens = []
    for token in words:
        if len(token) <= 1 or token in POLISH_STOPWORDS:
            continue
        tokens.append(token)
        if stemmer is not None:
            lemma = stemmer(token)
            if lemma and lemma != token and lemma not in POLISH_STOPWORDS:
                tokens.append(lemma)
    return tokens


def get_bm25_index():
    global _bm25_index, _bm25_chunks

    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    if BM25_CACHE_PATH.exists():
        try:
            with open(BM25_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
                _bm25_index = data["index"]
                _bm25_chunks = data["chunks"]
                print(f"[BM25] Loaded index from cache ({len(_bm25_chunks)} chunks)", flush=True)
                return _bm25_index, _bm25_chunks
        except Exception as e:
            print(f"[BM25] Cache load failed: {e}, rebuilding index...", flush=True)

    print(f"[BM25] Building lemmatized index from {CHUNKS_PATH}...", flush=True)
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    tokenized_corpus = [tokenize_text(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_chunks = chunks

    try:
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump({"index": _bm25_index, "chunks": _bm25_chunks}, f)
        print(f"[BM25] Saved index cache to {BM25_CACHE_PATH}", flush=True)
    except Exception as e:
        print(f"[BM25] Could not save cache: {e}", flush=True)
    return _bm25_index, _bm25_chunks


def bm25_retrieve(query, top_k=15):
    index, chunks = get_bm25_index()
    tokens = tokenize_text(query)
    if not tokens:
        return []

    scores = index.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0.0]


def multi_bm25_retrieve(queries, top_k_per_query=15):
    chunk_map = {}
    for q in queries:
        results = bm25_retrieve(q, top_k=top_k_per_query)
        for chunk, score in results:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in chunk_map or score > chunk_map[chunk_id][1]:
                chunk_map[chunk_id] = (chunk, score)

    sorted_items = sorted(chunk_map.values(), key=lambda x: x[1], reverse=True)
    return sorted_items
