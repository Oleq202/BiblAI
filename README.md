# BiblAI
### Claim Verification Engine with Agentic RAG and LangGraph

[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Vector DB](https://img.shields.io/badge/Vector_DB-Qdrant-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Orchestration](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LLM](https://img.shields.io/badge/LLM-Gemini_3.5_Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Deployment](https://img.shields.io/badge/Deployment-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## Overview

BiblAI is a fact-checking and claim verification system that evaluates natural language assertions against the *Biblia Tysiąclecia* text corpus. 

The project addresses the vocabulary mismatch between modern user queries and historical religious texts using an agentic hybrid retrieval pipeline. It combines **Hypothetical Document Embeddings (HyDE)**, **Hybrid Search (Dense vector search in Qdrant + Sparse BM25 retrieval)**, **Reciprocal Rank Fusion (RRF)**, **cross-encoder reranking**, and a **LangGraph state machine** that automatically broadens search queries when initial retrieval confidence is below threshold.

---

## Technical Highlights

- **LangGraph State Graph with Feedback Loop**: Implements a cyclical graph with `expand_query`, `retrieve_and_rerank`, and `classify` nodes. If top-ranked candidate relevance scores fall below a configurable threshold, the graph loops back with a broader search scope before final classification.
- **Domain-Specific Query Expansion (HyDE)**: Uses Gemini 3.5 Flash to expand modern or abstract concepts into 3-5 hypothetical archaic/biblical phrasing variants before retrieval, significantly improving both semantic and keyword matching.
- **Hybrid Retrieval (Dense + Sparse BM25)**: Executes parallel retrieval across all expanded queries:
  - **Dense Path**: Cosine similarity search over embedded Qdrant vectors (`paraphrase-multilingual-mpnet-base-v2`).
  - **Sparse Path**: BM25 Okapi search with Polish morphological lemmatization (PoliMorf dictionary via `pystempel`) and stopword filtering for inflection-aware keyword and proper noun matching.
- **Reciprocal Rank Fusion (RRF)**: Merges ranked candidate lists from dense and sparse retrieval using reciprocal rank weighting ($RRF(d) = \sum \frac{w}{60 + \text{rank}(d)}$) to yield a balanced candidate pool.
- **Two-Stage Reranking**: Fused candidates can be further reranked via a multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- **Structured Pydantic Outputs**: Enforces JSON Schema constraints on LLM responses to extract structured verdicts (`directly_supported`, `directly_contradicted`, `not_directly_stated`), confidence scores, and verse citations with exact quotes and relation tags.
- **Evaluation Framework (R&D)**: Includes automated benchmarks for retrieval recall (Recall@K), verdict accuracy across categorized test sets, and an independent **LLM-as-a-Judge** script for citation grounding and reasoning verification.
- **FastAPI Service & Caching**: Asynchronous backend with request validation, static file serving, and SHA-256 disk caching for LLM calls and BM25 index serialization for instantaneous startup.

---

## Pipeline Architecture

```text
                          [ User Statement ]
                                  │
                                  ▼
                      [ expand_query (HyDE) ]
                    (3-5 biblical query variants)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
      [ Dense Path (Qdrant) ]           [ Sparse Path (BM25) ]
     (Vector similarity search)        (BM25Okapi + Polish tokens)
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                    [ Reciprocal Rank Fusion (RRF) ] ◄────────┐
                      (Fused top candidate pool)               │
                                  │                            │
                                  ▼                            │ Low score
                     [ rerank (Cross-Encoder) ]                │ (Attempts < 2)
                                  │                            │
                                  ▼                            │
                         { check_relevance } ──────────────────┘
                                  │
                                  │ Score >= Threshold
                                  ▼
                       [ classify (LLM Reasoner) ]
                     (Gemini 3.5 Flash + JSON Schema)
                                  │
                                  ▼
                         [ StatementVerdict ]
                   (Verdict, Confidence, Citations)
                                  │
                                  ▼
                     [ FastAPI Endpoint / Web UI ]
```

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Agent Orchestration** | LangGraph, StateGraph |
| **LLM & Inference** | Google Gemini 3.5 Flash (`google-genai`), Structured JSON Schema |
| **Vector Database (Dense)** | Qdrant (Embedded local storage) |
| **Sparse Retrieval** | BM25 Okapi (`rank-bm25`), Polish morphological lemmatization (PoliMorf via `pystempel`) |
| **Fusion Algorithm** | Reciprocal Rank Fusion (RRF, $k=60$) |
| **Embeddings** | Sentence-Transformers (`paraphrase-multilingual-mpnet-base-v2`) |
| **Reranking** | Cross-Encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) |
| **Backend & API** | FastAPI, Uvicorn, Pydantic v2, Python 3.11 |
| **Frontend** | Vanilla JavaScript, HTML5, CSS3 |
| **Evaluation** | LLM-as-a-Judge (`gemini-3.1-flash-lite`), Recall@K metrics |
| **Deployment** | Docker, CPU-optimized PyTorch build |

---

## Evaluation & Benchmarks

The project includes an evaluation suite under [`backend/eval/`](file:///backend/eval):

### 1. Retrieval Recall (`retrieval_eval.py`)
Evaluates whether target verse references appear in candidate chunks retrieved by bi-encoder search vs. cross-encoder reranking over a curated test set.

### 2. Verdict Accuracy (`run_eval.py`)
Runs the full verification pipeline over test statements categorized into:
- `supported_explicit`: Direct factual statements clearly present in scripture.
- `contradicted`: Inverted claims, wrong attributions, or chronology errors.
- `not_directly_stated`: Concepts absent from canonical scripture.
- `obscure_facts`: Fine-grained narrative details.
- `adversarial_trick`: Semantically altered statements designed to test naive keyword matching.

### 3. LLM-as-a-Judge (`llm_judge.py`)
Uses an independent LLM prompt and schema to evaluate:
- Whether the system's verdict strictly follows from the retrieved citations.
- Whether the generated reasoning introduces ungrounded external assumptions.

---

## Project Structure

```text
BiblAI/
├── Dockerfile                  # Docker container definition
├── requirements.txt            # Python dependencies
├── frontend/
│   ├── index.html              # Web interface
│   └── public/                 # Static assets and icons
└── backend/
    ├── data/                   # Parsed chunks, Qdrant index, and cache
    ├── eval/                   # Evaluation suite
    │   ├── eval_set.jsonl      # Labeled test queries and expected verdicts
    │   ├── retrieval_eval.py   # Retrieval recall benchmark script
    │   ├── run_eval.py         # End-to-end classification test runner
    │   ├── llm_judge.py        # LLM-as-a-judge validation script
    │   ├── citation_check.py   # Citation verification script
    │   └── results/            # Benchmark outputs (JSONL)
    └── src/
        ├── ingestion/
        │   ├── parser.py       # Raw text parser
        │   ├── chunker.py      # Sliding-window scripture chunker
        │   └── book_mapping.py # Book abbreviations and metadata mapping
        ├── embedding/
        │   └── embed_and_store.py # Scripture chunking and vector index generation
        ├── agent/              # LangGraph workflow and tools
        │   ├── graph.py        # StateGraph, nodes, and conditional edges
        │   ├── bm_25.py        # BM25Okapi sparse search and Polish text tokenization
        │   ├── tools.py        # HyDE expansion, Hybrid retrieval (Qdrant + BM25 + RRF), and Gemini API calls
        │   └── schemas.py      # Pydantic models (StatementVerdict, Citation, Verdict)
        └── api/
            └── main.py         # FastAPI routes and middleware
```

---

## API Reference

### `POST /verify`
Evaluates a statement against the corpus.

**Request:**
```json
{
  "statement": "Mojżesz wyprowadził Izraelitów z Egiptu."
}
```

**Response (`200 OK`):**
```json
{
  "statement": "Mojżesz wyprowadził Izraelitów z Egiptu.",
  "verdict": "directly_supported",
  "confidence": 1.0,
  "citations": [
    {
      "ref": "Wj 3:10",
      "quote": "Idź przeto teraz, oto posyłam cię do faraona, i wyprowadź mój lud, Izraelitów, z Egiptu",
      "relation": "supports"
    },
    {
      "ref": "Lb 33:1",
      "quote": "Oto miejsca postoju Izraelitów, którzy swoimi oddziałami wojskowymi wyszli z Egiptu pod wodzą Mojżesza i Aarona.",
      "relation": "supports"
    }
  ],
  "reasoning": "Fragmenty z Księgi Wyjścia i Księgi Liczb bezpośrednio wskazują, że Bóg nakazał Mojżeszowi wyprowadzić Izraelitów z Egiptu oraz że wyszli oni pod wodzą Mojżesza."
}
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Google Gemini API key

### 1. Clone Repository
```bash
git clone https://github.com/Oleq202/BiblAI.git
cd BiblAI
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate Virtual Environment
Linux / macOS:
```bash
source .venv/bin/activate
```

Windows:
```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the project root or `backend/` directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
USE_CROSS_ENCODER=false
```

### 6. Run Development Server
```bash
python -m uvicorn backend.src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## Docker Deployment

Build image:
```bash
docker build -t biblai:latest .
```

Run container:
```bash
docker run -p 8080:8080 -e GEMINI_API_KEY="your_api_key_here" biblai:latest
```

---

## Running Evaluations

Retrieval recall benchmark:
```bash
python backend/eval/retrieval_eval.py
```

End-to-end verdict evaluation:
```bash
python backend/eval/run_eval.py
```

LLM-as-a-judge validation:
```bash
python backend/eval/llm_judge.py
```

Citation grounding check:
```bash
python backend/eval/citation_check.py
```

---

## Technical Competencies Demonstrated

- **Agentic Workflows**: Stateful graph design with conditional branching and fallback loops using LangGraph.
- **RAG & Hybrid Information Retrieval**: Handling cross-domain vocabulary mismatch with HyDE, parallel Dense (Qdrant) + Sparse (BM25) multi-query retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder reranking.
- **Structured LLM Generation**: Pydantic schema validation and JSON mode configuration with zero hallucinated references.
- **Evaluation & Benchmarking**: Building custom test datasets, computing retrieval recall metrics, and setting up automated LLM-as-a-judge pipelines.
- **Production Engineering**: FastAPI web service design, disk-level caching strategies (LLM + BM25 serialization), and Docker deployment.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

