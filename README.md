# BiblAI
### Claim Verification Engine with Agentic RAG and LangGraph

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Orchestration-FF6F00?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Database-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-LLM_%26_Structured_Output-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## Overview

BiblAI is a fact-checking and claim verification system that evaluates natural language assertions against the *Biblia Tysiąclecia* text corpus. 

The project addresses the vocabulary mismatch between modern user queries and historical religious texts using an agentic retrieval pipeline. It combines **Hypothetical Document Embeddings (HyDE)**, **dense vector search in Qdrant**, **cross-encoder reranking**, and a **LangGraph state machine** that automatically broadens search queries when initial retrieval confidence is below threshold.

---

## Technical Highlights

- **LangGraph State Graph with Feedback Loop**: Implements a cyclical graph with `expand_query`, `retrieve_and_rerank`, and `classify` nodes. If top-ranked candidate relevance scores fall below a configurable threshold, the graph loops back with a broader search scope before final classification.
- **Domain-Specific Query Expansion (HyDE)**: Uses Gemini 3.5 Flash to expand modern or abstract concepts into 3-5 hypothetical archaic/biblical phrasing variants before computing embeddings, significantly improving semantic retrieval recall.
- **Two-Stage Retrieval & Reranking**: Queries an embedded Qdrant vector database using `paraphrase-multilingual-mpnet-base-v2` dense vectors. Retrieved candidates are deduplicated and can be reranked via a multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- **Structured Pydantic Outputs**: Enforces JSON Schema constraints on LLM responses to extract structured verdicts (`directly_supported`, `directly_contradicted`, `not_directly_stated`), confidence scores, and verse citations with exact quotes and relation tags.
- **Evaluation Framework (R&D)**: Includes automated benchmarks for retrieval recall (Recall@K), verdict accuracy across categorized test sets, and an independent **LLM-as-a-Judge** script for citation grounding and reasoning verification.
- **FastAPI Service & Caching**: Asynchronous backend with request validation, static file serving, and SHA-256 disk caching for LLM calls during local testing and evaluations.

---

## Pipeline Architecture

```mermaid
flowchart TD
    A[User Statement] --> B[expand_query: HyDE Prompting]
    B -->|Generates 3-5 biblical variants| C[multi_retrieve: Qdrant Vector DB]
    C -->|Top-N Candidate Chunks| D[rerank: Cross-Encoder]
    D --> E{check_relevance}
    E -- Score < Threshold & Attempts < 2 -->|Broaden Search| C
    E -- Score >= Threshold --> F[classify: LLM Reasoning]
    F -->|Structured JSON| G[StatementVerdict]
    G --> H[FastAPI Endpoint & Web UI]
```

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Agent Orchestration** | LangGraph, StateGraph |
| **LLM & Inference** | Google Gemini 3.5 Flash (`google-genai`), Structured JSON Schema |
| **Vector Database** | Qdrant (Embedded local storage) |
| **Embeddings** | Sentence-Transformers (`paraphrase-multilingual-mpnet-base-v2`) |
| **Reranking** | Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`) |
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
        ├── agent/              # LangGraph workflow and tools
        │   ├── graph.py        # StateGraph, nodes, and conditional edges
        │   ├── tools.py        # HyDE expansion, Qdrant retrieval, and Gemini API calls
        │   └── schemas.py      # Pydantic models (StatementVerdict, Citation, Verdict)
        ├── api/
        │   └── main.py         # FastAPI routes and middleware
        ├── embedding/
        │   └── embed_and_store.py # Scripture chunking and vector index generation
        └── ingestion/
            ├── parser.py       # Raw text parser
            ├── chunker.py      # Sliding-window scripture chunker
            └── book_mapping.py # Book abbreviations and metadata mapping
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
- **RAG & Information Retrieval**: Handling cross-domain vocabulary mismatch with HyDE, multi-query retrieval in Qdrant, and cross-encoder reranking.
- **Structured LLM Generation**: Pydantic schema validation and JSON mode configuration with zero hallucinated references.
- **Evaluation & Benchmarking**: Building custom test datasets, computing retrieval recall metrics, and setting up automated LLM-as-a-judge pipelines.
- **Production Engineering**: FastAPI web service design, disk-level caching strategies, and Docker deployment.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

