## RAG AI Research Assistant – End‑to‑End Implementation Plan (Local Notes)

> This file is for **your local planning only**. It is listed in `.gitignore` and will not be committed.

---

## Phase 0 – Repo + Environment Hygiene

**Goal:** One clean project, reproducible environment, no path/import confusion.

- **0.1 – Repo structure**
  - Ensure root layout:
    - `rag_arxiv_qa/` (code, data, notebooks, tests)
    - `config/` (`config.yaml`, `dev.yaml`, `prod.yaml`)
    - `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `README.md`
  - Confirm Python package structure:
    - `rag_arxiv_qa/__init__.py`
    - `rag_arxiv_qa/src/__init__.py`
    - `rag_arxiv_qa/src/utils/__init__.py`

- **0.2 – Virtualenv + dependencies**
  - Create / use venv: `python -m venv .venv`
  - Activate and install: `pip install -r requirements.txt`
  - Sanity checks:
    - `python -c "from rag_arxiv_qa.src.utils.config import load_config; print(load_config()['embeddings']['model'])"`
    - `python -c "import langchain, langgraph, chromadb, weaviate, mlflow, ragas; print('Core imports OK')"`

- **0.3 – Baseline tests (even if empty)**
  - Run `pytest -q` to ensure no discovery/import errors.
  - Verify `rag_arxiv_qa/tests/test_retrieval.py` runs once implemented later.

---

## Phase 1 – Configuration & Settings

**Goal:** Central, environment‑aware configuration for all components.

- **1.1 – Config schema**
  - Define the top‑level sections in `config/config.yaml`:
    - `paths`: raw/processed data, vector DB directories.
    - `ingestion`: query string, categories, max results, date range.
    - `chunking`: strategy, chunk_size, chunk_overlap, splitting rules.
    - `embeddings`: model_name, batch_size, device, normalize_embeddings.
    - `retrieval`: top_k, score_threshold, use_faiss, filters.
    - `rerank`: model_name, top_k_rerank.
    - `generation`: model, endpoint, max_tokens, temperature, system_prompt.
    - `evaluation`: metrics to run, eval dataset paths.

- **1.2 – Config loader (`rag_arxiv_qa/src/utils/config.py`)**
  - Implement:
    - `load_config(env: str | None = None) -> dict`
      - Decide env (`ENV` or `APP_ENV` or `RAG_ENV`, default `dev`).
      - Load base `config.yaml`.
      - If `dev.yaml`/`prod.yaml` present, deep‑merge overrides.
  - Provide helper:
    - `get_path(name: str) -> Path` to resolve paths from config.

- **1.3 – Environment handling**
  - Use `.env` / `.env.example` for secrets (API keys, model endpoints).
  - In `config.py`, load `.env` via `python-dotenv` if present.

Deliverable: **Any module (ingestion, chunking, embeddings, retrieval, API) reads from `load_config()` only, no hard‑coded paths/models.**

---

## Phase 2 – Data Ingestion (ArXiv → Local Corpus)

**Goal:** Stable pipeline to fetch/save AI/ML arXiv papers as a structured corpus.

- **2.1 – Design ingestion contract**
  - Input:
    - Search query, categories, date range, max_papers from config.
  - Output:
    - Parquet/JSONL file with columns:
      - `paper_id`, `title`, `abstract`, `authors`, `categories`,
      - `published_at`, `updated_at`, `url_pdf`, `source="arxiv"`,
      - `full_text` (when available) or `abstract_only` flag.

- **2.2 – Implement loader (`rag_arxiv_qa/src/ingestion/load_arxiv.py`)**
  - Functions:
    - `fetch_metadata(query, max_results, ...) -> list[dict]`
    - `download_pdfs_if_needed(...)` (optional for full‑text).
    - `build_corpus_dataframe(...) -> pd.DataFrame`
    - `save_corpus(df, path: Path) -> None`
  - Validate:
    - No duplicate `paper_id`.
    - Missing full text handled gracefully (`abstract_only=True`).

- **2.3 – Notebook integration (`notebooks/01_ingestion/arxiv_ingestion.ipynb`)**
  - Use the ingestion module, not ad‑hoc code.
  - Ensure notebook only orchestrates; all logic lives in `src/ingestion`.

Deliverable: **A reproducible corpus file in `rag_arxiv_qa/data/processed/` referenced in config.**

---

## Phase 3 – Chunking

**Goal:** Convert raw papers into semantically meaningful, retrievable chunks.

- **3.1 – Chunking strategy**
  - Choose approach (config‑driven):
    - Text normalization: strip HTML/LaTeX, normalize whitespace.
    - Logical splitting:
      - By sections (Introduction, Methods, Results, Conclusion) when parseable.
      - Fallback: sentence/paragraph splitting + fixed token/char windows.
  - Decide:
    - `chunk_size` (e.g., 512–1024 tokens).
    - `chunk_overlap` (e.g., 128–256 tokens).

- **3.2 – Implement chunker (`rag_arxiv_qa/src/chunking/chunker.py`)**
  - Key functions:
    - `load_corpus(path) -> DataFrame`
    - `split_document(row) -> list[dict]` of chunks with metadata.
    - `chunk_corpus(df) -> DataFrame` with columns:
      - `chunk_id`, `paper_id`, `title`, `section`, `text`, `position`, `n_tokens`.
    - `save_chunks(df, path)`
  - Respect configuration:
    - Use values from `config["chunking"]`.

- **3.3 – Notebook integration (`notebooks/02_chunking/chunking.ipynb`)**
  - Drive the chunker on the ingested corpus.
  - Explore distributions:
    - Chunks per paper.
    - Token length per chunk.

Deliverable: **A chunks parquet file in `rag_arxiv_qa/data/processed/` feeding embeddings.**

---

## Phase 4 – Embeddings + Indexing (Vector Store)

**Goal:** Persisted vector store (Chroma) built from chunks with rich metadata.

- **4.1 – Embedding models (`rag_arxiv_qa/src/embeddings/embedder.py`)**
  - Implement an `Embedder` class:
    - Config‑driven model name (e.g., `BAAI/bge-base-en-v1.5`).
    - Methods:
      - `embed_documents(texts: list[str]) -> np.ndarray`
      - `embed_query(text: str) -> np.ndarray`
    - Use `sentence-transformers` with batching and device selection.

- **4.2 – Vector DB integration (`rag_arxiv_qa/src/indexing/faiss_index.py` or better: Chroma wrapper)**
  - Design an `Indexer` / `VectorStore` interface:
    - `index_chunks(chunks_df) -> None`
    - `query(query_text, top_k) -> list[ScoredChunk]`
  - For Chroma:
    - Collection name from config.
    - Store metadata: `paper_id`, `title`, `section`, `position`, `source`.
    - Ensure idempotent builds (drop/rebuild or upsert semantics).

- **4.3 – Index build script / entrypoint**
  - CLI or Python entrypoint:
    - Load chunks.
    - Embed with `Embedder`.
    - Upsert into Chroma.

- **4.4 – Notebook integration (`notebooks/03_embeddings/embeddings.ipynb`)**
  - Use the embedder + indexer modules.
  - Visualize:
    - Example nearest neighbors.
    - Sanity‑check of semantic similarity.

Deliverable: **Chroma collection (on disk) ready for query‑time retrieval.**

---

## Phase 5 – Retrieval + Re‑ranking

**Goal:** High‑quality context selection for the LLM.

- **5.1 – Retriever (`rag_arxiv_qa/src/retrieval/retriever.py`)**
  - Implement:
    - `retrieve(query: str, top_k: int) -> list[ScoredChunk]`
  - Use:
    - Embed query with `Embedder`.
    - Perform vector search in Chroma.
    - Return chunks + scores + metadata.

- **5.2 – Re‑ranker (optional new module, e.g., `reranker.py`)**
  - Use cross‑encoder (e.g., `BAAI/bge-reranker-base` or MiniLM).
  - API:
    - `rerank(query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]`
  - Logic:
    - Given dense retrieval top_k (e.g., 50), score with cross‑encoder, keep top 5–10.

- **5.3 – Retrieval pipeline**
  - Implement in retrieval module:
    - `retrieve_with_rerank(query: str) -> list[ScoredChunk]`
    - Steps:
      1. Dense retrieval.
      2. Optional filters (date, category).
      3. Re‑ranking.
      4. Return final ordered context.

Deliverable: **One function you can call from anywhere to get high‑quality context chunks for a query.**

---

## Phase 6 – Generation (LLM + Prompting)

**Goal:** Grounded answers with citations using the retrieved context.

- **6.1 – Prompt design (`rag_arxiv_qa/src/generation/prompt.py`)**
  - Define templates:
    - System prompt: role, constraints, citation format.
    - User prompt with:
      - Query.
      - Serialized context chunks with paper ids/titles.
    - Emphasize:
      - Answer strictly based on context.
      - Include citations like `[Paper: <title>, id: <paper_id>]`.

- **6.2 – Generator abstraction (`rag_arxiv_qa/src/generation/generator.py`)**
  - Interface:
    - `generate_answer(query: str, context_chunks: list[ScoredChunk]) -> dict`
      - Returns: `{ "answer": str, "citations": list[dict], "raw_response": Any }`
  - Implementation variants:
    - Local open‑source model via `transformers` (baseline).
    - Remote / vLLM endpoint (production‑like).

- **6.3 – Inference backends (`rag_arxiv_qa/src/inference/`)**
  - `tgi/serve.py`, `vllm/serve.py`:
    - Provide HTTP endpoints for text‑generation.
    - Configurable model, max_tokens, temperature.
  - The generator calls these endpoints via a thin client.

Deliverable: **A single `generate_answer` entrypoint that wires query → retrieval → generation given a context list.**

---

## Phase 7 – Orchestration with LangGraph

**Goal:** Move from a linear script to a maintainable, debuggable graph.

- **7.1 – Define nodes**
  - Nodes:
    - `parse_query`
    - `retrieve` (dense)
    - `rerank` (optional)
    - `generate`
    - `postprocess` (e.g., format answer + citations)
    - (later) `retry_or_reformulate`

- **7.2 – Implement LangGraph graph**
  - Graph type:
    - Simple DAG for now (no loops yet).
  - Edges:
    - `parse_query -> retrieve -> rerank -> generate -> postprocess`

- **7.3 – Expose a single orchestrator function**
  - e.g., `rag_pipeline.run(query: str) -> dict`
  - This is what the FastAPI route will call.

Deliverable: **A graph‑based pipeline that encapsulates the full RAG workflow.**

---

## Phase 8 – Evaluation & Guardrails

**Goal:** Quantify performance and catch hallucinations.

- **8.1 – Evaluation dataset**
  - Create a small labeled set:
    - Each item: `{ "question", "expected_answer", "paper_id(s)_expected" }`
  - Store in `rag_arxiv_qa/data/processed/eval_set.jsonl` (or parquet).

- **8.2 – RAGAS integration (`rag_arxiv_qa/src/evaluation/eval_basic.py`)**
  - Use RAGAS metrics:
    - Context recall.
    - Faithfulness.
    - Answer similarity.
  - Pipeline:
    - For each eval query:
      1. Run full RAG pipeline.
      2. Pass question, context, answer, ground truth to RAGAS.
      3. Aggregate metrics into a DataFrame.

- **8.3 – Tracking with MLflow**
  - Log:
    - Metric averages (recall, faithfulness, etc.).
    - Model/config hash (embedding model, chunk size, top_k, rerank model).
    - Git commit hash.
  - This lets you compare configurations over time.

Deliverable: **A repeatable eval script that logs RAGAS metrics + config to MLflow.**

---

## Phase 9 – API, UI, Deployment

**Goal:** Production‑style interface to serve the RAG system.

- **9.1 – FastAPI layer (`rag_arxiv_qa/src/api/main.py`, `routes.py`, `schemas.py`)**
  - Define Pydantic schemas:
    - `QueryRequest { query: str }`
    - `QueryResponse { answer: str, citations: list[...], metadata: ... }`
  - Route:
    - `POST /query`:
      - Calls LangGraph RAG pipeline.
      - Returns structured response.
  - Add:
    - `/healthz` for health checks.

- **9.2 – Dockerization**
  - `Dockerfile`:
    - Base image (e.g., `python:3.11-slim`).
    - Install system deps (e.g., for `onnxruntime`, `chromadb`).
    - Copy code and `requirements.txt`, install, expose port 8000.
    - CMD: `uvicorn rag_arxiv_qa.src.api.main:app --host 0.0.0.0 --port 8000`
  - `docker-compose.yml`:
    - API service.
    - (Optional) separate vector DB service or GPU inference service.

- **9.3 – Basic front‑end (optional)**
  - Simple React/Vue/HTML page that:
    - Calls `/query`.
    - Displays answer + citations.

Deliverable: **RAG API running in Docker, reachable via `/query` and `/docs`.**

---

## Phase 10 – CI/CD, Monitoring, and Ops

**Goal:** Automated checks and visibility into runtime behavior.

- **10.1 – CI with GitHub Actions**
  - Workflow on PR / push:
    - `pip install -r requirements.txt`
    - `pytest -q` (or targeted test subset).
    - Optional: `ruff`/`black` for lint/format.

- **10.2 – Runtime logging & metrics**
  - Standardize logging:
    - Use `logging` with structured JSON logs for:
      - Query payload (anonymized).
      - Number of retrieved chunks.
      - Latency (retrieval, rerank, generation).
      - Errors/exceptions.
  - Hook logs into:
    - Cloud logging (if deployed) or local files for debugging.

- **10.3 – Basic monitoring**
  - Track:
    - QPS, latency, error rate.
    - Distribution of retrieved chunk counts and scores (to spot drift).

Deliverable: **A project that looks and behaves like a production RAG service, not just a notebook demo.**

